#!/usr/bin/env python3
"""
build_usfli_labels.py

Build US-FLI score + binary steatosis label, then merge onto an existing feature table by SEQN.

US-FLI formula:
y = -0.8073*(non_hisp_black) + 0.3458*(mexican_american)
    + 0.0093*age + 0.6151*ln(GGT) + 0.0249*waist_cm
    + 1.1792*ln(insulin) + 0.8242*ln(glucose) - 14.7812
USFLI = exp(y)/(1+exp(y))*100
Steatosis: USFLI >= 30

Important implementation notes:
- Insulin should be in pmol/L. NHANES provides LBDINSI (pmol/L) and LBXIN (uU/mL).
  If only LBXIN is present, convert via: pmol/L = uU/mL * 6.0
- Rows missing any required inputs should have usfli_score = NaN and label = <NA>,
  not label=0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def find_file(raw_dir: Path, stem: str) -> Path:
    """
    Find an NHANES XPT file in raw_dir matching a stem like 'P_DEMO' regardless of case/extension.
    """
    candidates = []
    patterns = [
        f"{stem}.XPT", f"{stem}.xpt",
        f"{stem}_*.XPT", f"{stem}_*.xpt",
        f"*{stem}*.XPT", f"*{stem}*.xpt",
    ]
    for pat in patterns:
        candidates.extend(raw_dir.glob(pat))

    if not candidates:
        raise FileNotFoundError(f"Could not find a file for '{stem}' under: {raw_dir}")

    # Prefer exact match if it exists
    for c in candidates:
        if c.name.lower() == f"{stem.lower()}.xpt":
            return c
    for c in candidates:
        if c.name.lower() == f"{stem.lower()}.xpt".replace(".xpt", ".xpt"):
            return c

    return sorted(candidates)[0]


def read_xpt(path: Path) -> pd.DataFrame:
    # pandas uses "xport" for SAS transport files (XPT)
    try:
        return pd.read_sas(path, format="xport")
    except Exception as e:
        raise ValueError(f"Failed to read XPT file '{path.name}'. Error: {e}") from e


def safe_ln(s: pd.Series) -> pd.Series:
    """ln(x) but returns NaN for non-positive values."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)


def build_usfli_table(raw_dir: Path) -> pd.DataFrame:
    raw_dir = raw_dir.resolve()

    # --- Load needed files ---
    demo_path = find_file(raw_dir, "P_DEMO")
    bmx_path = find_file(raw_dir, "P_BMX")
    biopro_path = find_file(raw_dir, "P_BIOPRO")
    glu_path = find_file(raw_dir, "P_GLU")
    ins_path = find_file(raw_dir, "P_INS")

    demo = read_xpt(demo_path)
    bmx = read_xpt(bmx_path)
    biopro = read_xpt(biopro_path)
    glu = read_xpt(glu_path)
    ins = read_xpt(ins_path)

    # --- DEMO: age + race/eth ---
    demo_cols = ["SEQN"]
    for c in ["RIDAGEYR", "RIDRETH3", "RIDRETH1"]:
        if c in demo.columns:
            demo_cols.append(c)
    demo = demo[demo_cols].copy()

    # Prefer RIDRETH3; fallback to RIDRETH1
    if "RIDRETH3" in demo.columns:
        demo["ridreth"] = pd.to_numeric(demo["RIDRETH3"], errors="coerce")
    elif "RIDRETH1" in demo.columns:
        demo["ridreth"] = pd.to_numeric(demo["RIDRETH1"], errors="coerce")
    else:
        raise KeyError("Neither RIDRETH3 nor RIDRETH1 found in P_DEMO")

    # --- BMX: waist circumference ---
    if "BMXWAIST" not in bmx.columns:
        raise KeyError("BMXWAIST not found in P_BMX (waist circumference)")
    bmx = bmx[["SEQN", "BMXWAIST"]].copy()

    # --- BIOPRO: GGT ---
    if "LBXSGTSI" not in biopro.columns:
        raise KeyError("LBXSGTSI (GGT) not found in P_BIOPRO")
    biopro = biopro[["SEQN", "LBXSGTSI"]].copy()

    # --- GLU: fasting glucose ---
    if "LBXGLU" not in glu.columns:
        raise KeyError("LBXGLU (fasting glucose mg/dL) not found in P_GLU")
    glu = glu[["SEQN", "LBXGLU"]].copy()

    # --- INS: insulin (prefer SI pmol/L) ---
    # Insulin: prefer SI unit LBDINSI (pmol/L); else convert LBXIN (µU/mL) * 6.0
    if "LBDINSI" in ins.columns:
        ins = ins[["SEQN", "LBDINSI"]].copy()
        ins = ins.rename(columns={"LBDINSI": "fasting_insulin_pmol_l"})
    elif "LBXIN" in ins.columns:
        ins = ins[["SEQN", "LBXIN"]].copy()
        ins["fasting_insulin_pmol_l"] = pd.to_numeric(ins["LBXIN"], errors="coerce") * 6.0
        ins = ins[["SEQN", "fasting_insulin_pmol_l"]]
    else:
        raise KeyError("Neither LBDINSI nor LBXIN found in P_INS")

    # --- Merge to compute US-FLI ---
    df = (
        demo.merge(bmx, on="SEQN", how="inner")
            .merge(biopro, on="SEQN", how="inner")
            .merge(glu, on="SEQN", how="inner")
            .merge(ins, on="SEQN", how="inner")
    )

    df = df.rename(
        columns={
            "RIDAGEYR": "age_years",
            "BMXWAIST": "waist_cm",
            "LBXSGTSI": "ggt_u_l",
            "LBXGLU": "fasting_glucose_mg_dl",
        }
    )

    # Race flags for US-FLI:
    # For RIDRETH3: 1=Mexican American, 4=Non-Hispanic Black
    df["mexican_american"] = (df["ridreth"] == 1).astype(int)
    df["non_hisp_black"] = (df["ridreth"] == 4).astype(int)

    # Ensure numeric
    req = ["age_years", "waist_cm", "ggt_u_l", "fasting_glucose_mg_dl", "fasting_insulin_pmol_l"]
    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Valid rows require all inputs present and log-safe (positive)
    valid = (
        df["age_years"].notna()
        & df["waist_cm"].notna() & (df["waist_cm"] > 0)
        & df["ggt_u_l"].notna() & (df["ggt_u_l"] > 0)
        & df["fasting_glucose_mg_dl"].notna() & (df["fasting_glucose_mg_dl"] > 0)
        & df["fasting_insulin_pmol_l"].notna() & (df["fasting_insulin_pmol_l"] > 0)
    )

    # Compute y for all, but we will only materialize score for valid rows
    y = (
        -0.8073 * df["non_hisp_black"]
        + 0.3458 * df["mexican_american"]
        + 0.0093 * df["age_years"]
        + 0.6151 * safe_ln(df["ggt_u_l"])
        + 0.0249 * df["waist_cm"]
        + 1.1792 * safe_ln(df["fasting_insulin_pmol_l"])
        + 0.8242 * safe_ln(df["fasting_glucose_mg_dl"])
        - 14.7812
    )

    # Avoid overflow in exp for extreme y
    y = y.clip(lower=-50, upper=50)

    # Score/label only for valid rows
    df["usfli_score"] = np.nan
    df.loc[valid, "usfli_score"] = (np.exp(y[valid]) / (1.0 + np.exp(y[valid]))) * 100.0

    df["masld_usfli_label"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
    has_score = df["usfli_score"].notna()
    df.loc[has_score, "masld_usfli_label"] = (df.loc[has_score, "usfli_score"] >= 30.0).astype("Int64")

    out_cols = [
        "SEQN",
        "age_years",
        "ridreth",
        "mexican_american",
        "non_hisp_black",
        "waist_cm",
        "ggt_u_l",
        "fasting_glucose_mg_dl",
        "fasting_insulin_pmol_l",
        "usfli_score",
        "masld_usfli_label",
    ]
    return df[out_cols].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--features_path", type=str, default="data/processed/hepaguard_features.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build US-FLI label table
    usfli = build_usfli_table(raw_dir)
    usfli_path_parq = out_dir / "hepaguard_usfli_labels.parquet"
    usfli_path_csv = out_dir / "hepaguard_usfli_labels.csv"
    usfli.to_parquet(usfli_path_parq, index=False)
    usfli.to_csv(usfli_path_csv, index=False)
    print(f"Wrote: {usfli_path_parq} ({len(usfli):,} rows)")
    print(f"Wrote: {usfli_path_csv}")

    # Merge onto your existing feature table
    feat_path = Path(args.features_path)
    if not feat_path.exists():
        print(f"\nERROR: features file not found: {feat_path}", file=sys.stderr)
        sys.exit(1)

    features = pd.read_parquet(feat_path) if feat_path.suffix.lower() == ".parquet" else pd.read_csv(feat_path)
    merged = features.merge(usfli[["SEQN", "usfli_score", "masld_usfli_label"]], on="SEQN", how="left")

    merged_parq = out_dir / "hepaguard_features_with_usfli.parquet"
    merged_csv = out_dir / "hepaguard_features_with_usfli.csv"
    merged.to_parquet(merged_parq, index=False)
    merged.to_csv(merged_csv, index=False)

    print(f"\nWrote: {merged_parq} ({len(merged):,} rows)")
    print(f"Wrote: {merged_csv}")

    labeled = merged["masld_usfli_label"].dropna()
    if len(labeled) > 0:
        pos_rate = float(labeled.mean())
        print(f"\nLabel coverage: {len(labeled):,}/{len(merged):,} ({len(labeled)/len(merged)*100:.1f}%)")
        print(f"Positive rate (US-FLI>=30): {pos_rate*100:.1f}%")
    else:
        print("\nNo labeled rows produced (check that P_INS/P_GLU/P_BIOPRO have matching SEQN rows).")


if __name__ == "__main__":
    main()
