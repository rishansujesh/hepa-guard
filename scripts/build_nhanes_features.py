#!/usr/bin/env python3
"""
build_nhanes_features.py

Merges NHANES 2017–March 2020 pre-pandemic "P_" XPT files into a single
feature table for HepaGuard (raw features only).

Expected input files in data/raw/ (extension case-insensitive):
  P_DEMO.XPT, P_BMX.XPT, P_BIOPRO.XPT, P_CBC.XPT, P_HDL.XPT,
  P_TRIGLY.XPT, P_GLU.XPT, P_ALQ.XPT, P_PAQ.XPT

Outputs:
  data/processed/hepaguard_features.parquet
  data/processed/hepaguard_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Questionnaire missing/refused codes
MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}


# ---------- Helpers ----------

def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def replace_missing_codes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: np.nan if pd.notna(x) and float(x) in MISSING_CODES else x)
    return df

def ensure_exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.name} (expected in {p.parent})")

def resolve_case_insensitive(p: Path) -> Path:
    if p.exists():
        return p
    parent = p.parent
    target = p.name.lower()
    for cand in parent.glob("*"):
        if cand.is_file() and cand.name.lower() == target:
            return cand
    # try extension-only mismatch
    if p.suffix.lower() == ".xpt":
        alt = p.with_suffix(".XPT")
        if alt.exists():
            return alt
    if p.suffix == ".XPT":
        alt = p.with_suffix(".xpt")
        if alt.exists():
            return alt
    return p


# ---------- Main pipeline ----------

def build_features(raw_dir: Path) -> pd.DataFrame:
    paths = {
        "DEMO": raw_dir / "P_DEMO.XPT",
        "BMX": raw_dir / "P_BMX.XPT",
        "BIOPRO": raw_dir / "P_BIOPRO.XPT",
        "CBC": raw_dir / "P_CBC.XPT",
        "HDL": raw_dir / "P_HDL.XPT",
        "TRIGLY": raw_dir / "P_TRIGLY.XPT",
        "GLU": raw_dir / "P_GLU.XPT",
        "ALQ": raw_dir / "P_ALQ.XPT",
        "PAQ": raw_dir / "P_PAQ.XPT",
    }

    for k, p in list(paths.items()):
        paths[k] = resolve_case_insensitive(p)
        ensure_exists(paths[k])

    # --- Load + select minimal columns ---

    demo = read_xpt(paths["DEMO"])[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
    bmx = read_xpt(paths["BMX"])[["SEQN", "BMXBMI", "BMXWAIST"]].copy()

    # ALT/AST
    biopro = read_xpt(paths["BIOPRO"])[["SEQN", "LBXSATSI", "LBXSASSI"]].copy()

    # Platelets
    cbc = read_xpt(paths["CBC"])[["SEQN", "LBXPLTSI"]].copy()

    # HDL (your cycle uses LBDHDD)
    hdl_raw = read_xpt(paths["HDL"])
    if "LBDHDD" in hdl_raw.columns:
        hdl = hdl_raw[["SEQN", "LBDHDD"]].rename(columns={"LBDHDD": "LBXHDD"}).copy()
    elif "LBXHDD" in hdl_raw.columns:
        hdl = hdl_raw[["SEQN", "LBXHDD"]].copy()
    else:
        raise KeyError(f"HDL column not found. Available: {list(hdl_raw.columns)}")

    # Triglycerides, Glucose
    trig = read_xpt(paths["TRIGLY"])[["SEQN", "LBXTR"]].copy()
    glu = read_xpt(paths["GLU"])[["SEQN", "LBXGLU"]].copy()

    # Alcohol frequency + avg drinks/day
    alq = read_xpt(paths["ALQ"])[["SEQN", "ALQ121", "ALQ130"]].copy()

    # Sedentary minutes/day
    paq = read_xpt(paths["PAQ"])[["SEQN", "PAD680"]].copy()

    # Questionnaire missing/refused codes
    alq = replace_missing_codes(alq, ["ALQ121", "ALQ130"])
    paq = replace_missing_codes(paq, ["PAD680"])

    # --- Merge on SEQN (left join from DEMO) ---
    df = (
        demo.merge(bmx, on="SEQN", how="left")
            .merge(biopro, on="SEQN", how="left")
            .merge(cbc, on="SEQN", how="left")
            .merge(hdl, on="SEQN", how="left")
            .merge(trig, on="SEQN", how="left")
            .merge(glu, on="SEQN", how="left")
            .merge(alq, on="SEQN", how="left")
            .merge(paq, on="SEQN", how="left")
    )

    # --- Coerce numeric types ---
    num_cols = [
        "RIDAGEYR","RIAGENDR","BMXBMI","BMXWAIST",
        "LBXSATSI","LBXSASSI","LBXPLTSI",
        "LBXHDD","LBXTR","LBXGLU",
        "ALQ121","ALQ130","PAD680"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # Adults only (ALQ/PAQ intended for adults)
    df = df[df["RIDAGEYR"] >= 18].copy()

    # --- Fix tiny float artifacts that represent 0 ---
    # (You observed 5.397605e-79 which is essentially 0.)
    for c in ["ALQ121", "ALQ130", "PAD680"]:
        if c in df.columns:
            df.loc[df[c].notna() & (df[c].abs() < 1e-6), c] = 0

    # Sedentary sanity: keep 0 as valid, drop negatives and >24h
    df.loc[df["PAD680"] > 1440, "PAD680"] = np.nan
    df.loc[df["PAD680"] < 0, "PAD680"] = np.nan

    # Alcohol logic:
    # ALQ121 is a frequency category; 0 means "never in last 12 months"
    df.loc[df["ALQ121"] == 0, "ALQ130"] = 0

    # ALQ130 can be 15 = "15 or more"; keep 0..15
    df.loc[(df["ALQ130"] < 0) | (df["ALQ130"] > 15), "ALQ130"] = np.nan

    # Sex should be 1/2
    df.loc[~df["RIAGENDR"].isin([1, 2]), "RIAGENDR"] = np.nan

    # Platelets plausible range filter (thousand/µL)
    df.loc[(df["LBXPLTSI"] < 50) | (df["LBXPLTSI"] > 1000), "LBXPLTSI"] = np.nan

    # --- Rename columns into your model schema ---
    out = df.rename(columns={
        "RIDAGEYR": "age_years",
        "RIAGENDR": "sex_code",          # 1=Male, 2=Female
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_cm",
        "LBXSATSI": "alt_u_l",
        "LBXSASSI": "ast_u_l",
        "LBXPLTSI": "platelets_1000cells_ul",
        "LBXHDD": "hdl_mg_dl",
        "LBXTR": "triglycerides_mg_dl",
        "LBXGLU": "fasting_glucose_mg_dl",
        "ALQ130": "alcohol_drinks_per_day",
        "PAD680": "sedentary_min_per_day",
    })

    # Drop helper column ALQ121
    out = out.drop(columns=["ALQ121"], errors="ignore")

    # Final column order (only what you need)
    keep = [
        "SEQN",
        "age_years",
        "sex_code",
        "bmi",
        "waist_cm",
        "alt_u_l",
        "ast_u_l",
        "platelets_1000cells_ul",
        "triglycerides_mg_dl",
        "hdl_mg_dl",
        "fasting_glucose_mg_dl",
        "alcohol_drinks_per_day",
        "sedentary_min_per_day",
    ]
    out = out[keep]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing NHANES P_*.XPT files")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(raw_dir)

    parquet_path = out_dir / "hepaguard_features.parquet"
    csv_path = out_dir / "hepaguard_features.csv"

    features.to_parquet(parquet_path, index=False)
    features.to_csv(csv_path, index=False)

    print(f"Wrote: {parquet_path} ({len(features):,} rows)")
    print(f"Wrote: {csv_path}")

    miss = features.isna().mean().sort_values(ascending=False)
    print("\nMissingness (%):")
    print((miss * 100).round(1).to_string())


if __name__ == "__main__":
    main()
