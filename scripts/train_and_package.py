#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pandas as pd
import numpy as np

def to_jsonable(x):
    """Recursively convert numpy/pandas scalars/arrays to JSON-serializable Python types."""
    # dict
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    # list/tuple
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    # pathlib
    if isinstance(x, Path):
        return str(x)

    # pandas NA
    if x is pd.NA:
        return None

    # numpy scalars (int64/float64/bool_)
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # pandas Timestamp etc.
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()

    return x

from hepaguard_ml.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_STATS_PATH,
    FEATURE_SETS,
    ID_COL,
    LABEL_COL,
    MIN_RECALL,
)
from hepaguard_ml.data import compute_dataset_fingerprint, filter_labeled, load_dataset
from hepaguard_ml.preprocess import (
    build_imputer,
    fit_imputer,
    get_feature_order,
    get_imputer_medians,
    get_indicator_feature_names,
    get_indicator_indices,
    transform_imputer,
)
from hepaguard_ml.shap_report import compute_shap_importance
from hepaguard_ml.splits import (
    apply_split_indices,
    compute_split_stats,
    load_split_indices,
    make_split_indices,
    save_split_indices,
)
from hepaguard_ml.train import evaluate_metrics, predict_proba, train_xgb, tune_threshold


RISK_CUTOFFS = {"low": 0.33, "medium": 0.66}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--splits_path", type=str, default=str(DEFAULT_SPLIT_PATH))
    ap.add_argument("--split_stats_path", type=str, default=str(DEFAULT_SPLIT_STATS_PATH))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--report_dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    ap.add_argument("--min_recall", type=float, default=MIN_RECALL)
    ap.add_argument("--max_shap_rows", type=int, default=1000)
    return ap.parse_args()


def ensure_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def ensure_splits(
    labeled: pd.DataFrame,
    splits_path: Path,
    stats_path: Path,
) -> dict[str, list[int]]:
    if splits_path.exists():
        return load_split_indices(splits_path)

    split_indices = make_split_indices(labeled)
    save_split_indices(splits_path, split_indices)
    split_stats = compute_split_stats(labeled, split_indices)
    stats_payload = {
        "overall": {
            "labeled_rows": int(len(labeled)),
            "positive_rate": float(labeled[LABEL_COL].mean()) if len(labeled) else 0.0,
        },
        "splits": split_stats,
    }
    write_json(stats_path, stats_payload)
    return split_indices


def save_artifacts(
    out_dir: Path,
    model,
    base_features: list[str],
    feature_order: list[str],
    imputer,
    threshold: float,
    selection_method: str,
    min_recall: float,
    metrics: dict,
    dataset_fingerprint: str,
    rows: dict,
    git_hash: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.json"
    model.get_booster().save_model(model_path)

    write_json(out_dir / "feature_order.json", {"feature_order": feature_order})

    indicator_indices = get_indicator_indices(imputer)
    indicator_features = get_indicator_feature_names(base_features, indicator_indices)

    preprocessing_payload = {
        "strategy": "median",
        "add_indicator": True,
        "base_features": base_features,
        "medians": get_imputer_medians(imputer, base_features),
        "indicator_indices": indicator_indices,
        "indicator_features": indicator_features,
    }
    write_json(out_dir / "preprocessing.json", preprocessing_payload)

    threshold_payload = {
        "threshold_used": float(threshold),
        "min_recall": float(min_recall),
        "selection_method": selection_method,
    }
    write_json(out_dir / "threshold.json", threshold_payload)

    write_json(out_dir / "risk_cutoffs.json", RISK_CUTOFFS)

    model_card = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "rows": rows,
        "metrics": metrics,
    }
    write_json(out_dir / "model_card.json", model_card)


def run_variant(
    name: str,
    feature_cols: list[str],
    labeled_df: pd.DataFrame,
    split_indices: dict[str, list[int]],
    out_root: Path,
    report_root: Path,
    min_recall: float,
    max_shap_rows: int,
    dataset_fingerprint: str,
    rows: dict,
    git_hash: str | None,
) -> None:
    ensure_features(labeled_df, feature_cols)
    train_df, val_df, test_df = apply_split_indices(labeled_df, split_indices)

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COL].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL].astype(int)

    imputer = fit_imputer(build_imputer(), X_train)
    X_train_proc = transform_imputer(imputer, X_train)
    X_val_proc = transform_imputer(imputer, X_val)
    X_test_proc = transform_imputer(imputer, X_test)

    model = train_xgb(X_train_proc, y_train)

    val_proba = predict_proba(model, X_val_proc)
    threshold, selection_method = tune_threshold(y_val.to_numpy(), val_proba, min_recall)

    test_proba = predict_proba(model, X_test_proc)
    metrics = evaluate_metrics(y_test.to_numpy(), test_proba, threshold)

    indicator_indices = get_indicator_indices(imputer)
    feature_order = get_feature_order(feature_cols, indicator_indices)

    out_dir = out_root / name
    save_artifacts(
        out_dir=out_dir,
        model=model,
        base_features=feature_cols,
        feature_order=feature_order,
        imputer=imputer,
        threshold=threshold,
        selection_method=selection_method,
        min_recall=min_recall,
        metrics=metrics,
        dataset_fingerprint=dataset_fingerprint,
        rows=rows,
        git_hash=git_hash,
    )

    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{name}_metrics.json"
    write_json(report_path, {"metrics": metrics, "threshold": threshold})

    sample_df = pd.concat([train_df, val_df], axis=0)
    sample_n = min(len(sample_df), max_shap_rows)
    sample_df = sample_df.sample(sample_n, random_state=42)
    X_sample_proc = transform_imputer(imputer, sample_df[feature_cols])

    shap_df = compute_shap_importance(model, X_sample_proc, feature_order)
    shap_path = report_root / f"{name}_shap_importance.csv"
    shap_df.to_csv(shap_path, index=False)

    print(f"Saved model artifacts: {out_dir}")
    print(f"Saved report: {report_path}")
    print(f"Saved SHAP importance: {shap_path}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    splits_path = Path(args.splits_path)
    split_stats_path = Path(args.split_stats_path)
    out_root = Path(args.out_dir)
    report_root = Path(args.report_dir)

    df = load_dataset(data_path)
    labeled = filter_labeled(df, LABEL_COL)

    split_indices = ensure_splits(labeled, splits_path, split_stats_path)

    rows = {
        "total": int(len(df)),
        "labeled": int(len(labeled)),
        "train": int(len(labeled[labeled[ID_COL].isin(split_indices.get("train", []))])),
        "val": int(len(labeled[labeled[ID_COL].isin(split_indices.get("val", []))])),
        "test": int(len(labeled[labeled[ID_COL].isin(split_indices.get("test", []))])),
    }

    dataset_fingerprint = compute_dataset_fingerprint(
        labeled,
        [ID_COL, LABEL_COL] + FEATURE_SETS["enhanced"],
    )
    git_hash = get_git_hash()

    for name, feature_cols in FEATURE_SETS.items():
        run_variant(
            name=name,
            feature_cols=feature_cols,
            labeled_df=labeled,
            split_indices=split_indices,
            out_root=out_root,
            report_root=report_root,
            min_recall=args.min_recall,
            max_shap_rows=args.max_shap_rows,
            dataset_fingerprint=dataset_fingerprint,
            rows=rows,
            git_hash=git_hash,
        )


if __name__ == "__main__":
    main()
