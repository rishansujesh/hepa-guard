#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from hepaguard_ml.config import (
    FEATURE_SETS,
    ID_COL,
    LABEL_COL,
    N_TOP_FEATURES,
    RANDOM_SEED,
)
from hepaguard_ml.data import (
    build_xy,
    compute_dataset_fingerprint,
    filter_labeled,
    load_dataset,
    make_splits,
)
from hepaguard_ml.preprocess import (
    build_preprocessor,
    get_final_feature_names,
    get_imputer_statistics,
    get_indicator_feature_names,
)
from hepaguard_ml.models import (
    predict_proba,
    train_logreg,
    train_rf,
    train_xgb,
)
from hepaguard_ml.eval import pick_threshold_for_recall, summarize_metrics
from hepaguard_ml.shap_utils import compute_shap_importance_xgb, select_top_features
from xgboost import XGBClassifier


DEFAULT_RISK_BINS = {"low": 0.33, "medium": 0.66}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_path",
        type=str,
        default="data/processed/hepaguard_features_with_usfli.parquet",
    )
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--report_dir", type=str, default="reports")
    ap.add_argument("--n_top_features", type=int, default=N_TOP_FEATURES)
    return ap.parse_args()


def ensure_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_xgb_model(seed: int, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def save_xgb_artifacts(
    out_dir: Path,
    model: XGBClassifier,
    base_feature_names: list[str],
    final_feature_names: list[str],
    preprocessor,
    threshold: float,
    metrics: dict,
    dataset_hash: str,
    row_counts: dict,
    selected_feature_names: list[str] | None = None,
    git_commit: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.json"
    model.get_booster().save_model(model_path)

    write_json(out_dir / "feature_list.json", {"features": final_feature_names})

    preprocess_config = {
        "strategy": "median",
        "add_indicator": True,
        "base_features": base_feature_names,
        "imputer_medians": get_imputer_statistics(preprocessor),
        "indicator_features": get_indicator_feature_names(preprocessor, base_feature_names),
        "final_feature_names": final_feature_names,
    }
    if selected_feature_names is not None:
        preprocess_config["selected_feature_names"] = selected_feature_names

    write_json(out_dir / "preprocess_config.json", preprocess_config)

    threshold_config = {
        "threshold": float(threshold),
        "risk_bins": DEFAULT_RISK_BINS,
    }
    write_json(out_dir / "threshold_config.json", threshold_config)

    model_card = {
        "dataset_hash": dataset_hash,
        "row_counts": row_counts,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }
    write_json(out_dir / "model_card.json", model_card)


def run_variant(
    variant: str,
    feature_cols: list[str],
    df: pd.DataFrame,
    out_root: Path,
    report_root: Path,
    n_top_features: int,
    git_commit: str | None,
) -> None:
    ensure_features(df, feature_cols)

    labeled_df = filter_labeled(df, LABEL_COL)
    train_df, val_df, test_df = make_splits(labeled_df, RANDOM_SEED)

    X_train, y_train = build_xy(train_df, feature_cols, LABEL_COL)
    X_val, y_val = build_xy(val_df, feature_cols, LABEL_COL)
    X_test, y_test = build_xy(test_df, feature_cols, LABEL_COL)

    dataset_hash = compute_dataset_fingerprint(
        labeled_df,
        [ID_COL, LABEL_COL] + feature_cols,
    )
    row_counts = {
        "total": int(len(df)),
        "labeled": int(len(labeled_df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }

    # Baselines
    logreg = train_logreg(X_train, y_train, build_preprocessor(feature_cols))
    logreg_proba = predict_proba(logreg, X_test)
    logreg_metrics = summarize_metrics(y_test, logreg_proba, threshold=0.5)

    rf = train_rf(X_train, y_train, build_preprocessor(feature_cols))
    rf_proba = predict_proba(rf, X_test)
    rf_metrics = summarize_metrics(y_test, rf_proba, threshold=0.5)

    # XGBoost full
    xgb_model, xgb_preprocessor = train_xgb(
        X_train, y_train, build_preprocessor(feature_cols), seed=RANDOM_SEED
    )

    X_train_proc = xgb_preprocessor.transform(X_train)
    X_val_proc = xgb_preprocessor.transform(X_val)
    X_test_proc = xgb_preprocessor.transform(X_test)

    val_proba = predict_proba(xgb_model, X_val_proc)
    threshold_full, threshold_table_full = pick_threshold_for_recall(
        y_val, val_proba, min_recall=0.85
    )

    test_proba_full = predict_proba(xgb_model, X_test_proc)
    metrics_full = summarize_metrics(y_test, test_proba_full, threshold_full)

    full_feature_names = get_final_feature_names(xgb_preprocessor, feature_cols)

    # SHAP importance
    combined = pd.concat([train_df, val_df], axis=0)
    sample_n = min(len(combined), 1000)
    sample = combined.sample(sample_n, random_state=RANDOM_SEED)
    X_sample_proc = xgb_preprocessor.transform(sample[feature_cols])

    importance_df = compute_shap_importance_xgb(
        xgb_model, X_sample_proc, full_feature_names
    )
    top_features = select_top_features(importance_df, n=n_top_features)

    # XGBoost top features
    idx_map = {name: i for i, name in enumerate(full_feature_names)}
    selected_indices = [idx_map[name] for name in top_features if name in idx_map]
    if not selected_indices:
        raise ValueError("No top features selected; check SHAP feature names.")

    y_train_arr = np.asarray(y_train)
    pos = float((y_train_arr == 1).sum())
    neg = float((y_train_arr == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb_top = make_xgb_model(RANDOM_SEED, scale_pos_weight)
    xgb_top.fit(X_train_proc[:, selected_indices], y_train_arr)

    val_proba_top = predict_proba(xgb_top, X_val_proc[:, selected_indices])
    threshold_top, threshold_table_top = pick_threshold_for_recall(
        y_val, val_proba_top, min_recall=0.85
    )

    test_proba_top = predict_proba(xgb_top, X_test_proc[:, selected_indices])
    metrics_top = summarize_metrics(y_test, test_proba_top, threshold_top)

    # Save artifacts
    variant_dir = out_root / variant
    full_dir = variant_dir / "xgb_full"
    top_dir = variant_dir / "xgb_top_features"

    save_xgb_artifacts(
        full_dir,
        xgb_model,
        feature_cols,
        full_feature_names,
        xgb_preprocessor,
        threshold_full,
        metrics_full,
        dataset_hash,
        row_counts,
        git_commit=git_commit,
    )

    save_xgb_artifacts(
        top_dir,
        xgb_top,
        feature_cols,
        top_features,
        xgb_preprocessor,
        threshold_top,
        metrics_top,
        dataset_hash,
        row_counts,
        selected_feature_names=top_features,
        git_commit=git_commit,
    )

    report = {
        "variant": variant,
        "feature_cols": feature_cols,
        "dataset_hash": dataset_hash,
        "row_counts": row_counts,
        "baselines": {
            "logreg": logreg_metrics,
            "random_forest": rf_metrics,
        },
        "xgb_full": {
            "metrics": metrics_full,
            "threshold": threshold_full,
            "threshold_table": threshold_table_full,
        },
        "xgb_top_features": {
            "metrics": metrics_top,
            "threshold": threshold_top,
            "threshold_table": threshold_table_top,
            "selected_features": top_features,
        },
        "comparison": {
            "full": metrics_full,
            "top_features": metrics_top,
        },
    }

    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{variant}_report.json"
    write_json(report_path, report)

    importance_path = report_root / f"{variant}_shap_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print(f"Saved artifacts for {variant}: {variant_dir}")
    print(f"Report: {report_path}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_root = Path(args.out_dir)
    report_root = Path(args.report_dir)

    df = load_dataset(data_path)
    git_commit = get_git_commit()

    for variant, feature_cols in FEATURE_SETS.items():
        run_variant(
            variant,
            feature_cols,
            df,
            out_root,
            report_root,
            args.n_top_features,
            git_commit,
        )


if __name__ == "__main__":
    main()
