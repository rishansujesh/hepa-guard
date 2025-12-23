from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def roc_auc(y_true, y_proba) -> float:
    return float(roc_auc_score(y_true, y_proba))


def pr_auc(y_true, y_proba) -> float:
    return float(average_precision_score(y_true, y_proba))


def precision(y_true, y_pred) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def confusion_matrix_at_threshold(y_true, y_proba, threshold: float) -> dict[str, int]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def brier_score(y_true, y_proba) -> float:
    return float(brier_score_loss(y_true, y_proba))


def calibration_curve_data(y_true, y_proba, n_bins: int = 10) -> list[dict[str, float]]:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return [
        {"mean_predicted": float(mp), "fraction_positive": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]


def pick_threshold_for_recall(
    y_true,
    y_proba,
    min_recall: float = 0.85,
) -> tuple[float, list[dict[str, float]]]:
    thresholds = np.linspace(0.0, 1.0, 101)
    table: list[dict[str, float]] = []
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    for thr in thresholds:
        y_pred = (y_proba_arr >= thr).astype(int)
        prec = precision(y_true_arr, y_pred)
        rec = recall(y_true_arr, y_pred)
        f1_val = f1(y_true_arr, y_pred)
        table.append({"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1_val})

    eligible = [row for row in table if row["recall"] >= min_recall]
    if eligible:
        best = max(eligible, key=lambda r: (r["f1"], r["precision"]))
    else:
        best = max(table, key=lambda r: (r["recall"], r["precision"]))

    return best["threshold"], table


def summarize_metrics(y_true, y_proba, threshold: float) -> dict[str, object]:
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    y_pred = (y_proba_arr >= threshold).astype(int)

    return {
        "auroc": roc_auc(y_true_arr, y_proba_arr),
        "pr_auc": pr_auc(y_true_arr, y_proba_arr),
        "precision": precision(y_true_arr, y_pred),
        "recall": recall(y_true_arr, y_pred),
        "f1": f1(y_true_arr, y_pred),
        "brier_score": brier_score(y_true_arr, y_proba_arr),
        "confusion_matrix": confusion_matrix_at_threshold(y_true_arr, y_proba_arr, threshold),
        "threshold": float(threshold),
        "calibration_curve": calibration_curve_data(y_true_arr, y_proba_arr),
    }
