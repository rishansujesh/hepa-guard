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
from xgboost import XGBClassifier

from .config import MIN_RECALL, RANDOM_SEED


def train_xgb(
    X_train,
    y_train,
    seed: int = RANDOM_SEED,
) -> XGBClassifier:
    y_arr = np.asarray(y_train)
    pos = float((y_arr == 1).sum())
    neg = float((y_arr == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
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
    model.fit(X_train, y_arr)
    return model


def predict_proba(model: XGBClassifier, X) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba
    return proba[:, 1]


def tune_threshold(
    y_true,
    y_proba,
    min_recall: float = MIN_RECALL,
) -> tuple[float, str]:
    thresholds = np.linspace(0.0, 1.0, 101)
    best = None
    best_any = None

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        any_candidate = (rec, f1_val, prec, thr)
        if best_any is None or any_candidate > best_any:
            best_any = any_candidate

        if rec >= min_recall:
            candidate = (f1_val, prec, thr)
            if best is None or candidate > best:
                best = candidate

    if best is None and best_any is None:
        return 0.5, "fallback"

    if best is not None:
        _, _, thr = best
        return float(thr), "max_f1_given_recall"

    rec, _, _, thr = best_any
    return float(thr), "max_recall_fallback"


def evaluate_metrics(y_true, y_proba, threshold: float) -> dict[str, object]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
