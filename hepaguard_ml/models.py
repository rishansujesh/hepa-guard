from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from .config import RANDOM_SEED


def train_logreg(
    X_train,
    y_train,
    preprocessor: Pipeline,
) -> Pipeline:
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_rf(
    X_train,
    y_train,
    preprocessor: Pipeline,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgb(
    X_train,
    y_train,
    preprocessor: Pipeline,
    seed: int = RANDOM_SEED,
) -> tuple[XGBClassifier, Pipeline]:
    X_proc = preprocessor.fit_transform(X_train)
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
    model.fit(X_proc, y_arr)
    return model, preprocessor


def predict_proba(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba.squeeze()
