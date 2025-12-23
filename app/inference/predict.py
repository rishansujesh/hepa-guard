from __future__ import annotations

import math

import xgboost as xgb


def predict_proba(booster: xgb.Booster, x: list[float]) -> float:
    dmatrix = xgb.DMatrix([x])
    pred = booster.predict(dmatrix)
    prob = float(pred[0])
    if prob < 0.0 or prob > 1.0:
        prob = 1.0 / (1.0 + math.exp(-prob))
    return prob


def bin_risk(prob: float, risk_cutoffs: dict) -> str:
    low_lt = risk_cutoffs.get("low_lt", risk_cutoffs.get("low"))
    medium_lt = risk_cutoffs.get("medium_lt", risk_cutoffs.get("medium"))
    if low_lt is None or medium_lt is None:
        raise ValueError("Missing risk cutoffs for binning.")

    if prob < float(low_lt):
        return "low"
    if prob < float(medium_lt):
        return "medium"
    return "high"
