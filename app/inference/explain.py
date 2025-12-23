from __future__ import annotations

import xgboost as xgb


def top_factors(
    booster: xgb.Booster,
    feature_order: list[str],
    x: list[float],
    top_k: int = 3,
) -> list[dict]:
    dmatrix = xgb.DMatrix([x], feature_names=feature_order)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contrib_list = contribs[0].tolist()

    if len(contrib_list) == len(feature_order) + 1:
        contrib_list = contrib_list[:-1]

    rows = []
    for feature, contrib, value in zip(feature_order, contrib_list, x):
        direction = "increases_risk" if contrib > 0 else "decreases_risk"
        rows.append(
            {
                "feature": feature,
                "direction": direction,
                "impact": float(abs(contrib)),
                "value": float(value),
            }
        )

    rows.sort(key=lambda r: r["impact"], reverse=True)
    return rows[:top_k]
