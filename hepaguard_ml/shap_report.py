from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_importance(
    model,
    X_sample,
    feature_names: list[str],
) -> pd.DataFrame:
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    return importance.reset_index(drop=True)
