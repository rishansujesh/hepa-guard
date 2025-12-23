from __future__ import annotations

from sklearn.impute import SimpleImputer
import numpy as np


def build_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", add_indicator=True)


def fit_imputer(imputer: SimpleImputer, X_train) -> SimpleImputer:
    imputer.fit(X_train)
    return imputer


def transform_imputer(imputer: SimpleImputer, X) -> np.ndarray:
    return imputer.transform(X)


def get_imputer_medians(imputer: SimpleImputer, base_feature_names: list[str]) -> dict[str, float]:
    stats = imputer.statistics_.tolist()
    return {name: float(val) for name, val in zip(base_feature_names, stats)}


def get_indicator_indices(imputer: SimpleImputer) -> list[int]:
    indicator = getattr(imputer, "indicator_", None)
    if indicator is None:
        return []
    return list(indicator.features_)


def get_indicator_feature_names(
    base_feature_names: list[str],
    indicator_indices: list[int],
) -> list[str]:
    return [f"{base_feature_names[i]}_missing" for i in indicator_indices]


def get_feature_order(
    base_feature_names: list[str],
    indicator_indices: list[int],
) -> list[str]:
    return list(base_feature_names) + get_indicator_feature_names(
        base_feature_names, indicator_indices
    )
