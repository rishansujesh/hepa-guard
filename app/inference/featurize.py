from __future__ import annotations

import math
from typing import Any

from app.schemas import HepaGuardRiskRequest


def map_request_to_base_features(
    req: HepaGuardRiskRequest,
) -> tuple[dict[str, float | None], list[str]]:
    warnings: list[str] = []

    sex_code: float | None
    if req.patient.sex == "male":
        sex_code = 1.0
    elif req.patient.sex == "female":
        sex_code = 2.0
    else:
        sex_code = None
        warnings.append("Sex is other/unknown; imputing sex_code.")

    base = {
        "age_years": req.patient.age_years,
        "sex_code": sex_code,
        "bmi": req.patient.bmi_kg_m2,
        "waist_cm": req.patient.waist_cm,
        "alt_u_l": req.labs.alt_u_l,
        "ast_u_l": req.labs.ast_u_l,
        "platelets_1000cells_ul": req.labs.platelets_10e3_per_uL,
        "hdl_mg_dl": req.labs.hdl_mg_dl,
        "alcohol_drinks_per_day": req.lifestyle.alcohol_drinks_per_day,
        "sedentary_min_per_day": req.lifestyle.sedentary_min_per_day,
        "fasting_glucose_mg_dl": req.labs.fasting_glucose_mg_dl,
        "triglycerides_mg_dl": req.labs.triglycerides_mg_dl,
    }

    return base, warnings


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def build_model_vector(
    feature_order: list[str],
    base: dict[str, float | None],
    preprocessing: dict,
) -> list[float]:
    medians = preprocessing.get("medians", {})
    vector: list[float] = []

    for feature in feature_order:
        if feature.endswith("_missing"):
            base_name = feature[: -len("_missing")]
            is_missing = _is_missing(base.get(base_name))
            vector.append(1.0 if is_missing else 0.0)
            continue

        value = base.get(feature)
        if _is_missing(value):
            if feature not in medians:
                raise KeyError(f"Missing median for feature '{feature}'")
            value = medians[feature]
        vector.append(float(value))

    return vector
