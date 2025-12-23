from __future__ import annotations

import json
import os
from typing import Any

import xgboost as xgb

INCH_TO_CM = 2.54
GLUCOSE_MMOL_TO_MGDL = 18.0182
TRIGLYCERIDES_MMOL_TO_MGDL = 88.57
HDL_MMOL_TO_MGDL = 38.67

DISCLAIMER = (
    "HepaGuard is a prototype CDSS demo. It is not a diagnostic tool or medical device. "
    "It does not replace clinical judgment."
)

REQUIRED_RANGES = {
    "patient.age_years": (18, 90),
    "patient.sex": None,
    "patient.bmi_kg_m2": (10, 80),
    "patient.waist_cm": (40, 200),
    "labs.alt_u_l": (1, 1000),
    "labs.ast_u_l": (1, 1000),
    "labs.platelets_10e3_per_uL": (50, 1000),
    "labs.hdl_mg_dl": (5, 200),
    "lifestyle.alcohol_drinks_per_day": (0, 20),
    "lifestyle.sedentary_min_per_day": (0, 1440),
}

OPTIONAL_RANGES = {
    "labs.triglycerides_mg_dl": (10, 3000),
    "labs.fasting_glucose_mg_dl": (40, 600),
}

MODEL_BUNDLES: dict[str, "ModelBundle"] = {}


class ModelBundle:
    def __init__(
        self,
        variant: str,
        booster: xgb.Booster,
        feature_order: list[str],
        preprocessing: dict,
        threshold: dict,
        risk_cutoffs: dict,
        model_card: dict,
    ) -> None:
        self.variant = variant
        self.booster = booster
        self.feature_order = feature_order
        self.preprocessing = preprocessing
        self.threshold = threshold
        self.risk_cutoffs = risk_cutoffs
        self.model_card = model_card


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_bundle(base_dir: str, variant: str) -> ModelBundle:
    variant_dir = os.path.join(base_dir, variant)
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"Model directory not found: {variant_dir}")

    model_path = os.path.join(variant_dir, "model.json")
    feature_path = os.path.join(variant_dir, "feature_order.json")
    preprocessing_path = os.path.join(variant_dir, "preprocessing.json")
    threshold_path = os.path.join(variant_dir, "threshold.json")
    risk_cutoffs_path = os.path.join(variant_dir, "risk_cutoffs.json")
    model_card_path = os.path.join(variant_dir, "model_card.json")

    booster = xgb.Booster()
    booster.load_model(model_path)

    feature_payload = _read_json(feature_path)
    feature_order = feature_payload.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError(f"Invalid feature_order in {feature_path}")

    return ModelBundle(
        variant=variant,
        booster=booster,
        feature_order=feature_order,
        preprocessing=_read_json(preprocessing_path),
        threshold=_read_json(threshold_path),
        risk_cutoffs=_read_json(risk_cutoffs_path),
        model_card=_read_json(model_card_path),
    )


def init() -> None:
    base_dir = os.environ.get("AZUREML_MODEL_DIR", "models")
    models_subdir = os.path.join(base_dir, "models")
    if os.path.isdir(models_subdir):
        base_dir = models_subdir
    MODEL_BUNDLES["core"] = _load_bundle(base_dir, "core")
    MODEL_BUNDLES["enhanced"] = _load_bundle(base_dir, "enhanced")


def _parse_body(raw_data: Any) -> dict:
    if isinstance(raw_data, (bytes, bytearray)):
        return json.loads(raw_data.decode("utf-8"))
    if isinstance(raw_data, str):
        return json.loads(raw_data)
    if isinstance(raw_data, dict):
        return raw_data
    raise ValueError("Invalid request payload; expected JSON object.")


def _get_nested(data: dict, path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_required_fields(payload: dict) -> list[dict]:
    errors: list[dict] = []

    patient = payload.get("patient")
    labs = payload.get("labs")
    lifestyle = payload.get("lifestyle")

    if not isinstance(patient, dict):
        errors.append({"field": "patient", "issue": "required"})
        return errors
    if not isinstance(labs, dict):
        errors.append({"field": "labs", "issue": "required"})
        return errors
    if not isinstance(lifestyle, dict):
        errors.append({"field": "lifestyle", "issue": "required"})
        return errors

    for field, bounds in REQUIRED_RANGES.items():
        value = _get_nested(payload, field)
        if value is None:
            errors.append({"field": field, "issue": "required"})
            continue
        if field == "patient.sex":
            if not isinstance(value, str):
                errors.append({"field": field, "issue": "invalid_type"})
            elif value not in {"male", "female", "other", "unknown"}:
                errors.append({"field": field, "issue": "invalid"})
            continue
        if not _is_number(value):
            errors.append({"field": field, "issue": "invalid_type"})
            continue
        if bounds is None:
            continue
        min_v, max_v = bounds
        if value < min_v or value > max_v:
            errors.append(
                {
                    "field": field,
                    "issue": "out_of_range",
                    "allowed": f"{min_v}-{max_v}",
                    "received": value,
                }
            )

    return errors


def _validate_optional_fields(payload: dict) -> list[dict]:
    errors: list[dict] = []
    for field, bounds in OPTIONAL_RANGES.items():
        value = _get_nested(payload, field)
        if value is None:
            continue
        if not _is_number(value):
            errors.append({"field": field, "issue": "invalid_type"})
            continue
        min_v, max_v = bounds
        if value < min_v or value > max_v:
            errors.append(
                {
                    "field": field,
                    "issue": "out_of_range",
                    "allowed": f"{min_v}-{max_v}",
                    "received": value,
                }
            )
    return errors


def _error_response(validation_errors: list[dict]) -> dict:
    return {
        "error": "validation_error",
        "message": "One or more fields are missing or out of allowed range.",
        "validation_errors": validation_errors,
    }


def _convert_units(payload: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    units_version = meta.get("units_version")

    if units_version is None or units_version == "v1":
        return payload, warnings

    if units_version == "imperial":
        waist = _get_nested(payload, "patient.waist_cm")
        if waist is not None and _is_number(waist):
            payload = json.loads(json.dumps(payload))
            payload["patient"]["waist_cm"] = waist * INCH_TO_CM
            warnings.append("Converted waist_cm from inches to cm (units_version=imperial).")
        return payload, warnings

    if units_version == "si":
        payload = json.loads(json.dumps(payload))
        hdl = _get_nested(payload, "labs.hdl_mg_dl")
        if hdl is not None and _is_number(hdl):
            payload["labs"]["hdl_mg_dl"] = hdl * HDL_MMOL_TO_MGDL
            warnings.append("Converted hdl_mg_dl from mmol/L to mg/dL (units_version=si).")
        glucose = _get_nested(payload, "labs.fasting_glucose_mg_dl")
        if glucose is not None and _is_number(glucose):
            payload["labs"]["fasting_glucose_mg_dl"] = glucose * GLUCOSE_MMOL_TO_MGDL
            warnings.append(
                "Converted fasting_glucose_mg_dl from mmol/L to mg/dL (units_version=si)."
            )
        trig = _get_nested(payload, "labs.triglycerides_mg_dl")
        if trig is not None and _is_number(trig):
            payload["labs"]["triglycerides_mg_dl"] = trig * TRIGLYCERIDES_MMOL_TO_MGDL
            warnings.append(
                "Converted triglycerides_mg_dl from mmol/L to mg/dL (units_version=si)."
            )
        return payload, warnings

    warnings.append("Unknown units_version; assuming canonical v1.")
    return payload, warnings


def _map_features(payload: dict) -> tuple[dict[str, float | None], list[str]]:
    warnings: list[str] = []
    sex_value = _get_nested(payload, "patient.sex")
    sex_code: float | None
    if sex_value == "male":
        sex_code = 1.0
    elif sex_value == "female":
        sex_code = 2.0
    else:
        sex_code = None
        warnings.append("Sex is other/unknown; imputing sex_code.")

    base = {
        "age_years": _get_nested(payload, "patient.age_years"),
        "sex_code": sex_code,
        "bmi": _get_nested(payload, "patient.bmi_kg_m2"),
        "waist_cm": _get_nested(payload, "patient.waist_cm"),
        "alt_u_l": _get_nested(payload, "labs.alt_u_l"),
        "ast_u_l": _get_nested(payload, "labs.ast_u_l"),
        "platelets_1000cells_ul": _get_nested(payload, "labs.platelets_10e3_per_uL"),
        "hdl_mg_dl": _get_nested(payload, "labs.hdl_mg_dl"),
        "alcohol_drinks_per_day": _get_nested(payload, "lifestyle.alcohol_drinks_per_day"),
        "sedentary_min_per_day": _get_nested(payload, "lifestyle.sedentary_min_per_day"),
        "fasting_glucose_mg_dl": _get_nested(payload, "labs.fasting_glucose_mg_dl"),
        "triglycerides_mg_dl": _get_nested(payload, "labs.triglycerides_mg_dl"),
    }

    return base, warnings


def _is_missing(value: Any) -> bool:
    return value is None


def _build_vector(feature_order: list[str], base: dict, preprocessing: dict) -> list[float]:
    medians = preprocessing.get("medians", {})
    vector: list[float] = []

    for feature in feature_order:
        if feature.endswith("_missing"):
            base_name = feature[: -len("_missing")]
            vector.append(1.0 if _is_missing(base.get(base_name)) else 0.0)
            continue

        value = base.get(feature)
        if _is_missing(value):
            if feature not in medians:
                raise KeyError(f"Missing median for feature '{feature}'")
            value = medians[feature]
        vector.append(float(value))

    return vector


def _predict_proba(booster: xgb.Booster, x: list[float]) -> float:
    dmatrix = xgb.DMatrix([x])
    pred = booster.predict(dmatrix)
    prob = float(pred[0])
    if prob < 0.0 or prob > 1.0:
        prob = 1.0 / (1.0 + pow(2.718281828459045, -prob))
    return prob


def _bin_risk(prob: float, risk_cutoffs: dict) -> str:
    low_lt = risk_cutoffs.get("low_lt", risk_cutoffs.get("low"))
    medium_lt = risk_cutoffs.get("medium_lt", risk_cutoffs.get("medium"))
    if low_lt is None or medium_lt is None:
        raise ValueError("Missing risk cutoffs for binning.")

    if prob < float(low_lt):
        return "low"
    if prob < float(medium_lt):
        return "medium"
    return "high"


def _top_factors(
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


def run(raw_data: Any) -> dict:
    payload = _parse_body(raw_data)

    # meta.units_version controls conversion to canonical units before validation.
    payload, unit_warnings = _convert_units(payload)

    errors = _validate_required_fields(payload)
    errors.extend(_validate_optional_fields(payload))
    if errors:
        return _error_response(errors)

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    model_preference = meta.get("model_preference", "auto")
    warnings: list[str] = []

    if model_preference == "enhanced":
        variant = "enhanced"
        if _get_nested(payload, "labs.triglycerides_mg_dl") is None:
            warnings.append("Enhanced requested but triglycerides missing; imputing triglycerides.")
        if _get_nested(payload, "labs.fasting_glucose_mg_dl") is None:
            warnings.append("Enhanced requested but fasting_glucose missing; imputing fasting_glucose.")
    elif model_preference == "core":
        variant = "core"
    else:
        if _get_nested(payload, "labs.triglycerides_mg_dl") is not None and _get_nested(
            payload, "labs.fasting_glucose_mg_dl"
        ) is not None:
            variant = "enhanced"
        else:
            variant = "core"
            warnings.append("Enhanced fields missing; falling back to core model.")

    if variant not in MODEL_BUNDLES:
        raise RuntimeError(f"Model variant '{variant}' not available.")

    bundle = MODEL_BUNDLES[variant]
    base_features, base_warnings = _map_features(payload)
    warnings.extend(base_warnings)
    warnings.extend(unit_warnings)

    x = _build_vector(bundle.feature_order, base_features, bundle.preprocessing)
    prob = _predict_proba(bundle.booster, x)
    risk_label = _bin_risk(prob, bundle.risk_cutoffs)

    risk_cutoffs = {
        "low_lt": bundle.risk_cutoffs.get("low_lt", bundle.risk_cutoffs.get("low")),
        "medium_lt": bundle.risk_cutoffs.get("medium_lt", bundle.risk_cutoffs.get("medium")),
    }

    return {
        "request_id": payload.get("request_id"),
        "model_used": bundle.model_card.get("model_id", bundle.variant),
        "risk_probability": prob,
        "risk_label": risk_label,
        "risk_cutoffs": risk_cutoffs,
        "top_factors": _top_factors(bundle.booster, bundle.feature_order, x, top_k=3),
        "guideline_next_steps": None,
        "citations": [],
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }
