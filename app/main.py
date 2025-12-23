from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.inference.artifacts import load_all_bundles
from app.inference.explain import top_factors
from app.inference.featurize import build_model_vector, map_request_to_base_features
from app.inference.predict import bin_risk, predict_proba
from app.inference.units import convert_request_units
from app.schemas import HepaGuardRiskRequest, HepaGuardRiskResponse, ValidationErrorResponse

app = FastAPI(title="HepaGuard Inference API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.on_event("startup")
def startup_load_bundles() -> None:
    app.state.bundles = load_all_bundles()


def _loc_to_path(loc: tuple) -> str:
    parts = []
    for item in loc:
        if item == "body":
            continue
        parts.append(str(item))
    return ".".join(parts)


def _issue_from_type(type_str: str) -> str:
    if "missing" in type_str:
        return "required"
    if "type_error" in type_str or "parsing" in type_str or type_str.endswith("_type"):
        return "invalid_type"
    return "invalid"


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    validation_errors = []
    for err in exc.errors():
        loc = err.get("loc", ())
        path = _loc_to_path(loc) or "body"
        issue = _issue_from_type(err.get("type", ""))
        validation_errors.append({"field": path, "issue": issue})

    payload = ValidationErrorResponse(
        error="validation_error",
        message="One or more fields are missing or out of allowed range.",
        validation_errors=validation_errors,
    )
    return JSONResponse(status_code=400, content=payload.model_dump())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, object]:
    bundles = getattr(app.state, "bundles", {})
    model_cards = {k: v.model_card for k, v in bundles.items()}
    return {
        "service": "hepaguard-inference",
        "api_contract": "v1",
        "available_models": sorted(model_cards.keys()),
        "model_cards": model_cards,
    }


def _validate_ranges(req: HepaGuardRiskRequest) -> list[dict]:
    errors: list[dict] = []

    required_values = {
        "patient.age_years": req.patient.age_years,
        "patient.sex": req.patient.sex,
        "patient.bmi_kg_m2": req.patient.bmi_kg_m2,
        "patient.waist_cm": req.patient.waist_cm,
        "labs.alt_u_l": req.labs.alt_u_l,
        "labs.ast_u_l": req.labs.ast_u_l,
        "labs.platelets_10e3_per_uL": req.labs.platelets_10e3_per_uL,
        "labs.hdl_mg_dl": req.labs.hdl_mg_dl,
        "lifestyle.alcohol_drinks_per_day": req.lifestyle.alcohol_drinks_per_day,
        "lifestyle.sedentary_min_per_day": req.lifestyle.sedentary_min_per_day,
    }

    for field, bounds in REQUIRED_RANGES.items():
        value = required_values.get(field)
        if value is None:
            errors.append({"field": field, "issue": "required"})
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

    optional_values = {
        "labs.triglycerides_mg_dl": req.labs.triglycerides_mg_dl,
        "labs.fasting_glucose_mg_dl": req.labs.fasting_glucose_mg_dl,
    }
    for field, bounds in OPTIONAL_RANGES.items():
        value = optional_values.get(field)
        if value is None:
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


@app.post("/predict")
def predict(req: HepaGuardRiskRequest):
    # units_version controls conversion to canonical model units before validation.
    req, unit_warnings = convert_request_units(req)

    errors = _validate_ranges(req)
    if errors:
        payload = ValidationErrorResponse(
            error="validation_error",
            message="One or more fields are missing or out of allowed range.",
            validation_errors=errors,
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    bundles = getattr(app.state, "bundles", None)
    if not bundles:
        raise RuntimeError("Model bundles not loaded.")

    warnings: list[str] = []
    model_preference = None
    if req.meta is not None:
        model_preference = req.meta.model_preference

    if model_preference in ("core", "enhanced"):
        variant = model_preference
    else:
        variant = "auto"
        if model_preference not in (None, "auto"):
            warnings.append("Unknown model_preference; defaulting to auto.")

    if variant == "auto":
        if req.labs.triglycerides_mg_dl is not None and req.labs.fasting_glucose_mg_dl is not None:
            variant = "enhanced"
        else:
            variant = "core"
            if req.labs.triglycerides_mg_dl is None or req.labs.fasting_glucose_mg_dl is None:
                warnings.append("Enhanced fields missing; falling back to core model.")
    elif variant == "enhanced":
        if req.labs.triglycerides_mg_dl is None:
            warnings.append("Enhanced requested but triglycerides missing; imputing triglycerides.")
        if req.labs.fasting_glucose_mg_dl is None:
            warnings.append("Enhanced requested but fasting_glucose missing; imputing fasting_glucose.")

    if variant not in bundles:
        raise RuntimeError(f"Model variant '{variant}' not available.")

    bundle = bundles[variant]

    base_features, base_warnings = map_request_to_base_features(req)
    warnings.extend(base_warnings)
    warnings.extend(unit_warnings)

    x = build_model_vector(bundle.feature_order, base_features, bundle.preprocessing)
    prob = predict_proba(bundle.booster, x)
    risk_label = bin_risk(prob, bundle.risk_cutoffs)

    risk_cutoffs = {
        "low_lt": bundle.risk_cutoffs.get("low_lt", bundle.risk_cutoffs.get("low")),
        "medium_lt": bundle.risk_cutoffs.get("medium_lt", bundle.risk_cutoffs.get("medium")),
    }

    factors = top_factors(bundle.booster, bundle.feature_order, x, top_k=3)

    response = HepaGuardRiskResponse(
        request_id=req.request_id,
        model_used=bundle.model_card.get("model_id", bundle.variant),
        risk_probability=prob,
        risk_label=risk_label,
        risk_cutoffs=risk_cutoffs,
        top_factors=factors,
        guideline_next_steps=None,
        citations=[],
        warnings=warnings,
        disclaimer=DISCLAIMER,
    )
    return response
