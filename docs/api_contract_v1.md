# HepaGuard API Contract v1 (MVP)

This document defines the request/response contract for HepaGuard’s MVP inference API.

## Request schema (UI → Backend)

### HepaGuardRiskRequest

```json
{
  "request_id": "optional-string",
  "patient": {
    "age_years": 45,
    "sex": "male",
    "bmi_kg_m2": 29.7,
    "waist_cm": 102.4
  },
  "labs": {
    "alt_u_l": 35,
    "ast_u_l": 27,
    "platelets_10e3_per_uL": 250,
    "hdl_mg_dl": 45,
    "triglycerides_mg_dl": 150,
    "fasting_glucose_mg_dl": 105
  },
  "lifestyle": {
    "alcohol_drinks_per_day": 0.0,
    "sedentary_min_per_day": 360
  },
  "meta": {
    "fasting_state": "unknown",
    "units_version": "v1",
    "model_preference": "auto"
  }
}
```

### Required vs optional inputs

Required (Core model minimum):

- `patient.age_years` (18–90)
- `patient.sex` (male|female|other|unknown)
- `patient.bmi_kg_m2` (10–80)
- `patient.waist_cm` (40–200)
- `labs.alt_u_l` (1–1000)
- `labs.ast_u_l` (1–1000)
- `labs.platelets_10e3_per_uL` (50–1000)
- `labs.hdl_mg_dl` (5–200)
- `lifestyle.alcohol_drinks_per_day` (0–20)
- `lifestyle.sedentary_min_per_day` (0–1440)

Optional (Enhanced model inputs):

- `labs.triglycerides_mg_dl` (10–3000)
- `labs.fasting_glucose_mg_dl` (40–600)

Note: The example request includes enhanced fields, but they are optional.

### Validation behavior

- Missing/out-of-range required fields → HTTP 400 with `validation_errors[]`
- Missing enhanced-only fields → run Core model

### Canonical mapping (UI fields → model features)

The UI request uses clinical-friendly field names. The backend maps and transforms them into the trained model’s expected features.

Direct mappings:

- `patient.age_years` → `age_years`
- `patient.waist_cm` → `waist_cm`
- `labs.alt_u_l` → `alt_u_l`
- `labs.ast_u_l` → `ast_u_l`
- `labs.hdl_mg_dl` → `hdl_mg_dl`
- `lifestyle.alcohol_drinks_per_day` → `alcohol_drinks_per_day`
- `lifestyle.sedentary_min_per_day` → `sedentary_min_per_day`

Renamed mappings (same numeric units, different feature name):

- `patient.bmi_kg_m2` → `bmi`
- `labs.platelets_10e3_per_uL` → `platelets_1000cells_ul` (10^3/µL)

Enhanced-only:

- `labs.fasting_glucose_mg_dl` → `fasting_glucose_mg_dl`
- `labs.triglycerides_mg_dl` → `triglycerides_mg_dl`

Sex encoding:

`patient.sex` is mapped to numeric `sex_code`:

- "male" → `sex_code = 1`
- "female" → `sex_code = 2`
- "other" or "unknown" → `sex_code = null` (imputed by backend; see Warnings below)

Missingness indicator features (derived by backend):

The trained model expects additional boolean missingness indicator features (e.g., `bmi_missing`, `ast_u_l_missing`, etc.). The UI does not send these. The backend derives them based on whether a field is missing before imputation, following the packaged preprocessing configuration.

### Units and versioning

All numeric fields are assumed to be provided in the units implied by their names (e.g., `*_mg_dl`, `*_u_l`, `*_cm`).

`meta.units_version` is reserved for future support of alternate units. Only "v1" is supported in the MVP and is currently informational.

### Warnings (non-fatal)

The backend may return warnings for non-fatal conditions (e.g., when "other"/"unknown" sex is imputed, or when enhanced fields are missing and the system falls back to the core model). Warnings do not block inference unless a required field is missing/out of range.

## Response schema (Backend → UI)

### HepaGuardRiskResponse

```json
{
  "request_id": "optional-string",
  "model_used": "core_xgb_v1",
  "risk_probability": 0.72,
  "risk_label": "high",
  "risk_cutoffs": { "low_lt": 0.33, "medium_lt": 0.66 },
  "top_factors": [
    { "feature": "waist_cm", "direction": "increases_risk", "impact": 0.21, "value": 102.4 },
    { "feature": "alt_u_l", "direction": "increases_risk", "impact": 0.16, "value": 35 },
    { "feature": "hdl_mg_dl", "direction": "decreases_risk", "impact": 0.10, "value": 45 }
  ],
  "guideline_next_steps": null,
  "citations": [],
  "warnings": [],
  "disclaimer": "HepaGuard is a prototype CDSS demo. It is not a diagnostic tool or medical device. It does not replace clinical judgment."
}
```

### Notes on response fields

- `risk_cutoffs` in the response is the runtime source of truth for binning. UI clients should not hardcode thresholds.
- `top_factors[]` are computed per-patient at inference time using SHAP-style additive contributions:
  - `impact` is the absolute magnitude of the per-patient contribution (`abs(shap_value)`) in model output space (log-odds).
  - `direction` is derived from the sign of the contribution:
    - positive → `increases_risk`
    - negative → `decreases_risk`
- `guideline_next_steps` and `citations` may be empty/null in the MVP if the guideline retrieval system is unavailable or not integrated yet.

## Error schema

```json
{
  "error": "validation_error",
  "message": "One or more fields are missing or out of allowed range.",
  "validation_errors": [
    { "field": "patient.age_years", "issue": "required" },
    { "field": "labs.alt_u_l", "issue": "out_of_range", "allowed": "1-1000", "received": 5000 }
  ]
}
```

## Risk binning (UI labels)

- low: `risk_probability < risk_cutoffs.low_lt`
- medium: `risk_cutoffs.low_lt <= risk_probability < risk_cutoffs.medium_lt`
- high: `risk_probability >= risk_cutoffs.medium_lt`

(Default cutoffs for v1 are typically `low_lt=0.33`, `medium_lt=0.66`, but clients should use `risk_cutoffs` returned by the backend.)

## Disclaimer

HepaGuard is a prototype clinical decision support demo. It is not a diagnostic tool and is not a medical device. It does not replace clinical judgment. Results may be inaccurate or incomplete. Use only for educational purposes; clinicians should follow official guidelines and consider full patient context.
