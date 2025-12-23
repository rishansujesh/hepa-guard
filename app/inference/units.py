from __future__ import annotations

import copy

from app.schemas import HepaGuardRiskRequest

INCH_TO_CM = 2.54
GLUCOSE_MMOL_TO_MGDL = 18.0182
TRIGLYCERIDES_MMOL_TO_MGDL = 88.57
HDL_MMOL_TO_MGDL = 38.67


def _copy_request(req: HepaGuardRiskRequest) -> HepaGuardRiskRequest:
    if hasattr(req, "model_copy"):
        return req.model_copy(deep=True)
    return req.copy(deep=True)


def convert_request_units(
    req: HepaGuardRiskRequest,
) -> tuple[HepaGuardRiskRequest, list[str]]:
    warnings: list[str] = []
    units_version = req.meta.units_version if req.meta is not None else None

    if units_version is None or units_version == "v1":
        return req, warnings

    converted = _copy_request(req)

    if units_version == "imperial":
        if converted.patient.waist_cm is not None:
            converted.patient.waist_cm = converted.patient.waist_cm * INCH_TO_CM
            warnings.append(
                "Converted waist_cm from inches to cm (units_version=imperial)."
            )
        return converted, warnings

    if units_version == "si":
        if converted.labs.hdl_mg_dl is not None:
            converted.labs.hdl_mg_dl = converted.labs.hdl_mg_dl * HDL_MMOL_TO_MGDL
            warnings.append(
                "Converted hdl_mg_dl from mmol/L to mg/dL (units_version=si)."
            )
        if converted.labs.fasting_glucose_mg_dl is not None:
            converted.labs.fasting_glucose_mg_dl = (
                converted.labs.fasting_glucose_mg_dl * GLUCOSE_MMOL_TO_MGDL
            )
            warnings.append(
                "Converted fasting_glucose_mg_dl from mmol/L to mg/dL (units_version=si)."
            )
        if converted.labs.triglycerides_mg_dl is not None:
            converted.labs.triglycerides_mg_dl = (
                converted.labs.triglycerides_mg_dl * TRIGLYCERIDES_MMOL_TO_MGDL
            )
            warnings.append(
                "Converted triglycerides_mg_dl from mmol/L to mg/dL (units_version=si)."
            )
        return converted, warnings

    warnings.append("Unknown units_version; assuming canonical v1.")
    return req, warnings
