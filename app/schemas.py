from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel
from typing_extensions import Literal


class PatientInfo(BaseModel):
    age_years: float
    sex: Literal["male", "female", "other", "unknown"]
    bmi_kg_m2: float
    waist_cm: float


class LabsInfo(BaseModel):
    alt_u_l: float
    ast_u_l: float
    platelets_10e3_per_uL: float
    hdl_mg_dl: float
    triglycerides_mg_dl: Optional[float] = None
    fasting_glucose_mg_dl: Optional[float] = None


class LifestyleInfo(BaseModel):
    alcohol_drinks_per_day: float
    sedentary_min_per_day: float


class MetaInfo(BaseModel):
    fasting_state: Optional[str] = None
    units_version: Optional[str] = None
    model_preference: Optional[str] = None


class HepaGuardRiskRequest(BaseModel):
    request_id: Optional[str] = None
    patient: PatientInfo
    labs: LabsInfo
    lifestyle: LifestyleInfo
    meta: Optional[MetaInfo] = None


class RiskCutoffs(BaseModel):
    low_lt: float
    medium_lt: float


class TopFactor(BaseModel):
    feature: str
    direction: Literal["increases_risk", "decreases_risk"]
    impact: float
    value: float


class HepaGuardRiskResponse(BaseModel):
    request_id: Optional[str] = None
    model_used: str
    risk_probability: float
    risk_label: Literal["low", "medium", "high"]
    risk_cutoffs: RiskCutoffs
    top_factors: List[TopFactor]
    guideline_next_steps: Optional[Any] = None
    citations: List[str]
    warnings: List[str]
    disclaimer: str


class ValidationIssue(BaseModel):
    field: str
    issue: str
    allowed: Optional[str] = None
    received: Optional[Any] = None


class ValidationErrorResponse(BaseModel):
    error: str
    message: str
    validation_errors: List[ValidationIssue]
