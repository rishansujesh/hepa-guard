export type RiskLabel = "low" | "medium" | "high";

export interface PatientInfo {
  age_years: number;
  sex: "male" | "female" | "other" | "unknown";
  bmi_kg_m2: number;
  waist_cm: number;
}

export interface LabsInfo {
  alt_u_l: number;
  ast_u_l: number;
  platelets_10e3_per_uL: number;
  hdl_mg_dl: number;
  triglycerides_mg_dl?: number;
  fasting_glucose_mg_dl?: number;
}

export interface LifestyleInfo {
  alcohol_drinks_per_day: number;
  sedentary_min_per_day: number;
}

export interface MetaInfo {
  fasting_state?: string;
  units_version?: string;
  model_preference?: string;
}

export interface ClinicalBriefRequest {
  request_id?: string;
  patient: PatientInfo;
  labs: LabsInfo;
  lifestyle: LifestyleInfo;
  meta?: MetaInfo;
}

export interface RiskCutoffs {
  low_lt: number;
  medium_lt: number;
}

export interface TopFactor {
  feature: string;
  direction: "increases_risk" | "decreases_risk";
  impact: number;
  value: number;
}

export interface ClinicalBriefResponse {
  request_id?: string;
  model_used: string;
  risk_probability: number;
  risk_label: RiskLabel;
  risk_cutoffs: RiskCutoffs;
  top_factors: TopFactor[];
  guideline_next_steps: string | null;
  citations: string[];
  warnings: string[];
  disclaimer: string;
}

export interface ValidationIssue {
  field: string;
  issue: string;
  allowed?: string;
  received?: unknown;
}

export interface ValidationErrorResponse {
  error: "validation_error";
  message: string;
  validation_errors: ValidationIssue[];
}
