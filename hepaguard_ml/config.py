from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
MIN_RECALL = 0.85

ID_COL = "SEQN"
LABEL_COL = "masld_usfli_label"

CORE_FEATURES = [
    "age_years",
    "sex_code",
    "bmi",
    "waist_cm",
    "alt_u_l",
    "ast_u_l",
    "platelets_1000cells_ul",
    "hdl_mg_dl",
    "alcohol_drinks_per_day",
    "sedentary_min_per_day",
]

ENH_FEATURES = CORE_FEATURES + [
    "fasting_glucose_mg_dl",
    "triglycerides_mg_dl",
]

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "enhanced": ENH_FEATURES,
}

DEFAULT_DATA_PATH = Path("data/processed/hepaguard_features_with_usfli.parquet")
DEFAULT_SPLIT_PATH = Path("data/processed/split_indices.json")
DEFAULT_SPLIT_STATS_PATH = Path("data/processed/split_stats.json")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")
