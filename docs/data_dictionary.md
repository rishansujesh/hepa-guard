# HepaGuard Data Dictionary

This dictionary covers the columns in:
- `data/processed/hepaguard_features.parquet`
- `data/processed/hepaguard_features_with_usfli.parquet`

`hepaguard_features_with_usfli.parquet` includes all feature columns plus the derived US-FLI score and label.

## Features

| Column | Description | Units | Allowed range (coarse) | Missingness | Source module / NHANES file |
| --- | --- | --- | --- | --- | --- |
| SEQN | Participant identifier | none | positive integer | none expected | `scripts/build_nhanes_features.py` / P_DEMO |
| age_years | Age at exam | years | 18-120 | low | `scripts/build_nhanes_features.py` / P_DEMO |
| sex_code | Sex (1=Male, 2=Female) | code | 1-2 | low | `scripts/build_nhanes_features.py` / P_DEMO |
| bmi | Body mass index | kg/m^2 | 10-80 | low | `scripts/build_nhanes_features.py` / P_BMX |
| waist_cm | Waist circumference | cm | 50-200 | low | `scripts/build_nhanes_features.py` / P_BMX |
| alt_u_l | Alanine aminotransferase | U/L | 0-300 | low | `scripts/build_nhanes_features.py` / P_BIOPRO |
| ast_u_l | Aspartate aminotransferase | U/L | 0-300 | low | `scripts/build_nhanes_features.py` / P_BIOPRO |
| platelets_1000cells_ul | Platelets | 10^3 cells/uL | 50-1000 | low | `scripts/build_nhanes_features.py` / P_CBC |
| triglycerides_mg_dl | Triglycerides (fasting) | mg/dL | 20-1000 | moderate (fasting subsample) | `scripts/build_nhanes_features.py` / P_TRIGLY |
| hdl_mg_dl | HDL cholesterol | mg/dL | 10-120 | low | `scripts/build_nhanes_features.py` / P_HDL |
| fasting_glucose_mg_dl | Fasting glucose | mg/dL | 40-300 | moderate (fasting subsample) | `scripts/build_nhanes_features.py` / P_GLU |
| alcohol_drinks_per_day | Average drinks per day | drinks/day | 0-15 | moderate | `scripts/build_nhanes_features.py` / P_ALQ |
| sedentary_min_per_day | Minutes sedentary per day | minutes/day | 0-1440 | moderate | `scripts/build_nhanes_features.py` / P_PAQ |

## Label and derived fields

| Column | Description | Units | Allowed range (coarse) | Missingness | Source module / NHANES file |
| --- | --- | --- | --- | --- | --- |
| usfli_score | US-FLI score | score | 0-100 | missing when required inputs absent | `scripts/build_usfli_labels.py` / P_DEMO, P_BMX, P_BIOPRO, P_GLU, P_INS |
| masld_usfli_label | Binary label (1 if usfli_score >= 30, else 0) | label | 0-1 | missing when usfli_score missing | `scripts/build_usfli_labels.py` |

## Leakage and limitations

- The label is derived from US-FLI, which uses age, race/ethnicity, waist circumference, GGT, fasting glucose, and fasting insulin.
- Model features overlap with some inputs (age, waist, fasting glucose in the enhanced set). The feature set does not include fasting insulin, GGT, or race/ethnicity, but overlap still creates label leakage risk.
- US-FLI is a proxy score rather than a biopsy-confirmed diagnosis; predictions should be treated as risk estimation, not clinical diagnosis.
- Fasting labs are only available for a subsample, so enhanced features have more missingness.
