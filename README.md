# HepaGuard

**Explainable AI clinical decision support for earlier MASLD risk recognition.**

HepaGuard is a clinical decision support system designed to help primary care clinicians identify patterns associated with **Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD)** that may be easy to overlook during routine care.

Rather than relying on a single abnormal laboratory value, HepaGuard looks across a patient's metabolic, laboratory, anthropometric, and lifestyle data to produce:

- a **personalized MASLD risk probability**
- the **top three factors driving that prediction**
- **guideline-grounded next-step considerations**
- citations back to the clinical evidence used to generate those recommendations

The goal is not to diagnose liver disease or replace physician judgment. HepaGuard acts as a **second set of eyes**: surfacing patterns that may deserve additional attention and giving clinicians transparent evidence they can evaluate themselves.

> Built for the **Microsoft Imagine Cup 2026**.

---

## Why HepaGuard?

MASLD is closely associated with obesity, insulin resistance, dyslipidemia, and other common metabolic conditions. Yet identifying risk early can be difficult in primary care.

A patient may have:

- relatively unremarkable ALT and AST
- no heavy alcohol use
- moderately elevated glucose
- increased waist circumference
- borderline triglycerides
- significant sedentary behavior

None of these signals necessarily tells the whole story individually.

**The combination can.**

HepaGuard was built around that idea.

Instead of presenting another isolated metric, it attempts to connect these signals into an interpretable risk assessment that a clinician can use as an additional input to their own decision-making.

---

# How It Works

```text
Patient Data
    │
    ▼
Input Validation + Preprocessing
    │
    ▼
Azure Machine Learning
Custom XGBoost Risk Model
    │
    ├────► MASLD Risk Probability
    │
    └────► Top 3 SHAP Risk Drivers
                    │
                    ▼
             Azure Functions
              Orchestration
                    │
                    ▼
        Patient + ML Risk Context
                    │
                    ▼
         Azure OpenAI Embedding
                    │
                    ▼
          Azure AI Search
       Guideline Retrieval (RAG)
                    │
                    ▼
            Azure OpenAI
      Grounded Recommendation
                    │
                    ▼
            Clinical Brief
```

A single patient assessment therefore combines **predictive ML, explainable AI, retrieval-augmented generation, and clinical guideline retrieval** rather than asking a general-purpose LLM to reason about the patient from scratch.

---

# 1. Clinical Inputs

HepaGuard accepts information commonly available during primary care evaluation.

### Patient

- Age
- Sex
- BMI
- Waist circumference

### Laboratory values

**Core model**
- ALT
- AST
- Platelet count
- HDL

**Enhanced model**
- Fasting glucose
- Triglycerides

### Lifestyle

- Alcohol consumption
- Sedentary time

The backend also supports unit normalization, input validation, missing-data handling, and automatic selection between the Core and Enhanced models.

---

# 2. Risk Prediction

HepaGuard uses custom **XGBoost classifiers** trained using NHANES 2017–2020 data.

Two model variants are available:

### Core Model

Designed for situations where only routinely available clinical information exists.

This is important because fasting triglycerides and glucose may not always be available during a primary care encounter.

### Enhanced Model

Adds fasting glucose and triglycerides when those measurements are available, allowing additional metabolic information to contribute to the prediction.

When enhanced inputs are missing, HepaGuard can automatically fall back to the Core model rather than failing the assessment.

---

## Dataset & Labeling

The data pipeline combines NHANES demographic, anthropometric, laboratory, alcohol, and physical-activity datasets using participant identifier `SEQN`.

The resulting feature dataset contains approximately **9,700 participants**.

MASLD-related ground truth for the MVP was derived using the **U.S. Fatty Liver Index (US-FLI)**:

```text
US-FLI ≥ 30 → positive label
```

The entire dataset construction, labeling, splitting, training, and packaging pipeline is reproducible from scripts in the repository.

---

## Model Performance

The production models were evaluated using stable train/validation/test splits and screening-oriented metrics.

Example Core model performance:

| Metric | Result |
|---|---:|
| AUROC | ~0.89 |
| PR-AUC | ~0.83 |
| Recall | ~0.85 |
| Precision | ~0.72 |
| F1 | ~0.77 |

Because HepaGuard is intended as an **early-risk decision-support tool**, threshold selection emphasized recall rather than simply maximizing overall accuracy.

Logistic Regression and Random Forest models were also trained as baselines before selecting XGBoost for the production pipeline.

---

# 3. Explainable AI

A probability alone is not enough for clinical decision support.

For every prediction, HepaGuard calculates **SHAP values** and surfaces the three features contributing most strongly to that individual prediction.

Example:

```text
MASLD Risk
93.9% — High

Top Drivers

↑ Fasting Glucose       132 mg/dL
↑ Waist Circumference   108 cm
↑ ALT                    26 U/L
```

Each driver contains:

```json
{
  "feature": "waist_cm",
  "direction": "increases_risk",
  "impact": 0.79,
  "value": 108
}
```

This allows the clinician to understand whether the model's reasoning appears clinically plausible rather than treating the result as an unexplained black box.

---

# 4. Guideline-Grounded RAG

After generating the risk assessment, HepaGuard builds a retrieval query using more than the raw patient inputs.

The RAG pipeline receives context from:

- the patient profile
- predicted risk probability
- risk category
- top SHAP drivers
- metabolic and lifestyle characteristics

This allows the ML model to help **steer retrieval toward clinically relevant guidance**.

### Clinical knowledge base

The MVP knowledge base contains **488 indexed guideline chunks** derived from four authoritative sources, including:

- AASLD Practice Guidance on NAFLD
- MASLD clinical/nomenclature updates
- Multi-society MASLD/MASH Delphi consensus
- AASLD MASLD clinical decision tree

Each indexed chunk contains metadata such as:

```text
Document
Year
Section
Chunk ID
Source
Content
Vector embedding
```

---

## Retrieval

The patient's clinical context is embedded using Azure OpenAI and sent to **Azure AI Search**, where vector/hybrid retrieval identifies the most relevant guideline passages.

Instead of giving the LLM an entire guideline document, HepaGuard provides a small set of relevant evidence excerpts.

---

## Grounded Generation

Azure OpenAI then receives:

1. the patient's clinical profile
2. the ML risk result
3. the top SHAP drivers
4. retrieved clinical guideline excerpts
5. strict generation instructions

The model produces concise, primary-care-oriented considerations such as:

```text
• Calculate fibrosis risk using an appropriate non-invasive assessment.

• Address metabolic risk factors and reinforce physical activity and
  weight-management interventions.

• Consider additional fibrosis evaluation when supported by the
  patient's risk profile and guideline criteria.
```

The response is accompanied by citations such as:

```text
AASLD Practice Guidance on the clinical assessment and
management of NAFLD (2023) — Chunk 0320
```

The LLM is therefore used primarily for **grounded synthesis**, not as the source of medical knowledge itself.

---

# 5. The Clinical Brief

The complete system returns one unified response:

```json
{
  "risk_probability": 0.939,
  "risk_label": "high",
  "model_used": "enhanced",
  "top_factors": [
    {
      "feature": "fasting_glucose_mg_dl",
      "direction": "increases_risk",
      "value": 132
    },
    {
      "feature": "waist_cm",
      "direction": "increases_risk",
      "value": 108
    }
  ],
  "guideline_next_steps": "...",
  "citations": ["..."],
  "warnings": [],
  "disclaimer": "..."
}
```

The frontend converts this into a clinician-readable dashboard containing:

- risk meter and category
- model used
- top three risk drivers
- guideline-directed next steps
- supporting citations
- missing-data or validation warnings

---

# Architecture

HepaGuard uses several Azure services with clearly separated responsibilities.

### Azure Machine Learning

Hosts the custom XGBoost models behind a Managed Online Endpoint.

Responsible for:

```text
Patient Features → Risk Probability → Risk Category → SHAP Drivers
```

### Azure Functions

Acts as the orchestration layer.

It validates requests, calls the ML endpoint, invokes the RAG pipeline, handles failures, and returns the unified Clinical Brief.

### Azure AI Search

Stores and retrieves vectorized clinical guideline passages.

The production index contains **488 guideline chunks** using HNSW cosine vector search.

### Azure OpenAI / Azure AI Foundry

Used for:

- query embeddings
- grounded clinical recommendation synthesis

### Next.js

Provides the clinician-facing interface for patient entry and Clinical Brief visualization.

---

# Tech Stack

### Machine Learning

- Python
- XGBoost
- scikit-learn
- SHAP
- pandas / NumPy
- NHANES

### Backend

- FastAPI
- Pydantic
- Azure Functions

### AI / Cloud

- Azure Machine Learning
- Azure AI Search
- Azure OpenAI
- Azure AI Foundry

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui

### Infrastructure & Tooling

- Azure CLI
- Docker
- Git
- reproducible model artifacts and stable dataset splits

---

# Repository Structure

```text
hepa-guard/
│
├── app/
│   ├── inference/              # preprocessing, prediction, SHAP
│   ├── recommendations/        # retrieval + RAG generation
│   ├── main.py                 # FastAPI application
│   └── schemas.py              # API contracts
│
├── azureml/
│   └── online_endpoint/
│       ├── endpoint.yml
│       ├── deployment.yml
│       ├── score.py
│       └── conda.yaml
│
├── data/
│   ├── raw/                    # NHANES source datasets
│   └── processed/              # features, labels, stable splits
│
├── docs/
│   ├── api_contract_v1.md
│   └── data_dictionary.md
│
├── hepaguard_ml/               # training/evaluation package
│
├── models/
│   ├── core/
│   └── enhanced/
│
├── rag_docs/
│   ├── sources/                # clinical guideline sources
│   ├── index/
│   │   └── chunks.jsonl
│   └── manifest.json
│
├── reports/                    # evaluation + SHAP reports
│
├── samples/                    # example API requests
│
├── scripts/
│   ├── build_nhanes_features.py
│   ├── build_usfli_labels.py
│   ├── make_splits.py
│   ├── train_and_package.py
│   └── rag/
│
├── web/
│   └── hepaguard-ui/           # Next.js clinician interface
│
└── function_app.py             # Azure Functions orchestration
```

---

# Reproducible ML Pipeline

The underlying dataset and model artifacts can be rebuilt using:

```bash
python scripts/build_nhanes_features.py \
  --raw_dir data/raw \
  --out_dir data/processed

python scripts/build_usfli_labels.py \
  --raw_dir data/raw \
  --features_path data/processed/hepaguard_features.parquet \
  --out_dir data/processed

python -m scripts.make_splits

python -m scripts.train_and_package
```

This produces stable data splits and packaged Core/Enhanced model artifacts under `models/`.

---

# API

### `POST /predict`

Runs ML inference and explainability.

Returns:

```text
risk_probability
risk_label
model_used
top_factors
warnings
```

### `POST /clinical-brief`

Runs the complete decision-support workflow:

```text
Prediction
    +
SHAP Explanation
    +
Guideline Retrieval
    +
Grounded Recommendation
    =
Clinical Brief
```

### Azure Functions

The production-style orchestration endpoint exposes:

```text
POST /api/clinical-brief
```

allowing the frontend to make a **single request** for the complete assessment.

---

# Input Validation & Safety

Healthcare software should fail loudly rather than silently producing nonsense.

HepaGuard therefore includes backend guardrails for:

- missing required values
- invalid types
- clinically implausible ranges
- unsupported units
- missing enhanced-model inputs
- unknown demographic values

Where appropriate, the application returns warnings rather than failing entirely.

Example:

```text
Enhanced fields missing; falling back to core model.
```

---

# Responsible AI

HepaGuard was intentionally designed around **human-in-the-loop clinical decision support**.

### The system does not:

- diagnose MASLD
- replace physician judgment
- autonomously order tests
- autonomously initiate treatment
- present LLM output as clinical fact

### The system does:

- expose model reasoning through SHAP
- ground generated guidance in retrieved clinical evidence
- provide citations
- clearly communicate missing information
- validate clinical inputs
- preserve clinician authority over every downstream decision

Every Clinical Brief includes:

> **HepaGuard is a prototype CDSS demo. It is not a diagnostic tool or medical device. It does not replace clinical judgment.**

---

# Clinical Design Philosophy

HepaGuard was designed around a simple principle:

> **AI in healthcare should make clinicians more informed, not less involved.**

The value of the system is not simply predicting that a patient may be at higher risk.

It is showing:

**what the model noticed, why it mattered, what the evidence says, and where the clinician may want to look next.**

---

# Current Status

HepaGuard reached a functional end-to-end MVP with:

- reproducible NHANES data pipeline
- custom Core and Enhanced XGBoost models
- stable evaluation pipeline
- patient-level SHAP explanations
- Azure ML deployment
- Azure AI Search vector index
- Azure OpenAI RAG pipeline
- Azure Functions orchestration
- clinician-facing Next.js interface
- structured validation and failure handling
- guideline citations
- clinician feedback and workflow validation

The system was developed as a **Microsoft Imagine Cup 2026** project.

---

## Future Work

Potential next steps include:

- prospective clinical validation
- external validation against independent patient populations
- stronger probability calibration
- expanded MASLD/MASH guideline corpus
- automated FIB-4 and fibrosis-risk pathways
- EHR integration using FHIR
- embedded CDS workflows within systems such as Epic, Oracle Health/Cerner, and athenahealth
- longitudinal patient monitoring
- formal security, regulatory, and clinical validation processes

---

# Disclaimer

**HepaGuard is a research and demonstration prototype. It is not a diagnostic tool, medical device, or substitute for professional medical judgment. It has not been clinically validated or cleared for patient-care use.**
