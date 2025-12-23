from __future__ import annotations

from app.recommendations.retriever import retrieve_guideline_chunks
from app.recommendations.types import GuidelineChunk
from app.schemas import LabsInfo, LifestyleInfo, MetaInfo, PatientInfo

GUIDELINE_TEMPLATE = (
    "Next steps:\n"
    "- Consider repeating key labs to confirm trends over time.\n"
    "- Review metabolic risk factors (weight, glycemic control, lipids) and lifestyle context.\n"
    "- Consider non-invasive fibrosis risk stratification if clinically indicated.\n"
    "\n"
    "When to escalate:\n"
    "- Worsening labs, high clinical suspicion, or persistent abnormal results.\n"
    "\n"
    "Safety:\n"
    "- This is not a diagnosis; use clinical judgment and guideline-aligned care."
)


def format_citation(chunk: GuidelineChunk) -> str:
    return f"{chunk.doc_title} ({chunk.year}) — {chunk.section} — {chunk.locator}"


def generate_recommendations(
    patient: PatientInfo,
    labs: LabsInfo,
    lifestyle: LifestyleInfo,
    meta: MetaInfo | None,
) -> tuple[str, list[str], list[str]]:
    query_parts = [
        f"bmi:{patient.bmi_kg_m2}",
        f"waist:{patient.waist_cm}",
        f"alt:{labs.alt_u_l}",
        f"ast:{labs.ast_u_l}",
    ]
    if labs.triglycerides_mg_dl is not None:
        query_parts.append("triglycerides")
    if labs.fasting_glucose_mg_dl is not None:
        query_parts.append("fasting_glucose")
    if lifestyle.alcohol_drinks_per_day is not None:
        query_parts.append("alcohol")

    query = " ".join(query_parts)
    chunks = retrieve_guideline_chunks(query, k=4)

    if chunks:
        citations = [format_citation(chunk) for chunk in chunks]
        return GUIDELINE_TEMPLATE, citations, []

    warnings = [
        "No AASLD guideline chunks available; returning recommendations without citations."
    ]
    return GUIDELINE_TEMPLATE, [], warnings
