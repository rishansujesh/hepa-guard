from __future__ import annotations

import os

from app.recommendations.local_index import retrieve as retrieve_local
from app.recommendations.types import GuidelineChunk

MOCK_CHUNKS = [
    GuidelineChunk(
        chunk_id="aasld-2023-s3",
        content="Non-invasive fibrosis risk stratification is recommended for risk assessment.",
        doc_title="AASLD MASLD Practice Guidance",
        year="2023",
        source="AASLD",
        doc_id="aasld-masld-2023",
        section_path="Risk Stratification",
        section="Non-invasive fibrosis risk stratification",
        locator="Section 3",
    ),
    GuidelineChunk(
        chunk_id="aasld-2023-s4",
        content="Metabolic risk factors should be assessed alongside liver enzyme trends.",
        doc_title="AASLD MASLD Practice Guidance",
        year="2023",
        source="AASLD",
        doc_id="aasld-masld-2023",
        section_path="Metabolic Risk Factors",
        section="Metabolic risk factors",
        locator="Section 4",
    ),
    GuidelineChunk(
        chunk_id="aasld-2023-s5",
        content="Lifestyle counseling is a foundational recommendation for MASLD risk reduction.",
        doc_title="AASLD MASLD Practice Guidance",
        year="2023",
        source="AASLD",
        doc_id="aasld-masld-2023",
        section_path="Lifestyle Counseling",
        section="Lifestyle counseling",
        locator="Section 5",
    ),
]


def retrieve_guideline_chunks(query: str, k: int = 4) -> list[GuidelineChunk]:
    mode = os.environ.get("RECS_MODE", "mock")
    if mode == "mock":
        return MOCK_CHUNKS[:k]
    if mode == "local":
        return retrieve_local(query, k=k)
    return []
