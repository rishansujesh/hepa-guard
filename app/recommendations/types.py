from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuidelineChunk:
    chunk_id: str
    content: str
    doc_title: str
    year: str
    source: str
    doc_id: str
    section_path: str
    section: str
    locator: str
