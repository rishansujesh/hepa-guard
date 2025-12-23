from __future__ import annotations

import json
import os
import re
from typing import Iterable

from app.recommendations.types import GuidelineChunk

CHUNKS_PATH = os.path.join("rag_docs", "index", "chunks.jsonl")

_cached_chunks: list[GuidelineChunk] | None = None


def _load_chunks() -> list[GuidelineChunk]:
    if not os.path.exists(CHUNKS_PATH):
        return []
    chunks: list[GuidelineChunk] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(
                GuidelineChunk(
                    chunk_id=data["chunk_id"],
                    content=data["content"],
                    doc_title=data["doc_title"],
                    year=str(data["year"]),
                    source=data["source"],
                    doc_id=data["doc_id"],
                    section_path=data.get("section_path", ""),
                    section=data.get("section_path", ""),
                    locator=data.get("locator", ""),
                )
            )
    return chunks


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score(query_tokens: set[str], chunk: GuidelineChunk) -> int:
    chunk_tokens = _tokenize(chunk.content + " " + chunk.section_path)
    return len(query_tokens & chunk_tokens)


def _filter_aasld(chunks: Iterable[GuidelineChunk]) -> list[GuidelineChunk]:
    return [chunk for chunk in chunks if chunk.source.upper() == "AASLD"]


def retrieve(query: str, k: int = 4) -> list[GuidelineChunk]:
    global _cached_chunks
    if _cached_chunks is None:
        _cached_chunks = _filter_aasld(_load_chunks())

    query_tokens = _tokenize(query)
    scored = []
    for chunk in _cached_chunks:
        score = _score(query_tokens, chunk)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [chunk for _, chunk in scored[:k]]
