#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_MIN = 800
TARGET_MAX = 1200


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="rag_docs/manifest.json")
    ap.add_argument("--out", type=str, default="rag_docs/index/chunks.jsonl")
    return ap.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list of document entries.")
    return data


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing document: {path}")
    return path.read_text(encoding="utf-8")


def iter_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush_section() -> None:
        if not buffer:
            return
        section_path = " > ".join(heading_stack) if heading_stack else "General"
        sections.append((section_path, "\n".join(buffer).strip()))
        buffer.clear()

    for line in lines:
        if line.startswith("#"):
            flush_section()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            if level == 1:
                heading_stack = [title]
            else:
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
            continue
        buffer.append(line)

    flush_section()
    return sections


def chunk_text(section_path: str, text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        next_len = current_len + len(paragraph) + 1
        if current and next_len > TARGET_MAX:
            chunks.append((section_path, "\n".join(current).strip()))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len

    if current:
        chunks.append((section_path, "\n".join(current).strip()))

    return chunks


def build_chunks(doc: dict, text: str) -> list[dict]:
    chunks: list[dict] = []
    section_pairs = iter_sections(text)
    idx = 0

    for section_path, section_text in section_pairs:
        for section_path, content in chunk_text(section_path, section_text):
            if not content:
                continue
            chunk = {
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "year": doc["year"],
                "source": doc["source"],
                "section_path": section_path,
                "locator": section_path,
                "chunk_id": f"{doc['doc_id']}::{idx:04d}",
                "content": content,
            }
            chunks.append(chunk)
            idx += 1

    return chunks


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    all_chunks: list[dict] = []
    for doc in manifest:
        doc_path = Path(doc["path"])
        text = read_text(doc_path)
        all_chunks.extend(build_chunks(doc, text))

    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
