# Processed Guideline Documents

This folder contains cleaned guideline text used by the local retrieval mode.

How to add new documents:
1) Add a processed markdown file in this folder.
2) Update `rag_docs/manifest.json` with a new entry (doc_id, title, year, source, path).
3) Run `python scripts/build_local_guideline_index.py` to regenerate `rag_docs/index/chunks.jsonl`.

Only sources tagged with `source: "AASLD"` will be used for citations.
