#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hepaguard_ml.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_STATS_PATH,
    LABEL_COL,
)
from hepaguard_ml.data import filter_labeled, load_dataset
from hepaguard_ml.splits import make_split_indices, save_split_indices, compute_split_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default=str(DEFAULT_SPLIT_PATH),
    )
    ap.add_argument(
        "--stats_path",
        type=str,
        default=str(DEFAULT_SPLIT_STATS_PATH),
    )
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def write_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    stats_path = Path(args.stats_path)

    if out_path.exists() and not args.force:
        raise FileExistsError(f"Split file already exists: {out_path}")

    df = load_dataset(data_path)
    labeled = filter_labeled(df, LABEL_COL)

    split_indices = make_split_indices(labeled)
    save_split_indices(out_path, split_indices)

    split_stats = compute_split_stats(labeled, split_indices)
    overall = {
        "total_rows": int(len(df)),
        "labeled_rows": int(len(labeled)),
        "positive_rate": float(labeled[LABEL_COL].mean()) if len(labeled) else 0.0,
    }
    stats_payload = {"overall": overall, "splits": split_stats}
    write_stats(stats_path, stats_payload)

    print(f"Saved splits: {out_path}")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
