#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hepaguard_ml.config import DEFAULT_DATA_PATH, FEATURE_SETS, LABEL_COL
from hepaguard_ml.data import filter_labeled, load_dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.data_path))

    labeled = filter_labeled(df, LABEL_COL)
    total_rows = len(df)
    labeled_rows = len(labeled)
    pos_rate = float(labeled[LABEL_COL].mean()) if labeled_rows else 0.0

    print(f"Total rows: {total_rows:,}")
    print(f"Labeled rows: {labeled_rows:,}")
    print(f"Positive rate: {pos_rate * 100:.2f}%")

    missingness = df.isna().mean().sort_values(ascending=False) * 100
    print("\nMissingness (%):")
    print(missingness.round(2).to_string())

    for name, cols in FEATURE_SETS.items():
        stats = df[cols].agg(["min", "max"]).transpose()
        print(f"\nFeature ranges ({name}):")
        print(stats.to_string())


if __name__ == "__main__":
    main()
