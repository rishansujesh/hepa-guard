from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

from .config import LABEL_COL


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    if p.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(p)
        except Exception:
            csv_path = p.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    raise ValueError(f"Unsupported dataset extension: {p.suffix}")


def filter_labeled(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.DataFrame:
    labeled = df[df[label_col].notna()].copy()
    labeled[label_col] = labeled[label_col].astype(int)
    return labeled


def compute_dataset_fingerprint(df: pd.DataFrame, used_cols: list[str]) -> str:
    subset = df[used_cols].head(5000).copy()
    csv_bytes = subset.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:12]
