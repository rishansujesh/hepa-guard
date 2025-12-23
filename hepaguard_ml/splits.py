from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ID_COL, LABEL_COL, RANDOM_SEED


def make_split_indices(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    label_col: str = LABEL_COL,
    id_col: str = ID_COL,
) -> dict[str, list[int]]:
    if df[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col} values found; cannot split reliably.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[label_col],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[label_col],
        random_state=seed,
    )

    return {
        "train": train_df[id_col].astype(int).tolist(),
        "val": val_df[id_col].astype(int).tolist(),
        "test": test_df[id_col].astype(int).tolist(),
    }


def save_split_indices(path: str | Path, split_indices: dict[str, list[int]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)


def load_split_indices(path: str | Path) -> dict[str, list[int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split indices not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(map(int, v)) for k, v in data.items()}


def apply_split_indices(
    df: pd.DataFrame,
    split_indices: dict[str, list[int]],
    id_col: str = ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ids = set(split_indices.get("train", []))
    val_ids = set(split_indices.get("val", []))
    test_ids = set(split_indices.get("test", []))

    train_df = df[df[id_col].isin(train_ids)].copy()
    val_df = df[df[id_col].isin(val_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()

    return train_df, val_df, test_df


def compute_split_stats(
    df: pd.DataFrame,
    split_indices: dict[str, list[int]],
    label_col: str = LABEL_COL,
    id_col: str = ID_COL,
) -> dict[str, dict[str, float]]:
    stats = {}
    for split_name, ids in split_indices.items():
        sub = df[df[id_col].isin(ids)]
        pos_rate = float(sub[label_col].mean()) if len(sub) else 0.0
        stats[split_name] = {
            "rows": int(len(sub)),
            "positive_rate": pos_rate,
        }
    return stats
