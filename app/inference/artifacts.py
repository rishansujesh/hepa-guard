from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal

import xgboost as xgb


@dataclass
class ModelBundle:
    variant: str
    booster: xgb.Booster
    feature_order: list[str]
    preprocessing: dict
    threshold: dict
    risk_cutoffs: dict
    model_card: dict


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(
    variant: Literal["core", "enhanced"],
    base_dir: str = "models",
) -> ModelBundle:
    base_path = Path(base_dir) / variant
    if not base_path.exists():
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    model_path = base_path / "model.json"
    feature_path = base_path / "feature_order.json"
    preprocessing_path = base_path / "preprocessing.json"
    threshold_path = base_path / "threshold.json"
    risk_cutoffs_path = base_path / "risk_cutoffs.json"
    model_card_path = base_path / "model_card.json"

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    feature_payload = _read_json(feature_path)
    feature_order = feature_payload.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError(f"Invalid feature_order in {feature_path}")

    preprocessing = _read_json(preprocessing_path)
    threshold = _read_json(threshold_path)
    risk_cutoffs = _read_json(risk_cutoffs_path)
    model_card = _read_json(model_card_path)

    return ModelBundle(
        variant=variant,
        booster=booster,
        feature_order=feature_order,
        preprocessing=preprocessing,
        threshold=threshold,
        risk_cutoffs=risk_cutoffs,
        model_card=model_card,
    )


def load_all_bundles(base_dir: str = "models") -> dict[str, ModelBundle]:
    return {
        "core": load_bundle("core", base_dir=base_dir),
        "enhanced": load_bundle("enhanced", base_dir=base_dir),
    }
