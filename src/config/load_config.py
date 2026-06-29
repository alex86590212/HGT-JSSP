"""Load JSON configuration files for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_json_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_intersection_config(path: str | Path = DEFAULT_CONFIG_DIR / "intersection_config.json") -> Dict[str, Any]:
    return load_json_config(path)


def load_vehicle_types(path: str | Path = DEFAULT_CONFIG_DIR / "vehicle_types.json") -> Dict[str, Any]:
    return load_json_config(path)


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_DIR / "experiment_config.json") -> Dict[str, Any]:
    return load_json_config(path)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

