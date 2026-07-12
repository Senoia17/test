"""Helpers for resolving configured model weight paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _mission_model_config(config: Mapping[str, Any], mission: str) -> Mapping[str, Any] | None:
    models_config = config.get("models", {})
    if not isinstance(models_config, Mapping):
        return None

    mission_config = models_config.get(mission)
    return mission_config if isinstance(mission_config, Mapping) else None


def _resolve_model_file(config: Mapping[str, Any], mission: str, key: str) -> Path | None:
    mission_config = _mission_model_config(config, mission)
    if mission_config is None:
        return None

    directory = mission_config.get("directory")
    filename = mission_config.get(key)
    if filename is None and key == "weights":
        filename = mission_config.get("version")
    if directory and filename:
        return Path(str(directory)) / str(filename)
    if filename:
        return Path(str(filename))
    if directory and key == "weights":
        return Path(str(directory))
    return None


def resolve_model_weight_path(config: Mapping[str, Any], mission: str) -> Path | None:
    """Build an inference weight path from ``models.<mission>.directory/weights`` config."""
    return _resolve_model_file(config, mission, "weights")


def resolve_pretrained_model_path(config: Mapping[str, Any], mission: str) -> str | Path | None:
    """Resolve the configured pretrained checkpoint used to initialize training."""
    pretrained_path = _resolve_model_file(config, mission, "pretrained")
    if pretrained_path is None:
        return None
    return pretrained_path if pretrained_path.exists() else pretrained_path.name
