"""Helpers for resolving configured model weight paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_model_weight_path(config: Mapping[str, Any], mission: str) -> Path | None:
    """Build a model weight path from ``models.<mission>.directory/version`` config."""
    models_config = config.get("models", {})
    if not isinstance(models_config, Mapping):
        return None

    mission_config = models_config.get(mission)
    if isinstance(mission_config, Mapping):
        directory = mission_config.get("directory")
        version = mission_config.get("version")
        if directory and version:
            return Path(str(directory)) / str(version)
        if version:
            return Path(str(version))
        if directory:
            return Path(str(directory))

    return None
