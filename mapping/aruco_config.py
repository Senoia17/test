"""Shared ArUco configuration helpers.

The project-level ``config.yaml`` is the single source of truth for ArUco
settings. Function arguments in detector/localization/mapping APIs may still
explicitly override these defaults.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ArucoConfig:
    """ArUco settings loaded from project configuration."""

    dictionary: str
    corner_ids: tuple[int, ...]
    world_corners_cm: tuple[tuple[float, float], ...]

    @property
    def marker_positions_cm(self) -> dict[int, list[float]]:
        """Return marker ID to world-corner coordinate mapping."""
        return {
            marker_id: [float(point[0]), float(point[1])]
            for marker_id, point in zip(self.corner_ids, self.world_corners_cm)
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    configured = os.environ.get("DRONE_AI_CONFIG")
    if configured:
        return Path(configured)
    return _repo_root() / "config.yaml"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            return [_parse_scalar(item) for item in items]
    try:
        return int(value)
    except ValueError:
        return value.strip('"\'')


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """Parse the nested subset of YAML used by config.yaml."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or _config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_yaml_fallback(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_aruco_config(config: dict[str, Any] | None = None) -> ArucoConfig:
    """Return ArUco defaults from ``mapping.aruco`` in config.yaml."""
    data = config or _load_config()
    mapping_config = data.get("mapping")
    if not isinstance(mapping_config, dict):
        raise KeyError("config.yaml must define mapping.aruco.dictionary and mapping.aruco.corner_ids")
    aruco_config = mapping_config.get("aruco")
    if not isinstance(aruco_config, dict):
        raise KeyError("config.yaml must define mapping.aruco.dictionary and mapping.aruco.corner_ids")

    dictionary = aruco_config.get("dictionary")
    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError("mapping.aruco.dictionary must be a non-empty string")

    corner_ids_raw = aruco_config.get("corner_ids")
    if not isinstance(corner_ids_raw, Sequence) or isinstance(corner_ids_raw, (str, bytes)):
        raise ValueError("mapping.aruco.corner_ids must be a sequence of marker IDs")
    corner_ids = tuple(int(marker_id) for marker_id in corner_ids_raw)
    if not corner_ids:
        raise ValueError("mapping.aruco.corner_ids must not be empty")

    world_corners_raw = aruco_config.get("world_corners_cm")
    if not isinstance(world_corners_raw, Sequence) or isinstance(world_corners_raw, (str, bytes)):
        raise ValueError("mapping.aruco.world_corners_cm must be a sequence of [x, y] coordinates")
    world_corners = tuple(
        (float(point[0]), float(point[1]))  # type: ignore[index]
        for point in world_corners_raw
    )
    if len(world_corners) != len(corner_ids):
        raise ValueError("mapping.aruco.world_corners_cm must have the same length as mapping.aruco.corner_ids")

    return ArucoConfig(dictionary=dictionary, corner_ids=corner_ids, world_corners_cm=world_corners)


def get_aruco_dictionary_name(config: dict[str, Any] | None = None) -> str:
    """Return the configured ArUco dictionary name."""
    return get_aruco_config(config).dictionary


def get_aruco_corner_ids(config: dict[str, Any] | None = None) -> tuple[int, ...]:
    """Return configured ArUco corner IDs in TL, TR, BR, BL order."""
    return get_aruco_config(config).corner_ids


def get_aruco_marker_positions(config: dict[str, Any] | None = None) -> dict[int, list[float]]:
    """Return marker ID to world-corner coordinates from config."""
    return get_aruco_config(config).marker_positions_cm


def resolve_dictionary_name(dictionary_name: str | None = None, config: dict[str, Any] | None = None) -> str:
    """Use an explicit dictionary name when provided, otherwise config default."""
    return dictionary_name or get_aruco_dictionary_name(config)


def resolve_corner_ids(
    corner_ids: Sequence[int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    """Use explicit corner IDs when provided, otherwise config default."""
    if corner_ids is not None:
        return tuple(int(marker_id) for marker_id in corner_ids)
    return get_aruco_corner_ids(config)
