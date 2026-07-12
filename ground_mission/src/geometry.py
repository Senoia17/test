"""Backward-compatible wrappers for the shared geometry package.

The legacy ground mission imports this local module when executed from
``ground_mission/src``. Keep those imports working while delegating reusable math
to the Phase 4 geometry layer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_GEOMETRY_DIR = _REPO_ROOT / "geometry"


def _load_shared_module(module_name: str) -> ModuleType:
    module_path = _SHARED_GEOMETRY_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_shared_geometry_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared geometry module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_transform = _load_shared_module("transform")
_measurement = _load_shared_module("measurement")


def image_points_to_world(points_px: Sequence[Sequence[float]], homography: object):
    return _transform.apply_homography(points_px, homography)


def bbox_center_xyxy(xyxy: Sequence[float]) -> list[float]:
    return _transform.bbox_center(xyxy)


def bbox_corners_xyxy(xyxy: Sequence[float]) -> list[list[float]]:
    return _transform.bbox_corners(xyxy)


def polygon_size_mm(world_corners_cm: Sequence[Sequence[float]]) -> tuple[float, float]:
    return _measurement.polygon_size_mm(world_corners_cm)
