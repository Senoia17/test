"""Physical object measurement helpers."""

from __future__ import annotations

from typing import Sequence



def polygon_dimensions(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Measure average width and height of four transformed object corners."""
    import numpy as np

    pts = np.array(points, dtype=np.float32)

    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])

    width = (w1 + w2) / 2.0
    height = (h1 + h2) / 2.0

    return float(width), float(height)


def physical_size(points: Sequence[Sequence[float]], unit_scale: float = 1.0) -> tuple[float, float]:
    """Return physical ``(width, height)`` after applying a unit scale."""
    width, height = polygon_dimensions(points)
    return width * unit_scale, height * unit_scale


def polygon_size_mm(world_corners_cm: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Measure corner positions in centimeters and return width/height in mm."""
    return physical_size(world_corners_cm, unit_scale=10.0)
