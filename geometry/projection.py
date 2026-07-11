"""Coordinate-system projection helpers.

This module converts already-known coordinates between systems. It does not
estimate homographies, detect markers, or perform mission-specific logic.
"""

from __future__ import annotations

from typing import Sequence

from geometry.transform import apply_homography


def image_to_map(points_px: Sequence[Sequence[float]], homography: object):
    """Project image pixel coordinates into map coordinates using ``homography``."""
    return apply_homography(points_px, homography)


def map_to_world(
    points_map: Sequence[Sequence[float]],
    scale: float | Sequence[float] = 1.0,
    origin: Sequence[float] = (0.0, 0.0),
) -> object:
    """Convert map coordinates to real-world coordinates by scale and origin.

    Defaults preserve existing behavior when map coordinates are already world
    coordinates, as in the legacy ground mission homography measured in cm.
    """
    import numpy as np

    points = np.array(points_map, dtype=np.float32)
    scale_arr = np.array(scale if isinstance(scale, Sequence) else (scale, scale), dtype=np.float32)
    origin_arr = np.array(origin, dtype=np.float32)
    return points * scale_arr + origin_arr


# Backward-compatible name used by the legacy ground mission. The existing
# homography projects image pixels directly to real-world centimeters.
def image_points_to_world(points_px: Sequence[Sequence[float]], homography: object) -> object:
    return image_to_map(points_px, homography)
