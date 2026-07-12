"""Pure image/map coordinate transformation utilities."""

from __future__ import annotations

from typing import Sequence



def bbox_center(bbox: Sequence[float]) -> list[float]:
    """Return the center point ``[x, y]`` for an ``[x1, y1, x2, y2]`` bbox."""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bbox_corners(bbox: Sequence[float]) -> list[list[float]]:
    """Return bbox corners in clockwise order starting at top-left."""
    x1, y1, x2, y2 = bbox
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def apply_homography(points: Sequence[Sequence[float]], homography: object):
    """Apply a 3x3 homography to 2D points and return transformed points."""
    import cv2
    import numpy as np

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, np.array(homography, dtype=np.float32))
    return transformed.reshape(-1, 2)


# Backward-compatible names used by the legacy ground mission.
bbox_center_xyxy = bbox_center
bbox_corners_xyxy = bbox_corners
