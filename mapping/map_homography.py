"""Map homography calculation from detected ArUco image coordinates."""

from __future__ import annotations

from typing import Mapping, Sequence

from mapping.aruco_config import get_aruco_marker_positions, resolve_corner_ids
from mapping.aruco_detector import marker_centers


FIELD_CORNERS_CM: dict[int, list[float]] = get_aruco_marker_positions()


def _centers_from_markers_or_mapping(markers: object) -> dict[int, list[float]]:
    if isinstance(markers, Mapping):
        return {int(marker_id): [float(point[0]), float(point[1])] for marker_id, point in markers.items()}
    return marker_centers(markers)  # type: ignore[arg-type]


def calculate_map_homography(
    markers: object,
    marker_positions: Mapping[int, Sequence[float]] | None = None,
    required_ids: Sequence[int] | None = None,
):
    """Calculate image-to-global-map homography from matching ArUco points."""
    import cv2
    import numpy as np

    required_ids = resolve_corner_ids(required_ids)
    centers = _centers_from_markers_or_mapping(markers)
    positions = marker_positions or get_aruco_marker_positions()
    missing = [marker_id for marker_id in required_ids if marker_id not in centers]
    if missing:
        raise ValueError(f"Missing ArUco marker IDs: {missing}")

    image_points = np.array([centers[i] for i in required_ids], dtype=np.float32)
    world_points = np.array([positions[i] for i in required_ids], dtype=np.float32)

    homography, mask = cv2.findHomography(image_points, world_points)
    if homography is None:
        raise ValueError("Failed to compute homography.")

    return homography


# Backward-compatible name from ground_mission/src/aruco_homography.py.
def compute_homography(centers: Mapping[int, Sequence[float]]):
    return calculate_map_homography(centers)
