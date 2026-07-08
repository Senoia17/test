#!/usr/bin/env python3
"""Synthetic smoke test for aruco_diorama_mapper.py.

This test creates an artificial top-down arena image with four ArUco markers,
warps it into a simulated drone perspective image, then verifies that the mapper
recovers a top-down map with the expected size and detectable corner markers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from aruco_diorama_mapper import (
    create_map,
    detect_markers,
    marker_centers,
    select_source_points,
)

MARKER_IDS = [10, 11, 12, 13]
WIDTH_M = 4.0
HEIGHT_M = 2.5
PIXELS_PER_METER = 120
DICT_NAME = "DICT_4X4_50"


def draw_marker(dictionary: cv2.aruco.Dictionary, marker_id: int, size_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, size_px)


def build_top_down_arena() -> np.ndarray:
    width_px = int(WIDTH_M * PIXELS_PER_METER)
    height_px = int(HEIGHT_M * PIXELS_PER_METER)
    arena = np.full((height_px, width_px, 3), 245, dtype=np.uint8)
    cv2.rectangle(arena, (0, 0), (width_px - 1, height_px - 1), (30, 30, 30), 3)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    marker_size = 52
    margin = 18
    placements = [
        (margin, margin),
        (width_px - margin - marker_size, margin),
        (width_px - margin - marker_size, height_px - margin - marker_size),
        (margin, height_px - margin - marker_size),
    ]
    for marker_id, (x, y) in zip(MARKER_IDS, placements):
        marker = draw_marker(dictionary, marker_id, marker_size)
        arena[y : y + marker_size, x : x + marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return arena


def simulate_drone_view(top_down: np.ndarray) -> np.ndarray:
    height, width = top_down.shape[:2]
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = np.array([[90, 35], [width - 75, 5], [width - 25, height - 45], [35, height - 10]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(top_down, homography, (width, height), borderValue=(255, 255, 255))


def main() -> None:
    drone_image = simulate_drone_view(build_top_down_arena())
    with tempfile.TemporaryDirectory() as tmpdir:
        debug_path = Path(tmpdir) / "synthetic_drone.png"
        cv2.imwrite(str(debug_path), drone_image)

        corners, ids = detect_markers(drone_image, DICT_NAME)
        centers = marker_centers(corners, ids)
        source_points, selected_ids = select_source_points(centers, MARKER_IDS)
        mapped, destination_points, homography = create_map(
            drone_image, source_points, WIDTH_M, HEIGHT_M, PIXELS_PER_METER
        )

    assert selected_ids == MARKER_IDS
    assert mapped.shape[:2] == (int(HEIGHT_M * PIXELS_PER_METER), int(WIDTH_M * PIXELS_PER_METER))
    assert destination_points.shape == (4, 2)
    assert homography.shape == (3, 3)
    print("Synthetic ArUco mapping smoke test passed.")


if __name__ == "__main__":
    main()
