#!/usr/bin/env python3
"""Synthetic smoke test for aruco_diorama_mapper.py.

This test creates an artificial top-down arena image with four ArUco markers,
warps it into a simulated drone perspective image, then verifies that the mapper
recovers a top-down map with the expected size and detectable corner markers.

By default the test only prints a pass/fail result. Use --output-dir to keep
visual artifacts for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from aruco_diorama_mapper import (
    create_map,
    detect_markers,
    marker_corners_by_id,
    select_source_points,
    write_debug_image,
)

MARKER_IDS = [10, 11, 12, 13]
WIDTH_M = 5.0
HEIGHT_M = 4.0
PIXELS_PER_METER = 120
DICT_NAME = "DICT_4X4_50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic ArUco mapping smoke test.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory where synthetic input, mapped output, debug overlay, and metadata are saved.",
    )
    return parser.parse_args()


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
    placements = [
        (0, 0),
        (width_px - marker_size, 0),
        (width_px - marker_size, height_px - marker_size),
        (0, height_px - marker_size),
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


def save_artifacts(
    output_dir: Path,
    top_down: np.ndarray,
    drone_image: np.ndarray,
    mapped: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    homography: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "01_expected_top_down.png"), top_down)
    cv2.imwrite(str(output_dir / "02_synthetic_drone_input.png"), drone_image)
    cv2.imwrite(str(output_dir / "03_mapped_output.png"), mapped)
    write_debug_image(drone_image, corners, ids, source_points, output_dir / "04_detected_markers_debug.png")

    metadata = {
        "marker_ids": MARKER_IDS,
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "pixels_per_meter": PIXELS_PER_METER,
        "source_points_px": source_points.tolist(),
        "destination_points_px": destination_points.tolist(),
        "homography": homography.tolist(),
        "artifacts": [
            "01_expected_top_down.png",
            "02_synthetic_drone_input.png",
            "03_mapped_output.png",
            "04_detected_markers_debug.png",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    top_down = build_top_down_arena()
    drone_image = simulate_drone_view(top_down)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = args.output_dir or Path(tmpdir)
        corners, ids = detect_markers(drone_image, DICT_NAME)
        marker_points = marker_corners_by_id(corners, ids)
        source_points, selected_ids = select_source_points(marker_points, MARKER_IDS)
        mapped, destination_points, homography = create_map(
            drone_image, source_points, WIDTH_M, HEIGHT_M, PIXELS_PER_METER
        )

        assert selected_ids == MARKER_IDS
        assert mapped.shape[:2] == (int(HEIGHT_M * PIXELS_PER_METER), int(WIDTH_M * PIXELS_PER_METER))
        assert destination_points.shape == (4, 2)
        assert homography.shape == (3, 3)
        expected_corners = np.array(
            [
                [0, 0],
                [top_down.shape[1] - 1, 0],
                [top_down.shape[1] - 1, top_down.shape[0] - 1],
                [0, top_down.shape[0] - 1],
            ],
            dtype=np.float32,
        )
        reprojected = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        assert np.allclose(reprojected, expected_corners, atol=1.0)

        if args.output_dir:
            save_artifacts(
                work_dir,
                top_down,
                drone_image,
                mapped,
                corners,
                ids,
                source_points,
                destination_points,
                homography,
            )
            print(f"Synthetic ArUco mapping smoke test passed. Artifacts saved to: {work_dir}")
        else:
            print("Synthetic ArUco mapping smoke test passed. Use --output-dir test_outputs to save images.")


if __name__ == "__main__":
    main()
