#!/usr/bin/env python3
"""Create a top-down arena map from a drone image using four corner ArUco markers.

The script detects ArUco markers installed at the four corners of a rectangular
competition field, estimates a homography, and warps the input aerial image into
a metric bird's-eye map.

Example:
    python aruco_diorama_mapper.py input.jpg map.png \
        --marker-ids 10 11 12 13 \
        --map-size-mm 5000 4000 \
        --pixels-per-meter 300

Marker ID order is: top-left, top-right, bottom-right, bottom-left in the final
map coordinate system. The map boundary is defined by the outermost corner of
each ArUco marker, so the output includes the full corner markers. If IDs are
not provided, the script orders the four detected marker centers geometrically,
which works when the field is visible as a convex rectangle in the image.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class MapMetadata:
    input_image: str
    output_image: str
    width_mm: float
    height_mm: float
    width_m: float
    height_m: float
    pixels_per_meter: int
    output_width_px: int
    output_height_px: int
    marker_ids: list[int]
    source_points_px: list[list[float]]
    destination_points_px: list[list[float]]
    homography: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an orthorectified top-down map from four corner ArUco markers."
    )
    parser.add_argument("input", type=Path, help="Drone aerial image path.")
    parser.add_argument("output", type=Path, help="Output top-down map image path.")
    parser.add_argument(
        "--map-size-mm",
        type=float,
        nargs=2,
        metavar=("WIDTH_MM", "HEIGHT_MM"),
        default=(5000.0, 4000.0),
        help="Output map's real-world size in millimeters. Default: 5000 4000.",
    )
    parser.add_argument(
        "--arena-size",
        type=float,
        nargs=2,
        metavar=("WIDTH_M", "HEIGHT_M"),
        help="Deprecated alias for --map-size-mm, expressed in meters.",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=int,
        default=200,
        help="Output map resolution. Default: 200 px/m.",
    )
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs=4,
        metavar=("TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"),
        help="Corner marker IDs in final map order. Recommended for stable results.",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name. Default: DICT_4X4_50.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON path for homography and marker metadata.",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        help="Optional image path showing detected markers and selected outer marker corners.",
    )
    return parser.parse_args()


def get_aruco_dictionary(dictionary_name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, dictionary_name):
        supported = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise ValueError(
            f"Unknown ArUco dictionary '{dictionary_name}'. Supported examples: "
            f"{', '.join(supported[:8])} ..."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def detect_markers(image: np.ndarray, dictionary_name: str) -> tuple[list[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = get_aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # Compatibility with older OpenCV releases.
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None or len(ids) < 4:
        found = 0 if ids is None else len(ids)
        raise RuntimeError(f"Expected at least 4 ArUco markers, but detected {found}.")
    return list(corners), ids.flatten()


def marker_corners_by_id(corners: Iterable[np.ndarray], ids: Iterable[int]) -> dict[int, np.ndarray]:
    marker_points: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids):
        marker_points[int(marker_id)] = marker_corners.reshape(4, 2).astype(np.float32)
    return marker_points


def marker_centers(corners: Iterable[np.ndarray], ids: Iterable[int]) -> dict[int, np.ndarray]:
    return {
        marker_id: points.mean(axis=0)
        for marker_id, points in marker_corners_by_id(corners, ids).items()
    }


def order_points_geometrically(points: np.ndarray) -> np.ndarray:
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def select_source_points(
    marker_points: dict[int, np.ndarray], marker_ids: list[int] | None
) -> tuple[np.ndarray, list[int]]:
    centers = {marker_id: points.mean(axis=0) for marker_id, points in marker_points.items()}
    if marker_ids:
        missing = [marker_id for marker_id in marker_ids if marker_id not in marker_points]
        if missing:
            raise RuntimeError(f"Required corner marker IDs were not detected: {missing}")
        ordered_ids = marker_ids
    else:
        if len(marker_points) != 4:
            raise RuntimeError(
                "Detected more than four markers. Provide --marker-ids to identify the arena corners."
            )
        ids = list(centers.keys())
        points = np.array([centers[marker_id] for marker_id in ids], dtype=np.float32)
        ordered_centers = order_points_geometrically(points)
        ordered_ids = [
            ids[int(np.argmin(np.linalg.norm(points - point, axis=1)))]
            for point in ordered_centers
        ]

    arena_center = np.array([centers[marker_id] for marker_id in ordered_ids], dtype=np.float32).mean(axis=0)
    source_points = np.array(
        [
            marker_points[marker_id][
                np.argmax(np.linalg.norm(marker_points[marker_id] - arena_center, axis=1))
            ]
            for marker_id in ordered_ids
        ],
        dtype=np.float32,
    )
    return source_points, ordered_ids


def create_map(
    image: np.ndarray, source_points: np.ndarray, width_m: float, height_m: float, ppm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_width = int(round(width_m * ppm))
    output_height = int(round(height_m * ppm))
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Output dimensions must be positive. Check --map-size-mm and --pixels-per-meter.")

    destination_points = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_points, destination_points)
    warped = cv2.warpPerspective(image, homography, (output_width, output_height))
    return warped, destination_points, homography


def write_debug_image(
    image: np.ndarray, corners: list[np.ndarray], ids: np.ndarray, source_points: np.ndarray, path: Path
) -> None:
    debug = image.copy()
    cv2.aruco.drawDetectedMarkers(debug, corners, ids.reshape(-1, 1))
    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, source_points):
        x, y = point.astype(int)
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(debug, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), debug)


def resolve_map_size_m(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.arena_size:
        width_m, height_m = args.arena_size
        width_mm, height_mm = width_m * 1000.0, height_m * 1000.0
    else:
        width_mm, height_mm = args.map_size_mm
        width_m, height_m = width_mm / 1000.0, height_mm / 1000.0
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Map dimensions must be positive.")
    return width_m, height_m, width_mm, height_mm


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    corners, ids = detect_markers(image, args.dictionary)
    marker_points = marker_corners_by_id(corners, ids)
    source_points, selected_ids = select_source_points(marker_points, args.marker_ids)
    width_m, height_m, width_mm, height_mm = resolve_map_size_m(args)
    warped, destination_points, homography = create_map(
        image, source_points, width_m, height_m, args.pixels_per_meter
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), warped)

    if args.debug_image:
        write_debug_image(image, corners, ids, source_points, args.debug_image)

    metadata = MapMetadata(
        input_image=str(args.input),
        output_image=str(args.output),
        width_mm=width_mm,
        height_mm=height_mm,
        width_m=width_m,
        height_m=height_m,
        pixels_per_meter=args.pixels_per_meter,
        output_width_px=warped.shape[1],
        output_height_px=warped.shape[0],
        marker_ids=selected_ids,
        source_points_px=source_points.tolist(),
        destination_points_px=destination_points.tolist(),
        homography=homography.tolist(),
    )
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    else:
        print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
