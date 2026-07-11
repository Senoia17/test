"""Best-frame selection for video-based map generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibration.camera_model import CameraModel
from calibration.undistort import undistort_frame
from mapping.aruco_detector import detect_aruco_markers


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def score_markers(markers: list[dict[str, object]], frame_shape: tuple[int, ...]) -> float:
    """Score marker count, marker area, and image-space distribution."""
    if not markers:
        return 0.0

    height, width = frame_shape[:2]
    frame_area = max(float(width * height), 1.0)
    marker_count = len(markers)
    count_score = min(marker_count, 4) / 4.0

    total_marker_area = sum(_polygon_area(marker["corners"]) for marker in markers)  # type: ignore[arg-type]
    area_score = min(total_marker_area / frame_area * 20.0, 1.0)

    centers = []
    for marker in markers:
        corners = marker["corners"]
        cx = sum(float(point[0]) for point in corners) / 4.0  # type: ignore[index]
        cy = sum(float(point[1]) for point in corners) / 4.0  # type: ignore[index]
        centers.append((cx, cy))

    if len(centers) >= 2:
        xs = [point[0] for point in centers]
        ys = [point[1] for point in centers]
        distribution_score = min(((max(xs) - min(xs)) / max(width, 1) + (max(ys) - min(ys)) / max(height, 1)) / 2.0, 1.0)
    else:
        distribution_score = 0.0

    all_markers_bonus = 0.25 if marker_count >= 4 else 0.0
    return min(1.0, count_score * 0.6 + area_score * 0.2 + distribution_score * 0.2 + all_markers_bonus)


def select_best_frame(
    video_path: str | Path,
    *,
    sample_interval: int = 30,
    camera_model: CameraModel | None = None,
    dictionary_name: str = "DICT_4X4_50",
) -> tuple[Any, dict[str, object]]:
    """Sample a mapping video and return the best ArUco-rich frame plus metadata."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open mapping video: {video_path}")

    best_frame = None
    best_metadata: dict[str, object] | None = None
    frame_index = 0
    sample_interval = max(1, int(sample_interval))

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % sample_interval == 0:
            candidate = undistort_frame(frame, camera_model)
            markers = detect_aruco_markers(candidate, dictionary_name=dictionary_name)
            score = score_markers(markers, candidate.shape)
            metadata = {
                "frame_index": frame_index,
                "detected_markers": len(markers),
                "score": float(score),
                "marker_ids": [marker["id"] for marker in markers],
            }
            if best_metadata is None or score > float(best_metadata["score"]):
                best_frame = candidate
                best_metadata = metadata
                if len(markers) >= 4 and score >= 0.95:
                    break

        frame_index += 1

    capture.release()

    if best_frame is None or best_metadata is None:
        raise ValueError(f"No frames could be sampled from mapping video: {video_path}")

    return best_frame, best_metadata
