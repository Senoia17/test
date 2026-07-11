"""High-level video-based global map generation controller."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Mapping, Sequence

from calibration.camera_model import CameraModel
from mapping.aruco_detector import detect_aruco_markers
from mapping.coordinate_system import default_coordinate_system
from mapping.frame_selector import select_best_frame
from mapping.map_homography import FIELD_CORNERS_CM, calculate_map_homography


def _map_output_size(marker_positions: Mapping[int, Sequence[float]]) -> tuple[int, int]:
    xs = [float(point[0]) for point in marker_positions.values()]
    ys = [float(point[1]) for point in marker_positions.values()]
    width = max(1, int(round(max(xs) - min(xs))))
    height = max(1, int(round(max(ys) - min(ys))))
    return width, height


def build_global_map(
    video_path: str | Path,
    *,
    output_dir: str | Path = "data/map",
    camera_model: CameraModel | None = None,
    calibration_path: str | Path | None = None,
    marker_positions: Mapping[int, Sequence[float]] | None = None,
    sample_interval: int = 30,
    dictionary_name: str = "DICT_4X4_50",
) -> dict[str, object]:
    """Build a bird-eye global map from a mapping video and save artifacts."""
    import cv2

    if camera_model is None and calibration_path is not None and Path(calibration_path).exists():
        camera_model = CameraModel.load(calibration_path)

    positions = marker_positions or FIELD_CORNERS_CM
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_frame, metadata = select_best_frame(
        video_path,
        sample_interval=sample_interval,
        camera_model=camera_model,
        dictionary_name=dictionary_name,
    )
    markers = detect_aruco_markers(best_frame, dictionary_name=dictionary_name)
    homography = calculate_map_homography(markers, marker_positions=positions)

    width, height = _map_output_size(positions)
    global_map = cv2.warpPerspective(best_frame, homography, (width, height))

    global_map_path = output_path / "global_map.jpg"
    map_info_path = output_path / "map_info.json"
    aruco_points_path = output_path / "aruco_points.json"
    homography_path = output_path / "homography.pkl"

    cv2.imwrite(str(global_map_path), global_map)
    default_coordinate_system(positions).save(map_info_path)
    aruco_points_path.write_text(json.dumps({"markers": markers, "best_frame": metadata}, indent=2), encoding="utf-8")
    with homography_path.open("wb") as file:
        pickle.dump(homography, file)

    return {
        "global_map": str(global_map_path),
        "map_info": str(map_info_path),
        "aruco_points": str(aruco_points_path),
        "homography": str(homography_path),
        "best_frame": metadata,
    }
