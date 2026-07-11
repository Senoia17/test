"""High-level video-based global map generation controller.

The implementation delegates map construction to the teammate's feature-based
mosaic builder while preserving the public ``build_global_map`` interface used
by mission, calibration, localization, and ground-mission adapters.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from calibration.camera_model import CameraModel
from mapping.coordinate_system import default_coordinate_system
from mapping.map_homography import FIELD_CORNERS_CM


@dataclass
class MapBuildConfig:
    """Configuration for teammate feature-mosaic map building."""

    frame_stride: int = 10
    max_keyframes: int = 140
    resize_width: int = 960
    prefer_sift: bool = False
    orb_features: int = 5000
    sift_features: int = 4000
    max_match_candidates: int = 500
    min_good_matches: int = 45
    min_inliers: int = 35
    min_inlier_ratio: float = 0.20
    ransac_reproj_thr: float = 5.0
    max_canvas_side: int = 5200
    crop_black_border: bool = True
    save_keyframe_debug: bool = False
    use_aruco_rectification: bool = True
    aruco_dictionary: str = "DICT_5X5_50"
    aruco_corner_ids: tuple[int, int, int, int] | None = None
    aruco_min_markers: int = 4
    rectified_map_width: int = 1500
    rectified_map_height: int = 1200
    aruco_rotate_portrait_to_landscape: bool = True
    aruco_portrait_rotation: str = "ccw"
    aruco_best_frame_fallback: bool = True
    aruco_best_frame_stride: int = 5
    aruco_debug: bool = True


def _camera_model_from_optional_inputs(
    camera_model: CameraModel | None,
    calibration_path: str | Path | None,
) -> CameraModel | None:
    """Validate legacy calibration inputs without forcing the new mapper to use them."""
    if camera_model is not None:
        return camera_model
    if calibration_path is None:
        return None
    calibration_file = Path(calibration_path)
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    return CameraModel.load(calibration_file)



def _config_from_legacy_args(
    *,
    sample_interval: int,
    dictionary_name: str,
    marker_positions: Mapping[int, Sequence[float]] | None,
) -> MapBuildConfig:
    """Translate the previous ArUco-frame API knobs to the teammate mapper config."""
    cfg = MapBuildConfig(frame_stride=max(1, int(sample_interval)))
    if dictionary_name:
        cfg.aruco_dictionary = dictionary_name
    if marker_positions:
        # Previous callers may pass marker positions to identify the four corner
        # IDs. The teammate mapper expects IDs in final TL, TR, BR, BL order; sort
        # by normalized/yx position as a best-effort compatibility bridge.
        ordered = sorted(
            marker_positions.items(),
            key=lambda item: (float(item[1][1]), float(item[1][0])),
        )
        if len(ordered) >= 4:
            cfg.aruco_corner_ids = tuple(int(marker_id) for marker_id, _ in ordered[:4])  # type: ignore[assignment]
    return cfg


def _rectified_pixel_to_world_homography(
    image_width: int,
    image_height: int,
    marker_positions: Mapping[int, Sequence[float]],
):
    """Return a global-map-pixel to world-coordinate homography.

    The legacy homography artifact represented image pixels in the generated map
    coordinate system. With the teammate mapper, the generated image is already
    rectified, so this artifact maps rectified map pixels to the same coordinate
    system saved in ``map_info.json``.
    """
    import cv2
    import numpy as np

    src = np.array(
        [
            [0.0, 0.0],
            [float(image_width), 0.0],
            [float(image_width), float(image_height)],
            [0.0, float(image_height)],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            marker_positions[0],
            marker_positions[1],
            marker_positions[2],
            marker_positions[3],
        ],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise ValueError("Failed to compute rectified map homography.")
    return homography


def _legacy_aruco_points_payload(report: Mapping[str, object]) -> dict[str, object]:
    aruco_report = report.get("aruco_rectification", {})
    if not isinstance(aruco_report, Mapping):
        aruco_report = {}

    markers = aruco_report.get("rectified_selected_markers_tl_tr_br_bl", [])
    if not isinstance(markers, list):
        markers = []

    best_frame = {
        "source": aruco_report.get("source"),
        "frame_index": aruco_report.get("best_frame_index"),
        "sampled_frames": report.get("sampled_frames"),
        "accepted_keyframes": report.get("accepted_keyframes"),
        "accepted_source_indices": report.get("accepted_source_indices"),
        "aruco_rectification": aruco_report,
    }
    return {"markers": markers, "best_frame": best_frame}


def build_global_map(
    video_path: str | Path,
    *,
    output_dir: str | Path = "data/map",
    camera_model: CameraModel | None = None,
    calibration_path: str | Path | None = None,
    marker_positions: Mapping[int, Sequence[float]] | None = None,
    sample_interval: int = 10,
    dictionary_name: str = "DICT_5X5_50",
    map_config: MapBuildConfig | None = None,
) -> dict[str, object]:
    """Build a rectified bird's-eye global map from a top-view mapping video.

    The signature intentionally remains compatible with the original mapping
    implementation. Calibration arguments are validated for existing callers, but
    the teammate implementation performs feature stitching and ArUco corner
    rectification directly from the mapping video.
    """
    _camera_model_from_optional_inputs(camera_model, calibration_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = map_config or _config_from_legacy_args(
        sample_interval=sample_interval,
        dictionary_name=dictionary_name,
        marker_positions=marker_positions,
    )

    global_map_path = output_path / "global_map.jpg"
    debug_dir = output_path / "map_debug"
    from mapping.global_localization_teammate import (
        FastTopViewMosaicBuilder,
        MapBuildConfig as TeammateMapBuildConfig,
    )

    teammate_cfg = TeammateMapBuildConfig(**asdict(cfg))
    builder = FastTopViewMosaicBuilder(teammate_cfg)
    global_map, report = builder.build(
        video_path=video_path,
        output_path=global_map_path,
        debug_dir=debug_dir,
    )

    map_info_path = output_path / "map_info.json"
    aruco_points_path = output_path / "aruco_points.json"
    homography_path = output_path / "homography.pkl"

    positions = marker_positions or FIELD_CORNERS_CM
    height, width = global_map.shape[:2]
    default_coordinate_system(positions).save(map_info_path)

    aruco_points_path.write_text(
        json.dumps(_legacy_aruco_points_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    homography = _rectified_pixel_to_world_homography(width, height, positions)
    with homography_path.open("wb") as file:
        pickle.dump(homography, file)

    metadata = {
        "source": "teammate_fast_top_view_mosaic",
        "sampled_frames": report.get("sampled_frames") if isinstance(report, dict) else None,
        "accepted_keyframes": report.get("accepted_keyframes") if isinstance(report, dict) else None,
        "accepted_source_indices": report.get("accepted_source_indices") if isinstance(report, dict) else None,
        "build_report": str(output_path / "global_map_build_report.json"),
        "map_debug_dir": str(debug_dir),
        "config": asdict(cfg),
    }

    return {
        "global_map": str(global_map_path),
        "map_info": str(map_info_path),
        "aruco_points": str(aruco_points_path),
        "homography": str(homography_path),
        "best_frame": metadata,
    }
