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
from calibration.undistort import undistort_frame
from mapping.aruco_config import get_aruco_config, get_aruco_marker_positions, resolve_corner_ids, resolve_dictionary_name
from mapping.coordinate_system import default_coordinate_system


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
    aruco_dictionary: str | None = None
    aruco_corner_ids: tuple[int, ...] | None = None
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
    dictionary_name: str | None,
    marker_positions: Mapping[int, Sequence[float]] | None,
) -> MapBuildConfig:
    """Translate previous ArUco-frame API knobs to teammate mapper config."""
    aruco_config = get_aruco_config()
    cfg = MapBuildConfig(
        frame_stride=max(1, int(sample_interval)),
        aruco_dictionary=dictionary_name or aruco_config.dictionary,
        aruco_corner_ids=aruco_config.corner_ids,
    )
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
    corner_ids: Sequence[int],
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
    missing = [marker_id for marker_id in corner_ids if marker_id not in marker_positions]
    if missing:
        raise KeyError(f"Missing marker positions for configured corner IDs: {missing}")

    dst = np.array([marker_positions[marker_id] for marker_id in corner_ids], dtype=np.float32)
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
    dictionary_name: str | None = None,
    map_config: MapBuildConfig | None = None,
) -> dict[str, object]:
    """Build a rectified bird's-eye global map from a top-view mapping video.

    The signature intentionally remains compatible with the original mapping
    implementation. When calibration is provided, each sampled video frame is
    undistorted with the existing calibration module before feature stitching and
    ArUco corner rectification.
    """
    camera_model = _camera_model_from_optional_inputs(camera_model, calibration_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = map_config or _config_from_legacy_args(
        sample_interval=sample_interval,
        dictionary_name=dictionary_name,
        marker_positions=marker_positions,
    )
    if cfg.aruco_dictionary is None:
        cfg.aruco_dictionary = resolve_dictionary_name(None)
    if cfg.aruco_corner_ids is None:
        cfg.aruco_corner_ids = resolve_corner_ids(None)

    global_map_path = output_path / "global_map.jpg"
    debug_dir = output_path / "map_debug"
    from mapping.global_localization_teammate import (
        FastTopViewMosaicBuilder,
        MapBuildConfig as TeammateMapBuildConfig,
    )

    teammate_cfg = TeammateMapBuildConfig(**asdict(cfg))
    builder = FastTopViewMosaicBuilder(
        teammate_cfg,
        frame_transform=(lambda frame: undistort_frame(frame, camera_model)) if camera_model is not None else None,
    )
    global_map, report = builder.build(
        video_path=video_path,
        output_path=global_map_path,
        debug_dir=debug_dir,
    )

    map_info_path = output_path / "map_info.json"
    aruco_points_path = output_path / "aruco_points.json"
    homography_path = output_path / "homography.pkl"

    positions = marker_positions or get_aruco_marker_positions()
    height, width = global_map.shape[:2]
    default_coordinate_system(positions).save(map_info_path)

    aruco_points_path.write_text(
        json.dumps(_legacy_aruco_points_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    homography = _rectified_pixel_to_world_homography(width, height, positions, cfg.aruco_corner_ids)
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
