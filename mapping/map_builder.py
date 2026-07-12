"""Adapter for teammate FastTopViewMosaicBuilder global-map generation."""

from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from calibration.camera_model import CameraModel
from calibration.undistort import undistort_frame
from mapping.aruco_config import get_aruco_config, get_aruco_marker_positions, resolve_corner_ids, resolve_dictionary_name
from mapping.coordinate_system import default_coordinate_system


@dataclass
class MapBuildConfig:
    """Adapter-level configuration forwarded to teammate map builder."""

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
    aruco_config = get_aruco_config()
    cfg = MapBuildConfig(
        frame_stride=max(1, int(sample_interval)),
        aruco_dictionary=dictionary_name or aruco_config.dictionary,
        aruco_corner_ids=aruco_config.corner_ids,
    )
    if marker_positions:
        ordered = sorted(
            marker_positions.items(),
            key=lambda item: (float(item[1][1]), float(item[1][0])),
        )
        if len(ordered) >= 4:
            cfg.aruco_corner_ids = tuple(int(marker_id) for marker_id, _ in ordered[:4])
    return cfg


def _write_undistorted_video(source_video: Path, output_video: Path, camera_model: CameraModel) -> Path:
    """Create the calibrated video consumed unchanged by FastTopViewMosaicBuilder."""
    import cv2

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open mapping video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot create calibrated mapping video: {output_video}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(undistort_frame(frame, camera_model))
    finally:
        writer.release()
        cap.release()

    return output_video


def _rectified_pixel_to_world_homography(
    image_width: int,
    image_height: int,
    marker_positions: Mapping[int, Sequence[float]],
    corner_ids: Sequence[int],
):
    """Create the legacy map-pixel -> world-coordinate artifact only."""
    import cv2
    import numpy as np

    src = np.array(
        [[0.0, 0.0], [float(image_width), 0.0], [float(image_width), float(image_height)], [0.0, float(image_height)]],
        dtype=np.float32,
    )
    missing = [marker_id for marker_id in corner_ids if marker_id not in marker_positions]
    if missing:
        raise KeyError(f"Missing marker positions for configured corner IDs: {missing}")

    dst = np.array([marker_positions[marker_id] for marker_id in corner_ids], dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise ValueError("Failed to compute rectified map artifact homography.")
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
    """Build current map artifacts by adapting inputs to teammate builder.

    Call order is: input video -> existing calibration undistort -> teammate
    ``FastTopViewMosaicBuilder.build`` -> project artifact writers.
    """
    camera_model = _camera_model_from_optional_inputs(camera_model, calibration_path)
    source_video = Path(video_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Mapping video not found: {source_video}")

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

    from mapping.global_localization_teammate import FastTopViewMosaicBuilder, MapBuildConfig as TeammateMapBuildConfig

    global_map_path = output_path / "global_map.jpg"
    debug_dir = output_path / "map_debug"
    builder = FastTopViewMosaicBuilder(TeammateMapBuildConfig(**asdict(cfg)))

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    builder_input = source_video
    try:
        if camera_model is not None:
            temp_dir = tempfile.TemporaryDirectory(prefix="calibrated_mapping_")
            calibrated_video = Path(temp_dir.name) / f"{source_video.stem}_undistorted.mp4"
            builder_input = _write_undistorted_video(source_video, calibrated_video, camera_model)

        global_map, report = builder.build(
            video_path=builder_input,
            output_path=global_map_path,
            debug_dir=debug_dir,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

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
        "teammate_builder": "mapping.global_localization_teammate.FastTopViewMosaicBuilder.build",
        "calibrated_input": camera_model is not None,
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
