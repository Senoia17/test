"""End-to-end obstacle mission pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from calibration.camera_model import CameraModel
from calibration.undistort import undistort_frame
from detection.object_detector import ObjectDetector
from localization.localizer import FrameLocalizer
from mission.json_writer import write_json
from mission.model_weights import resolve_model_weight_path
from mission.zone_regions import UNKNOWN_ZONE, load_zone_regions
from obstacle.obstacle_analyzer import analyze_obstacles
from utils.video import iter_video_frames


def _load_camera_model(path: str | Path | None) -> CameraModel | None:
    if path is None:
        return None
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    return CameraModel.load(calibration_path)


def _load_marker_positions(map_info_path: str | Path | None) -> dict[int, list[float]] | None:
    if map_info_path is None:
        return None
    path = Path(map_info_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    markers = data.get("markers")
    if not isinstance(markers, Mapping):
        return None
    return {int(marker_id): [float(point[0]), float(point[1])] for marker_id, point in markers.items()}


def _load_pickle(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    with artifact_path.open("rb") as file:
        return pickle.load(file)


def _require_existing_file(path: str | Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"Obstacle mission requires {label}")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    return file_path


def _rank_regions(votes: Mapping[str, int], limit: int) -> list[dict[str, object]]:
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    return [{"zone": zone, "votes": count} for zone, count in ranked[: max(0, limit)]]


def _record_region_votes(items: list[dict[str, object]], votes: dict[str, int]) -> None:
    for item in items:
        zone = item.get("zone")
        if isinstance(zone, str) and zone and zone != UNKNOWN_ZONE:
            votes[zone] = votes.get(zone, 0) + 1


def _unlocalized_detection(detection: Mapping[str, Any]) -> dict[str, object]:
    class_name = str(detection.get("class"))
    result: dict[str, object] = {
        "type": class_name,
        "bbox": list(detection.get("bbox", [])),
        "confidence": detection.get("confidence"),
        "localized": False,
        "position": None,
    }
    if class_name in {"missile", "cluster", "dumb"}:
        result["group"] = "UXO"
    return result


def run_legacy_obstacle_pipeline(input_path: Path | None = None) -> None:
    """Run the preserved legacy ObstacleDetection pipeline on demand."""
    import importlib.util
    import sys

    legacy_obstacle_dir = Path(__file__).resolve().parents[1] / "ground_mission" / "ObstacleDetection"
    if str(legacy_obstacle_dir) not in sys.path:
        sys.path.insert(0, str(legacy_obstacle_dir))

    legacy_main_path = legacy_obstacle_dir / "main.py"
    spec = importlib.util.spec_from_file_location("legacy_obstacle_main", legacy_main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy obstacle pipeline: {legacy_main_path}")
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    if input_path is None:
        legacy_module.run_pipeline()
    else:
        legacy_module.run_pipeline(video_path=Path(input_path))


def run_obstacle_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run obstacle detection, localization, analysis, and JSON output."""
    if dry_run:
        print("[DRY-RUN] mission=obstacle adapter=mission.obstacle_pipeline")
        return {"mission": "obstacle", "adapter": "mission.obstacle_pipeline"}

    paths_config = config.get("paths", {})
    mission_config = config.get("missions", {}).get("obstacle", {})

    video_path = input_path
    if video_path is None:
        configured_input = paths_config.get("obstacle_video") or paths_config.get("input")
        video_path = Path(configured_input) if configured_input else None
    video_path = _require_existing_file(video_path, "obstacle video")

    weights_path = resolve_model_weight_path(config, "obstacle")
    weights_path = _require_existing_file(weights_path, "obstacle YOLO weights")

    calibration_path = _require_existing_file(paths_config.get("calibration"), "calibration file")

    if output_path is None:
        output_dir = Path(paths_config.get("output_dir", "outputs"))
        output_path = output_dir / "obstacle_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_map_path = _require_existing_file(paths_config.get("global_map", "data/map/global_map.jpg"), "global map artifact")
    map_info_path = _require_existing_file(paths_config.get("map_info", "data/map/map_info.json"), "map info artifact")
    configured_homography = paths_config.get("homography")
    homography_path = Path(configured_homography) if configured_homography else Path(paths_config.get("map_output_dir", "data/map")) / "homography.pkl"
    if not homography_path.exists():
        fallback_homography = Path(paths_config.get("map_output_dir", "data/map")) / "homography.pkl"
        homography_path = fallback_homography if fallback_homography.exists() else homography_path
    homography_path = _require_existing_file(homography_path, "homography artifact")

    zone_lookup = None
    configured_zones = paths_config.get("zones")
    if configured_zones:
        zones_path = _require_existing_file(configured_zones, "zones artifact")
        zone_lookup = load_zone_regions(zones_path).lookup

    camera_model = CameraModel.load(calibration_path)
    marker_positions = _load_marker_positions(map_info_path)
    homography_artifact = _load_pickle(homography_path)

    sample_interval = max(1, int(mission_config.get("sample_interval", 1)))

    detector = ObjectDetector(
        weights_path,
        conf=float(mission_config.get("conf", 0.3)),
        imgsz=int(mission_config.get("imgsz", 1280)),
    )
    localizer = FrameLocalizer(
        global_map_path=global_map_path if global_map_path.exists() else None,
        marker_positions=marker_positions,
    )

    crater_results: list[dict[str, object]] = []
    uxo_results: list[dict[str, object]] = []
    frame_summaries: list[dict[str, object]] = []
    crater_region_votes: dict[str, int] = {}
    uxo_region_votes: dict[str, int] = {}

    processed_frame_count = 0
    for frame_index, frame in iter_video_frames(video_path):
        if frame_index % sample_interval != 0:
            continue
        processed_frame_count += 1
        if processed_frame_count == 1 or processed_frame_count % 50 == 0:
            print(f"[Obstacle] Processing frame {frame_index}")

        undistorted = undistort_frame(frame, camera_model)
        detections = detector.detect(undistorted)
        localization = localizer.localize(undistorted)
        H_frame_to_global = None
        if localization is not None:
            H_frame_to_global = localization.get("H_frame_to_global")
            if H_frame_to_global is None:
                H_frame_to_global = localization.get("H")

        frame_summary: dict[str, object] = {
            "frame_index": frame_index,
            "detections": len(detections),
            "localized": H_frame_to_global is not None,
        }

        if localization is None or H_frame_to_global is None:
            frame_summary["reason"] = "localization_failed"
            for detection in detections:
                item = _unlocalized_detection(detection)
                item["frame_index"] = frame_index
                if item.get("type") == "crater":
                    crater_results.append(item)
                elif item.get("type") in {"missile", "cluster", "dumb"}:
                    uxo_results.append(item)
            frame_summaries.append(frame_summary)
            continue

        analyzed = analyze_obstacles(detections, H_frame_to_global, zone_lookup=zone_lookup)
        _record_region_votes(analyzed["craters"], crater_region_votes)
        _record_region_votes(analyzed["uxos"], uxo_region_votes)
        localization_method = localization.get("method")
        localization_confidence = localization.get("confidence")

        for crater in analyzed["craters"]:
            crater = dict(crater)
            crater["frame_index"] = frame_index
            crater["localization_method"] = localization_method
            crater["localized"] = True
            crater_results.append(crater)

        for uxo in analyzed["uxos"]:
            uxo = dict(uxo)
            uxo["frame_index"] = frame_index
            uxo["localization_method"] = localization_method
            uxo["localized"] = True
            uxo_results.append(uxo)

        frame_summary.update(
            {
                "localization_method": localization_method,
                "localization_confidence": localization_confidence,
                "center_px": localization.get("center_px"),
                "center_m": localization.get("center_m"),
                "zone": localization.get("zone"),
                "craters": len(analyzed["craters"]),
                "uxos": len(analyzed["uxos"]),
            }
        )
        frame_summaries.append(frame_summary)

    crater_region_count = int(mission_config.get("crater_count", len(crater_region_votes)))
    uxo_region_count = int(mission_config.get("uxo_count", len(uxo_region_votes)))

    payload = {
        "mission": "obstacle",
        "input": str(video_path),
        "map_artifacts": {
            "global_map": str(global_map_path),
            "map_info": str(paths_config.get("map_info", "data/map/map_info.json")),
            "homography": str(homography_path),
            "homography_loaded": homography_artifact is not None,
            "calibration": str(calibration_path),
        },
        "frames": frame_summaries,
        "region_votes": {
            "craters": crater_region_votes,
            "uxos": uxo_region_votes,
        },
        "selected_regions": {
            "craters": _rank_regions(crater_region_votes, crater_region_count),
            "uxos": _rank_regions(uxo_region_votes, uxo_region_count),
        },
        "results": {
            "craters": crater_results,
            "uxos": uxo_results,
        },
    }
    write_json(payload, output_path)
    return payload
