"""Central integrated Ground Mission pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from mission.adapters.calibration_adapter import CalibrationAdapter
from mission.adapters.detection_adapter import DetectionAdapter
from mission.adapters.localization_adapter import LocalizationAdapter
from mission.adapters.output_adapter import write_mission_output
from mission.types import DetectionResult, LocalizationResult
from obstacle.obstacle_analyzer import analyze_obstacles
from utils.video import iter_video_frames


IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".png", ".tif", ".tiff", ".webp"}


def _iter_input_frames(input_path: Path) -> Iterable[tuple[int, Any]]:
    if input_path.is_dir():
        import cv2

        image_paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is not None:
                yield index, frame
        return
    yield from iter_video_frames(input_path)


def _unlocalized_object(detection: DetectionResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": detection.class_name,
        "bbox": detection.bbox,
        "center_pixel": detection.center_pixel,
        "confidence": detection.confidence,
        "localized": False,
        "position": None,
    }
    if detection.class_name in {"missile", "cluster", "dumb"}:
        payload["group"] = "UXO"
    return payload


def _analyze_detections(detections: list[DetectionResult], localization: LocalizationResult) -> dict[str, list[dict[str, object]]]:
    if not localization.localized or localization.homography_for_analysis() is None:
        craters: list[dict[str, object]] = []
        uxos: list[dict[str, object]] = []
        for detection in detections:
            item = _unlocalized_object(detection)
            if detection.class_name == "crater":
                craters.append(item)
            elif detection.class_name in {"missile", "cluster", "dumb"}:
                uxos.append(item)
        return {"craters": craters, "uxos": uxos}

    raw_detections = [detection.to_detector_dict() for detection in detections]
    return analyze_obstacles(raw_detections, localization.homography_for_analysis())


def _save_debug_frame(debug_dir: Path, frame_index: int, name: str, image: Any) -> None:
    import cv2

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"frame_{frame_index:03d}_{name}.jpg"), image)


def _draw_detections(frame: Any, detections: list[DetectionResult]) -> Any:
    import cv2

    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, detection.class_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output


def _save_debug_outputs(
    debug_dir: Path,
    frame_index: int,
    original: Any,
    undistorted: Any,
    detections: list[DetectionResult],
    localization: LocalizationResult,
    global_map_shape: tuple[int, int] | None,
) -> None:
    import cv2

    _save_debug_frame(debug_dir, frame_index, "original", original)
    _save_debug_frame(debug_dir, frame_index, "undistorted", undistorted)
    _save_debug_frame(debug_dir, frame_index, "result", _draw_detections(undistorted, detections))

    if localization.match_visualization is not None:
        _save_debug_frame(debug_dir, frame_index, "matches", localization.match_visualization)
    else:
        matches = undistorted.copy()
        cv2.putText(matches, "feature match visualization unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        _save_debug_frame(debug_dir, frame_index, "matches", matches)

    if localization.inlier_visualization is not None:
        _save_debug_frame(debug_dir, frame_index, "inlier_matches", localization.inlier_visualization)

    if localization.localized and localization.homography_for_analysis() is not None:
        height, width = global_map_shape or undistorted.shape[:2]
        warped = cv2.warpPerspective(undistorted, localization.homography_for_analysis(), (width, height))
        _save_debug_frame(debug_dir, frame_index, "warped", warped)


class GroundMissionPipeline:
    """Coordinate calibration, detection, localization, analysis, and output."""

    def __init__(self, config: dict[str, Any], *, debug: bool = False, output: str | Path | None = None) -> None:
        self.config = config
        self.debug = debug
        self.output = output
        paths = config.get("paths", {})
        models = config.get("models", {})
        mission_config = config.get("missions", {}).get("obstacle", {})

        self.calibration = CalibrationAdapter(paths.get("calibration"))
        self.detection = DetectionAdapter(
            models.get("obstacle_weights") or models.get("ground_weights"),
            conf=float(mission_config.get("conf", 0.3)),
            imgsz=int(mission_config.get("imgsz", 1280)),
        )
        localization_config = config.get("localization", {})
        self.localization = LocalizationAdapter(
            paths.get("global_map"),
            paths.get("map_info"),
            min_matches=int(localization_config.get("min_matches", 20)),
            min_inliers=int(localization_config.get("min_inliers", 10)),
            min_inlier_ratio=float(localization_config.get("min_inlier_ratio", 0.4)),
        )

    def run(self, input_path: str | Path) -> dict[str, Any]:
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"Ground mission input not found: {source}")

        debug_dir = None
        if self.debug:
            base_output = Path(self.output) if self.output else Path("outputs")
            debug_dir = (base_output if base_output.suffix.lower() != ".json" else base_output.parent) / "debug"

        global_map_shape = None
        if self.localization.global_map_path and self.localization.global_map_path.exists():
            import cv2

            global_map = cv2.imread(str(self.localization.global_map_path), cv2.IMREAD_COLOR)
            if global_map is not None:
                global_map_shape = global_map.shape[:2]

        frames = []
        craters: list[dict[str, object]] = []
        uxos: list[dict[str, object]] = []

        for frame_index, frame in _iter_input_frames(source):
            undistorted = self.calibration.undistort(frame)
            detections = self.detection.detect(undistorted)
            localization = self.localization.localize(undistorted)
            analyzed = _analyze_detections(detections, localization)

            status = "SUCCESS" if localization.success else (localization.message or "FAILED")
            print(
                f"Frame {frame_index:03d}: matches: {localization.num_matches} "
                f"inliers: {localization.num_inliers} ratio: {localization.inlier_ratio:.2f} "
                f"status: {status}"
            )

            for crater in analyzed["craters"]:
                crater = dict(crater)
                crater["frame_index"] = frame_index
                crater["localized"] = localization.localized
                craters.append(crater)
            for uxo in analyzed["uxos"]:
                uxo = dict(uxo)
                uxo["frame_index"] = frame_index
                uxo["localized"] = localization.localized
                uxos.append(uxo)

            frame_payload = {
                "frame_index": frame_index,
                "detections": [detection.to_json() for detection in detections],
                "localization": localization.to_json(),
                "craters": len(analyzed["craters"]),
                "uxos": len(analyzed["uxos"]),
            }
            frames.append(frame_payload)

            if debug_dir is not None:
                _save_debug_outputs(debug_dir, frame_index, frame, undistorted, detections, localization, global_map_shape)

        payload = {
            "mission": "ground",
            "input": str(source),
            "status": "ok" if frames else "no_frames",
            "adapters": {
                "calibration": {"enabled": self.calibration.enabled, "error": self.calibration.error},
                "detection": {"available": self.detection.available, "error": self.detection.error},
                "localization": {"error": self.localization.error},
            },
            "frames": frames,
            "results": {"craters": craters, "uxos": uxos},
        }
        output_path = write_mission_output(payload, self.output)
        payload["output"] = str(output_path)
        return payload


def run_ground_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    *,
    debug: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run the central integrated Ground Mission pipeline."""
    if dry_run:
        print("[DRY-RUN] mission=ground adapter=mission.pipeline.GroundMissionPipeline")
        return {"mission": "ground", "adapter": "mission.pipeline.GroundMissionPipeline"}
    if input_path is None:
        configured_input = config.get("paths", {}).get("obstacle_video") or config.get("paths", {}).get("input")
        input_path = Path(configured_input) if configured_input else None
    if input_path is None:
        raise ValueError("Ground mission requires --input or paths.obstacle_video/paths.input in config.yaml")
    return GroundMissionPipeline(config, debug=debug, output=output_path).run(input_path)
