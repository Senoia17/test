"""ArUco video-based camera calibration helpers."""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .camera_model import CameraModel

MIN_CALIBRATION_OBSERVATIONS = 30


@dataclass(frozen=True)
class ArucoCalibrationConfig:
    """Validated ArUco settings needed only while calibrating a camera."""

    dictionary: str
    marker_ids: tuple[int, ...]
    marker_length: float
    sample_interval: int


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = (part.strip() for part in line.strip().split(":", 1))
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_calibration_config(path: Path | str = Path("config.yaml")) -> ArucoCalibrationConfig:
    """Load and validate calibration settings from the project YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        data = _load_yaml_fallback(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

    mapping_aruco = data.get("mapping", {}).get("aruco", {})
    calibration_aruco = data.get("calibration", {}).get("aruco", {})
    dictionary = mapping_aruco.get("dictionary")
    marker_ids = calibration_aruco.get("marker_ids")
    marker_length = calibration_aruco.get("marker_length")
    sample_interval = calibration_aruco.get("sample_interval")

    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError("mapping.aruco.dictionary must be a non-empty string")
    if not isinstance(marker_ids, Sequence) or isinstance(marker_ids, (str, bytes)) or not marker_ids:
        raise ValueError("calibration.aruco.marker_ids must be a non-empty sequence")
    if marker_length is None:
        raise ValueError("calibration.aruco.marker_length must be set to the measured marker side length")
    if float(marker_length) <= 0:
        raise ValueError("calibration.aruco.marker_length must be positive")
    if not isinstance(sample_interval, int) or isinstance(sample_interval, bool) or sample_interval <= 0:
        raise ValueError("calibration.aruco.sample_interval must be a positive integer")

    ids = tuple(int(marker_id) for marker_id in marker_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("calibration.aruco.marker_ids must not contain duplicates")
    return ArucoCalibrationConfig(dictionary, ids, float(marker_length), sample_interval)


def _aruco_detector(dictionary_name: str) -> Any:
    import cv2

    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    parameters = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def _detect(detector: Any, gray: Any) -> tuple[list[Any], Any]:
    import cv2

    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return list(corners), ids


def collect_aruco_observations(
    video_paths: Iterable[Path | str], config: ArucoCalibrationConfig
) -> tuple[list[Any], list[Any], tuple[int, int]]:
    """Collect valid single-marker observations from all calibration videos."""
    import cv2
    import numpy as np

    paths = [Path(path) for path in video_paths]
    if len(paths) != 4:
        raise ValueError(f"Exactly four calibration videos are required; received {len(paths)}")

    half = config.marker_length / 2.0
    # detectMarkers() returns TL, TR, BR, BL. These centered object points use
    # the same order as OpenCV's standard single-marker pose convention.
    object_template = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    valid_ids = set(config.marker_ids)
    detector = _aruco_detector(config.dictionary)
    object_points: list[Any] = []
    image_points: list[Any] = []
    image_size: tuple[int, int] | None = None

    for video_path in paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open calibration video: {video_path}")
        frame_index = 0
        video_observations = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % config.sample_interval != 0:
                    frame_index += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_size = gray.shape[::-1]
                if image_size is None:
                    image_size = current_size
                elif current_size != image_size:
                    raise ValueError(f"Calibration videos have inconsistent frame sizes: {image_size} and {current_size}")
                corners, ids = _detect(detector, gray)
                if ids is not None and len(ids) == 1:
                    marker_id = int(ids.flatten()[0])
                    if marker_id in valid_ids:
                        object_points.append(object_template.copy())
                        image_points.append(corners[0].reshape(4, 2).astype(np.float32))
                        video_observations += 1
                frame_index += 1
        finally:
            capture.release()
        if video_observations == 0:
            warnings.warn(
                f"Skipping calibration video with no valid single-marker observations: {video_path}",
                RuntimeWarning,
                stacklevel=2,
            )

    if image_size is None:
        raise RuntimeError("No frames could be read from the calibration videos")
    if len(image_points) < MIN_CALIBRATION_OBSERVATIONS:
        raise RuntimeError(
            f"Only {len(image_points)} valid ArUco observations found; "
            f"at least {MIN_CALIBRATION_OBSERVATIONS} are required"
        )
    return object_points, image_points, image_size


def estimate_aruco_calibration(
    video_paths: Iterable[Path | str], config: ArucoCalibrationConfig
) -> CameraModel:
    """Estimate intrinsics from independently viewed, known-size ArUco markers."""
    import cv2

    object_points, image_points, image_size = collect_aruco_observations(video_paths, config)
    error, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraModel(camera_matrix.tolist(), distortion_coeffs.reshape(-1).tolist(), list(image_size), float(error))


calibrate_camera = estimate_aruco_calibration
