"""Chessboard camera calibration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .camera_model import CameraModel

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".tif", ".tiff", ".webp"}


def sorted_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def read_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def estimate_chessboard_calibration(
    image_paths: Iterable[Path], chessboard_size: tuple[int, int] = (9, 6), square_size_mm: float = 25.0
) -> CameraModel:
    import cv2
    import numpy as np

    object_template = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    object_template *= square_size_mm

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = read_image(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, chessboard_size)
        if not found:
            continue
        refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_template.copy())
        image_points.append(refined)

    if image_size is None:
        raise RuntimeError("No calibration images were found.")
    if len(object_points) < 5:
        raise RuntimeError(f"Only {len(object_points)} usable chessboard images found; at least 5 are recommended.")

    error, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraModel(camera_matrix.tolist(), distortion_coeffs.reshape(-1).tolist(), list(image_size), float(error))


# Descriptive target-architecture name; points to the unchanged existing implementation.
calibrate_camera = estimate_chessboard_calibration
