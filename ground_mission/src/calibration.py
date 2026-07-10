"""Camera calibration and undistortion helpers for drone frames."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Calibration:
    """OpenCV camera calibration parameters."""

    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    image_size: list[int]
    reprojection_error: float | None = None

    def matrix(self) -> np.ndarray:
        return np.array(self.camera_matrix, dtype=np.float64)

    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.distortion_coefficients, dtype=np.float64)


def sorted_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_calibration(path: Path | str) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return Calibration(
            camera_matrix=data["camera_matrix"],
            distortion_coefficients=data["distortion_coefficients"],
            image_size=data["image_size"],
            reprojection_error=data.get("reprojection_error"),
        )
    except KeyError as exc:
        raise ValueError(f"Calibration JSON is missing key: {exc}") from exc


def save_calibration(calibration: Calibration, path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def estimate_chessboard_calibration(
    image_paths: Iterable[Path], chessboard_size: tuple[int, int] = (9, 6), square_size_mm: float = 25.0
) -> Calibration:
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
    return Calibration(camera_matrix.tolist(), distortion_coeffs.reshape(-1).tolist(), list(image_size), float(error))


def undistort_image(image: np.ndarray, calibration: Calibration | None, crop: bool = False) -> np.ndarray:
    """Remove lens distortion while preserving frame size by default."""
    if calibration is None:
        return image
    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(calibration.matrix(), calibration.dist_coeffs(), (width, height), 1, (width, height))
    undistorted = cv2.undistort(image, calibration.matrix(), calibration.dist_coeffs(), None, new_matrix)
    if crop:
        x, y, w, h = roi
        if w > 0 and h > 0:
            return undistorted[y : y + h, x : x + w]
    return undistorted


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Estimate drone camera calibration from chessboard images.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of chessboard calibration images.")
    parser.add_argument("--output", type=Path, default=Path("configs/calibration.json"))
    parser.add_argument("--chessboard-size", type=int, nargs=2, default=(9, 6), metavar=("INNER_X", "INNER_Y"))
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    args = parser.parse_args()

    calibration = estimate_chessboard_calibration(
        sorted_image_paths(args.images_dir), tuple(args.chessboard_size), args.square_size_mm
    )
    save_calibration(calibration, args.output)
    print(f"Saved calibration to {args.output} with reprojection error {calibration.reprojection_error:.4f}")


if __name__ == "__main__":
    main()
