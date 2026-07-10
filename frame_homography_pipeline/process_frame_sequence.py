#!/usr/bin/env python3
"""Align a time-ordered folder of drone frames to an existing top-down map.

Pipeline:
1. Optionally load or estimate camera lens calibration.
2. Apply the same undistortion to the global map and every drone frame.
3. Match features between the previous image and the current frame.
   - For the first frame, the previous image is the global map.
   - For later frames, the previous image is the immediately preceding frame.
4. Estimate the current frame's transform into the global map coordinates by
   composing consecutive homographies.
5. Optionally warp every frame into the global map canvas and write metadata.

Important: camera undistortion removes lens distortion only. Homography is still
estimated for every frame because camera pose changes are not corrected by
undistortion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Calibration:
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    image_size: list[int]
    reprojection_error: float | None = None

    def matrix(self) -> np.ndarray:
        return np.array(self.camera_matrix, dtype=np.float64)

    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.distortion_coefficients, dtype=np.float64)


@dataclass(frozen=True)
class FrameAlignment:
    frame: str
    previous_reference: str
    homography_current_to_previous: list[list[float]]
    homography_current_to_map: list[list[float]]
    raw_matches: int
    good_matches: int
    inliers: int
    warped_output: str | None


@dataclass(frozen=True)
class SequenceMetadata:
    map_image: str
    frame_dir: str
    map_width_mm: float
    map_height_mm: float
    map_width_px: int
    map_height_px: int
    undistortion_enabled: bool
    calibration: Calibration | None
    frames: list[FrameAlignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align sequential drone frames to an existing top-down map using feature matching."
    )
    parser.add_argument("--map-image", type=Path, required=True, help="Existing full top-down map image.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing drone frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for warped frames and metadata.")
    parser.add_argument(
        "--map-size-mm",
        type=float,
        nargs=2,
        default=(5000.0, 4000.0),
        metavar=("WIDTH_MM", "HEIGHT_MM"),
        help="Known real-world map size in millimeters. Default: 5000 4000.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help="Optional camera calibration JSON containing camera_matrix and distortion_coefficients.",
    )
    parser.add_argument(
        "--calibration-images-dir",
        type=Path,
        help="Optional chessboard calibration image directory used to estimate lens distortion once.",
    )
    parser.add_argument(
        "--chessboard-size",
        type=int,
        nargs=2,
        metavar=("INNER_CORNERS_X", "INNER_CORNERS_Y"),
        default=(9, 6),
        help="Chessboard inner-corner grid for --calibration-images-dir. Default: 9 6.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=25.0,
        help="Chessboard square size in millimeters for calibration. Default: 25.0.",
    )
    parser.add_argument(
        "--save-calibration-json",
        type=Path,
        help="Optional path where estimated calibration should be saved.",
    )
    parser.add_argument("--metadata", type=Path, help="Optional metadata JSON path. Defaults to output-dir/sequence_metadata.json.")
    parser.add_argument("--no-warped-frames", action="store_true", help="Only write metadata; do not save map-aligned frame images.")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum ORB features per image. Default: 5000.")
    parser.add_argument("--ratio-test", type=float, default=0.75, help="Lowe ratio-test threshold. Default: 0.75.")
    parser.add_argument("--ransac-threshold", type=float, default=4.0, help="RANSAC reprojection threshold in pixels. Default: 4.0.")
    parser.add_argument("--min-good-matches", type=int, default=20, help="Minimum good matches required per frame. Default: 20.")
    return parser.parse_args()


def sorted_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Frame directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda path: path.name)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_calibration(path: Path) -> Calibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return Calibration(
            camera_matrix=data["camera_matrix"],
            distortion_coefficients=data["distortion_coefficients"],
            image_size=data["image_size"],
            reprojection_error=data.get("reprojection_error"),
        )
    except KeyError as exc:
        raise ValueError(f"Calibration JSON is missing key: {exc}") from exc


def save_calibration(calibration: Calibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def estimate_chessboard_calibration(
    image_paths: Iterable[Path], chessboard_size: tuple[int, int], square_size_mm: float
) -> Calibration:
    object_points_template = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    object_points_template[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    object_points_template *= square_size_mm

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
        object_points.append(object_points_template.copy())
        image_points.append(refined)

    if image_size is None:
        raise RuntimeError("No calibration images were found.")
    if len(object_points) < 5:
        raise RuntimeError(
            f"Only {len(object_points)} calibration images had detectable chessboards; at least 5 are recommended."
        )

    error, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return Calibration(
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=distortion_coeffs.reshape(-1).tolist(),
        image_size=[image_size[0], image_size[1]],
        reprojection_error=float(error),
    )


def resolve_calibration(args: argparse.Namespace) -> Calibration | None:
    if args.calibration_json and args.calibration_images_dir:
        raise ValueError("Use either --calibration-json or --calibration-images-dir, not both.")
    if args.calibration_json:
        return load_calibration(args.calibration_json)
    if args.calibration_images_dir:
        calibration = estimate_chessboard_calibration(
            sorted_image_paths(args.calibration_images_dir),
            tuple(args.chessboard_size),
            args.square_size_mm,
        )
        if args.save_calibration_json:
            save_calibration(calibration, args.save_calibration_json)
        return calibration
    return None


def undistort_image(image: np.ndarray, calibration: Calibration | None) -> np.ndarray:
    if calibration is None:
        return image
    camera_matrix = calibration.matrix()
    dist_coeffs = calibration.dist_coeffs()
    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
    x, y, w, h = roi
    if w > 0 and h > 0:
        return undistorted[y : y + h, x : x + w]
    return undistorted


def detect_and_describe(image: np.ndarray, orb: cv2.ORB) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("Could not detect enough features in an image.")
    return keypoints, descriptors


def estimate_homography_between(
    current_image: np.ndarray,
    previous_image: np.ndarray,
    orb: cv2.ORB,
    ratio_test: float,
    ransac_threshold: float,
    min_good_matches: int,
) -> tuple[np.ndarray, int, int, int]:
    current_keypoints, current_descriptors = detect_and_describe(current_image, orb)
    previous_keypoints, previous_descriptors = detect_and_describe(previous_image, orb)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(current_descriptors, previous_descriptors, k=2)
    good_matches = [first for first, second in knn_matches if first.distance < ratio_test * second.distance]
    if len(good_matches) < min_good_matches:
        raise RuntimeError(f"Only {len(good_matches)} good matches found; need at least {min_good_matches}.")

    current_points = np.float32([current_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    previous_points = np.float32([previous_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(current_points, previous_points, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        raise RuntimeError("Homography estimation failed.")
    inliers = int(mask.ravel().sum())
    if inliers < 4:
        raise RuntimeError(f"Homography has too few inliers: {inliers}.")
    return homography, len(knn_matches), len(good_matches), inliers


def process_sequence(args: argparse.Namespace) -> SequenceMetadata:
    frame_paths = sorted_image_paths(args.frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {args.frames_dir}")

    calibration = resolve_calibration(args)
    map_image = undistort_image(read_image(args.map_image), calibration)
    map_height_px, map_width_px = map_image.shape[:2]
    output_frames_dir = args.output_dir / "warped_frames"
    if not args.no_warped_frames:
        output_frames_dir.mkdir(parents=True, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=args.max_features)
    previous_image = map_image
    previous_name = str(args.map_image)
    previous_to_map = np.eye(3, dtype=np.float64)
    alignments: list[FrameAlignment] = []

    for frame_path in frame_paths:
        current_image = undistort_image(read_image(frame_path), calibration)
        current_to_previous, raw_matches, good_matches, inliers = estimate_homography_between(
            current_image,
            previous_image,
            orb,
            args.ratio_test,
            args.ransac_threshold,
            args.min_good_matches,
        )
        current_to_map = previous_to_map @ current_to_previous

        warped_output: str | None = None
        if not args.no_warped_frames:
            warped = cv2.warpPerspective(current_image, current_to_map, (map_width_px, map_height_px))
            warped_path = output_frames_dir / frame_path.name
            cv2.imwrite(str(warped_path), warped)
            warped_output = str(warped_path)

        alignments.append(
            FrameAlignment(
                frame=str(frame_path),
                previous_reference=previous_name,
                homography_current_to_previous=current_to_previous.tolist(),
                homography_current_to_map=current_to_map.tolist(),
                raw_matches=raw_matches,
                good_matches=good_matches,
                inliers=inliers,
                warped_output=warped_output,
            )
        )
        previous_image = current_image
        previous_name = str(frame_path)
        previous_to_map = current_to_map

    width_mm, height_mm = args.map_size_mm
    metadata = SequenceMetadata(
        map_image=str(args.map_image),
        frame_dir=str(args.frames_dir),
        map_width_mm=width_mm,
        map_height_mm=height_mm,
        map_width_px=map_width_px,
        map_height_px=map_height_px,
        undistortion_enabled=calibration is not None,
        calibration=calibration,
        frames=alignments,
    )
    metadata_path = args.metadata or (args.output_dir / "sequence_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = process_sequence(args)
    print(
        f"Aligned {len(metadata.frames)} frames to {metadata.map_width_px}x{metadata.map_height_px} px map. "
        f"Metadata saved under: {args.metadata or (args.output_dir / 'sequence_metadata.json')}"
    )


if __name__ == "__main__":
    main()
