"""Frame-to-map homography alignment for drone video/image sequences."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration import Calibration, load_calibration, read_image, sorted_image_paths, undistort_image


@dataclass(frozen=True)
class FrameAlignment:
    frame: str
    previous_reference: str
    homography_current_to_previous: list[list[float]]
    homography_current_to_map_px: list[list[float]]
    homography_current_to_world: list[list[float]] | None
    raw_matches: int
    good_matches: int
    inliers: int
    warped_output: str | None


def detect_and_describe(image: np.ndarray, orb: cv2.ORB) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("Could not detect enough features in an image.")
    return keypoints, descriptors


def estimate_homography_between(
    current_image: np.ndarray,
    previous_image: np.ndarray,
    orb: cv2.ORB,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 20,
) -> tuple[np.ndarray, int, int, int]:
    current_keypoints, current_descriptors = detect_and_describe(current_image, orb)
    previous_keypoints, previous_descriptors = detect_and_describe(previous_image, orb)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(current_descriptors, previous_descriptors, k=2)
    good_matches = [first for first, second in knn_matches if first.distance < ratio_test * second.distance]
    if len(good_matches) < min_good_matches:
        raise RuntimeError(f"Only {len(good_matches)} good matches found; need at least {min_good_matches}.")
    current_points = np.float32([current_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    previous_points = np.float32([previous_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(current_points, previous_points, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        raise RuntimeError("Homography estimation failed.")
    inliers = int(mask.ravel().sum())
    if inliers < 4:
        raise RuntimeError(f"Homography has too few inliers: {inliers}.")
    return homography, len(knn_matches), len(good_matches), inliers


def load_homography_pickle(path: Path | str) -> np.ndarray:
    with open(path, "rb") as file:
        return np.asarray(pickle.load(file), dtype=np.float64)


def align_frame_sequence(
    map_image_path: Path,
    frames_dir: Path,
    output_dir: Path,
    calibration: Calibration | None = None,
    map_to_world_homography: np.ndarray | None = None,
    save_warped_frames: bool = True,
    max_features: int = 5000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 20,
) -> list[FrameAlignment]:
    frame_paths = sorted_image_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {frames_dir}")

    map_image = undistort_image(read_image(map_image_path), calibration)
    map_height_px, map_width_px = map_image.shape[:2]
    warped_dir = output_dir / "warped_frames"
    if save_warped_frames:
        warped_dir.mkdir(parents=True, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=max_features)
    previous_image = map_image
    previous_name = str(map_image_path)
    previous_to_map = np.eye(3, dtype=np.float64)
    alignments: list[FrameAlignment] = []

    for frame_path in frame_paths:
        current_image = undistort_image(read_image(frame_path), calibration)
        current_to_previous, raw_matches, good_matches, inliers = estimate_homography_between(
            current_image, previous_image, orb, ratio_test, ransac_threshold, min_good_matches
        )
        current_to_map = previous_to_map @ current_to_previous
        current_to_world = map_to_world_homography @ current_to_map if map_to_world_homography is not None else None
        warped_output = None
        if save_warped_frames:
            warped = cv2.warpPerspective(current_image, current_to_map, (map_width_px, map_height_px))
            warped_path = warped_dir / frame_path.name
            cv2.imwrite(str(warped_path), warped)
            warped_output = str(warped_path)
        alignments.append(
            FrameAlignment(
                str(frame_path),
                previous_name,
                current_to_previous.tolist(),
                current_to_map.tolist(),
                current_to_world.tolist() if current_to_world is not None else None,
                raw_matches,
                good_matches,
                inliers,
                warped_output,
            )
        )
        previous_image = current_image
        previous_name = str(frame_path)
        previous_to_map = current_to_map
    return alignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Align sequential drone frames to a top-down map.")
    parser.add_argument("--map-image", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--map-homography", type=Path, help="Optional map-pixel to world-coordinate homography pickle.")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--no-warped-frames", action="store_true")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--min-good-matches", type=int, default=20)
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_json) if args.calibration_json else None
    map_h = load_homography_pickle(args.map_homography) if args.map_homography else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignments = align_frame_sequence(
        args.map_image,
        args.frames_dir,
        args.output_dir,
        calibration=calibration,
        map_to_world_homography=map_h,
        save_warped_frames=not args.no_warped_frames,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        ransac_threshold=args.ransac_threshold,
        min_good_matches=args.min_good_matches,
    )
    metadata_path = args.metadata or (args.output_dir / "sequence_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps([asdict(item) for item in alignments], indent=2), encoding="utf-8")
    print(f"Aligned {len(alignments)} frames. Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
