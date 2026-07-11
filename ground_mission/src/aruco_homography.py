"""Backward-compatible ArUco homography CLI wrappers.

Phase 5 keeps this legacy entry point available while delegating reusable ArUco
and homography logic to the shared mapping package.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mapping.aruco_detector import detect_aruco_centers  # noqa: E402
from mapping.map_homography import FIELD_CORNERS_CM, compute_homography  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="configs/homography.pkl")
    args = parser.parse_args()

    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    centers = detect_aruco_centers(image)
    homography = compute_homography(centers)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(homography, file)

    print(f"Saved homography to {output_path}")


if __name__ == "__main__":
    main()
