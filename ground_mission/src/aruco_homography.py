import argparse
import pickle

import cv2
import numpy as np


FIELD_CORNERS_CM = {
    0: [0.0, 0.0],
    1: [500.0, 0.0],
    2: [500.0, 400.0],
    3: [0.0, 400.0],
}


def detect_aruco_centers(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return {}

    centers = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]
        center = pts.mean(axis=0)
        centers[int(marker_id)] = center.tolist()

    return centers


def compute_homography(centers):
    required_ids = [0, 1, 2, 3]
    missing = [marker_id for marker_id in required_ids if marker_id not in centers]
    if missing:
        raise ValueError(f"Missing ArUco marker IDs: {missing}")

    image_points = np.array([centers[i] for i in required_ids], dtype=np.float32)
    world_points = np.array([FIELD_CORNERS_CM[i] for i in required_ids], dtype=np.float32)

    homography, mask = cv2.findHomography(image_points, world_points)
    if homography is None:
        raise ValueError("Failed to compute homography.")

    return homography


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="configs/homography.pkl")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    centers = detect_aruco_centers(image)
    homography = compute_homography(centers)

    with open(args.output, "wb") as f:
        pickle.dump(homography, f)

    print(f"Saved homography to {args.output}")


if __name__ == "__main__":
    main()

