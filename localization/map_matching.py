"""Current-frame to global-map feature matching localization."""

from __future__ import annotations

from typing import Any

from localization.quality import STATUS_SUCCESS, validate_match_quality


def detect_and_describe(image: Any, orb: Any) -> tuple[list[Any], Any]:
    """Detect ORB features using the legacy frame-alignment preprocessing."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("Could not detect enough features in an image.")
    return keypoints, descriptors


def estimate_homography_between(
    current_image: Any,
    reference_image: Any,
    *,
    max_features: int = 5000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 20,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.4,
) -> dict[str, object]:
    """Estimate a homography from current frame coordinates to reference coordinates."""
    import cv2
    import numpy as np

    orb = cv2.ORB_create(nfeatures=max_features)
    current_keypoints, current_descriptors = detect_and_describe(current_image, orb)
    reference_keypoints, reference_descriptors = detect_and_describe(reference_image, orb)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(current_descriptors, reference_descriptors, k=2)
    good_matches = [first for first, second in knn_matches if first.distance < ratio_test * second.distance]
    if len(good_matches) < min_good_matches:
        raise RuntimeError(f"FAILED_TOO_FEW_MATCHES: Only {len(good_matches)} good matches found; need at least {min_good_matches}.")

    current_points = np.float32([current_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    reference_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(current_points, reference_points, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        raise RuntimeError("FAILED_INVALID_HOMOGRAPHY: Homography estimation failed.")

    mask_values = mask.ravel().astype(bool)
    inliers = int(mask_values.sum())
    inlier_ratio = inliers / max(float(len(good_matches)), 1.0)
    if inliers >= 1:
        projected = cv2.perspectiveTransform(current_points, homography)
        errors = np.linalg.norm(projected.reshape(-1, 2) - reference_points.reshape(-1, 2), axis=1)
        reprojection_error = float(errors[mask_values].mean())
    else:
        reprojection_error = None

    valid, status = validate_match_quality(
        homography=homography,
        num_matches=len(good_matches),
        num_inliers=inliers,
        inlier_ratio=inlier_ratio,
        min_matches=min_good_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
    )
    if not valid:
        raise RuntimeError(status)

    raw_match_visualization = cv2.drawMatches(
        current_image, current_keypoints, reference_image, reference_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    inlier_matches = [match for match, is_inlier in zip(good_matches, mask_values) if is_inlier]
    inlier_visualization = cv2.drawMatches(
        current_image, current_keypoints, reference_image, reference_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "H": homography,
        "raw_matches": len(knn_matches),
        "matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "reprojection_error": reprojection_error,
        "num_keypoints": len(current_keypoints),
        "num_reference_keypoints": len(reference_keypoints),
        "confidence": min(1.0, inlier_ratio),
        "status": STATUS_SUCCESS,
        "message": STATUS_SUCCESS,
        "match_visualization": raw_match_visualization,
        "inlier_visualization": inlier_visualization,
    }


def localize_with_map_matching(
    current_frame: Any,
    global_map: Any,
    *,
    max_features: int = 5000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 20,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.4,
) -> dict[str, object]:
    """Match a current frame directly to the generated global map."""
    result = estimate_homography_between(
        current_frame,
        global_map,
        max_features=max_features,
        ratio_test=ratio_test,
        ransac_threshold=ransac_threshold,
        min_good_matches=min_good_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
    )
    result["method"] = "map_matching"
    return result
