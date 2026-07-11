"""Previous-frame to current-frame temporal localization."""

from __future__ import annotations

from typing import Any

from localization.map_matching import estimate_homography_between


def localize_with_frame_matching(
    previous_frame: Any,
    current_frame: Any,
    previous_frame_to_map_H: Any,
    *,
    max_features: int = 5000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_good_matches: int = 20,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.4,
) -> dict[str, object]:
    """Estimate current frame-to-map H by composing current-to-previous and previous-to-map H."""
    import numpy as np

    relative = estimate_homography_between(
        current_frame,
        previous_frame,
        max_features=max_features,
        ratio_test=ratio_test,
        ransac_threshold=ransac_threshold,
        min_good_matches=min_good_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
    )
    current_to_previous = relative["H"]
    current_to_map = np.array(previous_frame_to_map_H, dtype=np.float64) @ np.array(current_to_previous, dtype=np.float64)
    return {
        "method": "frame_matching",
        "H": current_to_map,
        "relative_H": current_to_previous,
        "matches": relative["matches"],
        "raw_matches": relative["raw_matches"],
        "inliers": relative["inliers"],
        "inlier_ratio": relative.get("inlier_ratio", 0.0),
        "reprojection_error": relative.get("reprojection_error"),
        "num_keypoints": relative.get("num_keypoints", 0),
        "num_reference_keypoints": relative.get("num_reference_keypoints", 0),
        "confidence": relative["confidence"],
        "status": relative.get("status", "SUCCESS"),
        "message": relative.get("message", "SUCCESS"),
        "match_visualization": relative.get("match_visualization"),
        "inlier_visualization": relative.get("inlier_visualization"),
    }
