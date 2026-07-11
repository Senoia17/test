"""Homography candidate selection for localization."""

from __future__ import annotations

from typing import Mapping


METHOD_PRIORITY = {
    "aruco": 3,
    "map_matching": 2,
    "frame_matching": 1,
}


def select_best_homography(
    results: list[Mapping[str, object] | None],
    *,
    aruco_confidence: float = 0.9,
    map_matching_confidence: float = 0.7,
) -> dict[str, object] | None:
    """Select the best localization result without blindly averaging matrices."""
    candidates = [dict(result) for result in results if result is not None and result.get("H") is not None]
    if not candidates:
        return None

    high_aruco = [result for result in candidates if result.get("method") == "aruco" and float(result.get("confidence", 0.0)) >= aruco_confidence]
    if high_aruco:
        return max(high_aruco, key=lambda result: float(result.get("confidence", 0.0)))

    strong_map = [
        result
        for result in candidates
        if result.get("method") == "map_matching" and float(result.get("confidence", 0.0)) >= map_matching_confidence
    ]
    if strong_map:
        return max(strong_map, key=lambda result: (float(result.get("confidence", 0.0)), int(result.get("inliers", 0))))

    return max(
        candidates,
        key=lambda result: (
            METHOD_PRIORITY.get(str(result.get("method")), 0),
            float(result.get("confidence", 0.0)),
            int(result.get("inliers", 0)),
        ),
    )
