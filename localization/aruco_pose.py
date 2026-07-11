"""ArUco-marker-based frame localization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mapping.aruco_config import resolve_corner_ids, resolve_dictionary_name
from mapping.aruco_detector import detect_aruco_markers
from mapping.map_homography import FIELD_CORNERS_CM, calculate_map_homography


def localize_with_aruco(
    frame: Any,
    marker_positions: Mapping[int, Sequence[float]] | None = None,
    *,
    dictionary_name: str | None = None,
    required_ids: Sequence[int] | None = None,
) -> dict[str, object] | None:
    """Estimate current frame-to-map H from visible ArUco markers."""
    dictionary_name = resolve_dictionary_name(dictionary_name)
    required_ids = resolve_corner_ids(required_ids)

    markers = detect_aruco_markers(frame, dictionary_name=dictionary_name)
    positions = marker_positions or FIELD_CORNERS_CM
    visible_required = [marker_id for marker_id in required_ids if any(marker["id"] == marker_id for marker in markers)]
    if len(visible_required) < 4:
        return None

    homography = calculate_map_homography(markers, marker_positions=positions, required_ids=required_ids)
    confidence = min(1.0, len(visible_required) / max(float(len(required_ids)), 1.0))
    return {
        "method": "aruco",
        "H": homography,
        "confidence": confidence,
        "markers": markers,
        "detected_markers": len(markers),
    }
