"""Localization adapter for frame-to-global-map homography estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from localization.localizer import FrameLocalizer
from mission.types import LocalizationResult


def _load_marker_positions(map_info_path: str | Path | None) -> dict[int, list[float]] | None:
    if map_info_path is None:
        return None
    path = Path(map_info_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    markers = data.get("markers")
    if not isinstance(markers, Mapping):
        return None
    return {int(marker_id): [float(point[0]), float(point[1])] for marker_id, point in markers.items()}


class LocalizationAdapter:
    """Wrap FrameLocalizer and normalize localization results."""

    def __init__(
        self,
        global_map_path: str | Path | None = None,
        map_info_path: str | Path | None = None,
        *,
        min_matches: int = 20,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.4,
    ) -> None:
        self.global_map_path = Path(global_map_path) if global_map_path else None
        self.map_info_path = Path(map_info_path) if map_info_path else None
        marker_positions = _load_marker_positions(self.map_info_path)
        self.error: str | None = None
        if self.global_map_path is not None and not self.global_map_path.exists():
            self.error = f"Global map not found: {self.global_map_path}"
            self.global_map_path = None
        try:
            self.localizer = FrameLocalizer(
                global_map_path=self.global_map_path if self.global_map_path and self.global_map_path.exists() else None,
                marker_positions=marker_positions,
                min_good_matches=min_matches,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )
        except Exception as exc:
            self.error = f"Localization map initialization failed: {exc}"
            self.localizer = FrameLocalizer(
                marker_positions=marker_positions,
                min_good_matches=min_matches,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )

    def localize(self, frame: Any) -> LocalizationResult:
        result = self.localizer.localize(frame)
        if result is None:
            return LocalizationResult(None, None, "none", False, message="localization_failed", error="localization_failed")
        return LocalizationResult(
            homography_matrix=result.get("H"),
            confidence=float(result["confidence"]) if result.get("confidence") is not None else None,
            method=str(result.get("method", "unknown")),
            localized=result.get("H") is not None,
            success=result.get("H") is not None,
            num_keypoints=int(result.get("num_keypoints", 0) or 0),
            num_matches=int(result.get("matches", 0) or 0),
            num_inliers=int(result.get("inliers", 0) or 0),
            inlier_ratio=float(result.get("inlier_ratio", 0.0) or 0.0),
            reprojection_error=float(result["reprojection_error"]) if result.get("reprojection_error") is not None else None,
            message=str(result.get("message") or result.get("status") or ""),
            error=str(result["error"]) if result.get("error") else None,
            match_visualization=result.get("match_visualization"),
            inlier_visualization=result.get("inlier_visualization"),
        )
