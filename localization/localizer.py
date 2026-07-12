"""High-level frame localization controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from mapping.map_homography import FIELD_CORNERS_CM


DEFAULT_ROUTE = [
    "TW-A5", "TW-A4", "TW-A3", "TW-A2", "TW-A1",
    "RW-01", "RW-02", "RW-03", "RW-04", "RW-05", "RW-06", "RW-07", "RW-08", "RW-09", "RW-10",
    "TW-B5", "TW-B4", "TW-B3", "TW-B2", "TW-B1",
]


class FrameLocalizer:
    """Localize calibrated frames and return frame-to-global homographies.

    This adapter keeps the project-level ``FrameLocalizer`` API while using the
    teammate region-feature, route-gated, homography-validated localizer. Callers
    must pass frames that have already gone through the existing calibration
    module; mission pipelines do that before invoking ``localize``.
    """

    def __init__(
        self,
        global_map: Any | None = None,
        *,
        global_map_path: str | Path | None = None,
        marker_positions: Mapping[int, Sequence[float]] | None = None,
        route: Sequence[str] | None = None,
        localizer_config: Any | None = None,
        **_: object,
    ) -> None:
        if global_map is None and global_map_path is not None:
            import cv2

            global_map = cv2.imread(str(global_map_path), cv2.IMREAD_COLOR)
            if global_map is None:
                raise FileNotFoundError(global_map_path)
        if global_map is None:
            self.global_map = None
            self.marker_positions = marker_positions or FIELD_CORNERS_CM
            self.localizer = None
            self.previous_gray = None
            self.previous_result = None
            return

        from mapping.global_localization_teammate import CloseupToGlobalLocalizer, RouteState, make_default_airfield_regions

        self.global_map = global_map
        self.marker_positions = marker_positions or FIELD_CORNERS_CM
        self.regions = make_default_airfield_regions()
        self.route_state = RouteState(list(route or DEFAULT_ROUTE), initial_wide_search=True)
        self.localizer = CloseupToGlobalLocalizer(global_map, self.regions, config=localizer_config)
        self.previous_gray: Any | None = None
        self.previous_result: Any | None = None

    def localize(self, current_frame: Any) -> dict[str, object] | None:
        """Return localization payload including ``H_frame_to_global``.

        The obstacle analysis pipeline consumes the same matrix under ``H`` for
        backward compatibility, while richer fields are preserved for mission JSON.
        """
        if self.localizer is None:
            return None
        prev_H = self.previous_result.H_frame_to_global if self.previous_result is not None else None
        prev_center = self.previous_result.center_px if self.previous_result is not None else None
        result, gray = self.localizer.localize_frame(
            current_frame,
            self.route_state,
            prev_gray=self.previous_gray,
            prev_H_frame_to_global=prev_H,
            prev_center_px=prev_center,
        )
        self.previous_gray = gray
        if not result.ok or result.H_frame_to_global is None:
            return {
                "method": result.source,
                "H": None,
                "H_frame_to_global": None,
                "confidence": 0.0,
                "zone": result.zone,
                "center_px": result.center_px,
                "center_m": result.center_m,
                "error": result.reason or "localization_failed",
            }

        self.previous_result = result
        self.route_state.update(result.zone)
        confidence = float(max(0.0, min(1.0, result.inlier_ratio)))
        return {
            "method": result.source,
            "H": result.H_frame_to_global,
            "H_frame_to_global": result.H_frame_to_global,
            "confidence": confidence,
            "zone": result.zone,
            "center_px": result.center_px,
            "center_m": result.center_m,
            "matches": result.matches,
            "inliers": result.inliers,
            "inlier_ratio": result.inlier_ratio,
            "reproj_error": result.reproj_error,
            "area_ratio": result.area_ratio,
            "route_idx": result.route_idx,
            "error": None,
        }
