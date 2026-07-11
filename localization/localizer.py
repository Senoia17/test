"""High-level frame localization controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from mapping.map_homography import FIELD_CORNERS_CM
from localization.aruco_pose import localize_with_aruco
from localization.frame_matching import localize_with_frame_matching
from localization.homography_fusion import select_best_homography
from localization.map_matching import localize_with_map_matching


class FrameLocalizer:
    """Localize obstacle video frames into global map coordinates."""

    def __init__(
        self,
        global_map: Any | None = None,
        *,
        global_map_path: str | Path | None = None,
        marker_positions: Mapping[int, Sequence[float]] | None = None,
        max_features: int = 5000,
        ratio_test: float = 0.75,
        ransac_threshold: float = 4.0,
        min_good_matches: int = 20,
    ) -> None:
        if global_map is None and global_map_path is not None:
            import cv2

            global_map = cv2.imread(str(global_map_path), cv2.IMREAD_COLOR)
            if global_map is None:
                raise FileNotFoundError(global_map_path)

        self.global_map = global_map
        self.marker_positions = marker_positions or FIELD_CORNERS_CM
        self.max_features = max_features
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold
        self.min_good_matches = min_good_matches
        self.previous_frame: Any | None = None
        self.previous_result: dict[str, object] | None = None

    def localize(self, current_frame: Any) -> dict[str, object] | None:
        """Run ArUco, map-matching, temporal matching, then select the best H."""
        candidates: list[dict[str, object] | None] = []

        try:
            aruco_result = localize_with_aruco(current_frame, self.marker_positions)
        except Exception as exc:
            aruco_result = {"method": "aruco", "H": None, "confidence": 0.0, "error": str(exc)}
        candidates.append(aruco_result)

        map_result = None
        if self.global_map is not None:
            try:
                map_result = localize_with_map_matching(
                    current_frame,
                    self.global_map,
                    max_features=self.max_features,
                    ratio_test=self.ratio_test,
                    ransac_threshold=self.ransac_threshold,
                    min_good_matches=self.min_good_matches,
                )
            except Exception as exc:
                map_result = {"method": "map_matching", "H": None, "confidence": 0.0, "error": str(exc)}
        candidates.append(map_result)

        frame_result = None
        if self.previous_frame is not None and self.previous_result is not None:
            try:
                frame_result = localize_with_frame_matching(
                    self.previous_frame,
                    current_frame,
                    self.previous_result["H"],
                    max_features=self.max_features,
                    ratio_test=self.ratio_test,
                    ransac_threshold=self.ransac_threshold,
                    min_good_matches=self.min_good_matches,
                )
            except Exception as exc:
                frame_result = {"method": "frame_matching", "H": None, "confidence": 0.0, "error": str(exc)}
        candidates.append(frame_result)

        best = select_best_homography(candidates)
        if best is not None:
            self.previous_frame = current_frame
            self.previous_result = best
        return best
