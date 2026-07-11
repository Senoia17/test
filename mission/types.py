"""Stable data contracts for the ground mission pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionResult:
    class_name: str
    confidence: float | None
    bbox: list[float]
    center_pixel: list[float]
    class_id: int | None = None

    def to_detector_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalizationResult:
    homography_matrix: Any | None
    confidence: float | None
    method: str
    localized: bool
    success: bool | None = None
    num_keypoints: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float | None = None
    message: str = ""
    error: str | None = None
    match_visualization: Any | None = None
    inlier_visualization: Any | None = None

    def __post_init__(self) -> None:
        if self.success is None:
            object.__setattr__(self, "success", self.localized)

    def homography_for_analysis(self) -> Any | None:
        return self.homography_matrix

    def to_json(self) -> dict[str, Any]:
        homography = self.homography_matrix
        if hasattr(homography, "tolist"):
            homography = homography.tolist()
        return {
            "homography_matrix": homography,
            "confidence": self.confidence,
            "method": self.method,
            "localized": self.localized,
            "success": self.success,
            "num_keypoints": self.num_keypoints,
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_error": self.reprojection_error,
            "message": self.message,
            "error": self.error,
        }


@dataclass(frozen=True)
class MappedObject:
    object_info: dict[str, Any]
    global_coordinate: list[float] | None
    localized: bool

    def to_json(self) -> dict[str, Any]:
        payload = dict(self.object_info)
        payload["global_coordinate"] = self.global_coordinate
        payload["localized"] = self.localized
        return payload
