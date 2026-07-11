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
    error: str | None = None

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
