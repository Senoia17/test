"""Detection adapter for normalized YOLO object detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detection.object_detector import ObjectDetector
from geometry.transform import bbox_center
from mission.types import DetectionResult


class DetectionAdapter:
    """Wrap ObjectDetector with stable DetectionResult contracts."""

    def __init__(self, weights_path: str | Path | None, *, conf: float = 0.3, imgsz: int = 1280) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.detector: ObjectDetector | None = None
        self.error: str | None = None
        if self.weights_path is None:
            self.error = "YOLO weights path is not configured"
        elif not self.weights_path.exists():
            self.error = f"YOLO weights not found: {self.weights_path}"
        else:
            try:
                self.detector = ObjectDetector(self.weights_path, conf=conf, imgsz=imgsz)
            except Exception as exc:
                self.error = f"YOLO model unavailable: {exc}"
                self.detector = None

    @property
    def available(self) -> bool:
        return self.detector is not None

    def detect(self, frame: Any) -> list[DetectionResult]:
        if self.detector is None:
            return []
        detections = []
        for item in self.detector.detect(frame):
            bbox = [float(value) for value in item["bbox"]]  # type: ignore[index]
            detections.append(
                DetectionResult(
                    class_name=str(item.get("class")),
                    confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                    bbox=bbox,
                    center_pixel=bbox_center(bbox),
                    class_id=int(item["class_id"]) if item.get("class_id") is not None else None,
                )
            )
        return detections
