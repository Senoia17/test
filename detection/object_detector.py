"""YOLO object detection wrapper for the obstacle mission."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from detection.postprocess import filter_by_confidence, normalize_confidence, xyxy_to_list
from detection.yolo_runtime import YoloRuntime


STANDARD_OBSTACLE_CLASSES: dict[int, str] = {
    0: "crater",
    1: "missile",
    2: "cluster",
    3: "dumb",
}

LEGACY_OBSTACLE_CLASS_ORDER_A: dict[int, str] = {
    0: "crater",
    1: "cluster",
    2: "dumb",
    3: "missile",
}


class ObjectDetector:
    """Normalize YOLO detection boxes without mission-specific analysis."""

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        runtime: YoloRuntime | None = None,
        class_names: Mapping[int, str] | None = None,
        device: str | int | None = None,
        imgsz: int = 640,
        conf: float | None = None,
    ) -> None:
        if runtime is None and weights is None:
            raise ValueError("Either weights or runtime must be provided")
        self.runtime = runtime or YoloRuntime(weights, device=device, imgsz=imgsz, conf=conf)  # type: ignore[arg-type]
        self.class_names = dict(class_names or STANDARD_OBSTACLE_CLASSES)
        self.min_confidence = conf

    def detect(self, image: Any) -> list[dict[str, object]]:
        """Return normalized obstacle detections preserving YOLO class identity."""
        results = self.runtime.predict(image)
        detections: list[dict[str, object]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(normalize_confidence(box.cls))
                detections.append(
                    {
                        "class": self.class_names.get(class_id, str(class_id)),
                        "class_id": class_id,
                        "bbox": xyxy_to_list(box.xyxy),
                        "confidence": normalize_confidence(box.conf),
                    }
                )
        return filter_by_confidence(detections, self.min_confidence)  # type: ignore[return-value]

    __call__ = detect
