"""Frame-level YOLO detector."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """Run YOLO inference on individual frames."""

    def __init__(
        self,
        model_path: Path,
        confidence: float,
        image_size: int,
        device: str,
    ) -> None:
        """Initialize a YOLO detector."""
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.model = YOLO(str(model_path))

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """Detect objects in one frame without tracking."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        names = results[0].names
        for box in boxes:
            class_id = int(box.cls.item())
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            detections.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "bbox": [
                        float(value)
                        for value in box.xyxy[0].detach().cpu().tolist()
                    ],
                    "confidence": float(box.conf.item()),
                }
            )

        return detections
