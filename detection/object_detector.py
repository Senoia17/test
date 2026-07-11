"""YOLO object detection wrapper for the obstacle mission."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .postprocess import normalize_bbox_xyxy
from .yolo_runtime import YoloRuntime

OBSTACLE_CLASS_NAMES: dict[int, str] = {
    0: "crater",
    1: "missile",
    2: "cluster",
    3: "dumb",
}


class ObjectDetector:
    """Parse YOLO detection output into mission-neutral object detections."""

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 1280,
        conf: float = 0.3,
        device: Optional[Any] = None,
        class_names: Optional[Mapping[int, str]] = None,
        runtime: Optional[YoloRuntime] = None,
    ) -> None:
        self.runtime = runtime or YoloRuntime(
            weights_path=weights_path,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        self.class_names = dict(class_names or OBSTACLE_CLASS_NAMES)

    def detect(self, frame: Any) -> list[dict[str, object]]:
        """Detect obstacle objects in one frame and return normalized dictionaries."""

        results = self.runtime.predict(frame)
        detections: list[dict[str, object]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.class_names.get(class_id, str(class_id))
                bbox = box.xyxy[0].detach().cpu().numpy().tolist()
                confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
                detections.append(
                    {
                        "class": class_name,
                        "class_id": class_id,
                        "bbox": normalize_bbox_xyxy(bbox),
                        "confidence": confidence,
                    }
                )
        return detections
