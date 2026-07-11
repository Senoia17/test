"""UXO-specific obstacle analysis."""

from __future__ import annotations

from typing import Mapping

from geometry.transform import apply_homography, bbox_center


UXO_CLASSES = {"missile", "cluster", "dumb"}


def analyze_uxo(detection: Mapping[str, object], frame_to_map_H: object) -> dict[str, object]:
    """Analyze a UXO detection while preserving the original YOLO class."""
    class_name = str(detection.get("class"))
    if class_name not in UXO_CLASSES:
        raise ValueError(f"Expected UXO detection, received: {class_name}")

    bbox = detection["bbox"]  # type: ignore[index]
    center_px = bbox_center(bbox)  # type: ignore[arg-type]
    center_map = apply_homography([center_px], frame_to_map_H)[0]

    return {
        "group": "UXO",
        "type": class_name,
        "position": [float(center_map[0]), float(center_map[1])],
        "confidence": detection.get("confidence"),
        "bbox": list(bbox),  # type: ignore[arg-type]
    }
