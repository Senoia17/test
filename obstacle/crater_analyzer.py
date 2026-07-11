"""Crater-specific physical-size analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

from geometry.measurement import polygon_size_mm
from geometry.transform import apply_homography, bbox_center, bbox_corners


CRATER_SIZE_MM: dict[str, tuple[float, float]] = {
    "small": (102.5, 99.0),
    "medium": (159.0, 150.0),
    "large": (179.0, 200.0),
}


def classify_crater_size(width_mm: float, height_mm: float) -> str:
    """Classify crater size by closest physical top-view area."""
    area = width_mm * height_mm
    distances = {}
    for name, (ref_w, ref_h) in CRATER_SIZE_MM.items():
        distances[name] = abs(area - (ref_w * ref_h))
    return min(distances, key=distances.get)


def analyze_crater(detection: Mapping[str, object], frame_to_map_H: object) -> dict[str, object]:
    """Measure and classify a single YOLO crater detection."""
    if detection.get("class") != "crater":
        raise ValueError(f"Expected crater detection, received: {detection.get('class')}")

    bbox = detection["bbox"]  # type: ignore[index]
    corners_px = bbox_corners(bbox)  # type: ignore[arg-type]
    corners_map = apply_homography(corners_px, frame_to_map_H)
    width_mm, height_mm = polygon_size_mm(corners_map)

    center_px = bbox_center(bbox)  # type: ignore[arg-type]
    center_map = apply_homography([center_px], frame_to_map_H)[0]

    return {
        "type": "crater",
        "size": classify_crater_size(width_mm, height_mm),
        "width": float(width_mm),
        "height": float(height_mm),
        "position": [float(center_map[0]), float(center_map[1])],
        "confidence": detection.get("confidence"),
        "bbox": list(bbox),  # type: ignore[arg-type]
    }
