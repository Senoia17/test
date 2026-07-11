"""Mission-independent detection result helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def xyxy_to_list(value: object) -> list[float]:
    """Normalize a YOLO-style xyxy value to ``[x1, y1, x2, y2]`` floats."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "cpu"):
        value = value.cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        value = value[0]

    bbox = [float(v) for v in value]  # type: ignore[arg-type]
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bbox coordinates, received {len(bbox)}")
    return bbox


def normalize_confidence(value: object) -> float:
    """Convert tensor/scalar confidence values to a Python float."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    return float(value)


def filter_by_confidence(
    detections: Iterable[Mapping[str, object]],
    min_confidence: float | None,
) -> list[dict[str, object]]:
    """Return detections whose ``confidence`` is at least ``min_confidence``."""
    normalized = [dict(detection) for detection in detections]
    if min_confidence is None:
        return normalized
    return [d for d in normalized if float(d.get("confidence", 0.0)) >= min_confidence]
