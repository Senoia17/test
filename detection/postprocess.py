"""Generic detection post-processing helpers.

Keep this module mission-agnostic: no crater sizing, UXO grouping, facility
state decisions, zone mapping, or JSON writing belongs here.
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence


def normalize_bbox_xyxy(bbox: Sequence[float]) -> list[float]:
    """Return an ``[x1, y1, x2, y2]`` bbox as plain floats."""

    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(bbox)}")
    return [float(value) for value in bbox]


def passes_confidence(item: Mapping[str, object], min_confidence: float) -> bool:
    """Return whether an item has confidence greater than or equal to a threshold."""

    return float(item.get("confidence", 0.0)) >= float(min_confidence)


def filter_by_confidence(
    items: Iterable[MutableMapping[str, object]],
    min_confidence: float,
) -> list[MutableMapping[str, object]]:
    """Filter generic prediction dictionaries by their ``confidence`` field."""

    return [item for item in items if passes_confidence(item, min_confidence)]
