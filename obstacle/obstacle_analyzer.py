"""High-level obstacle analysis dispatcher."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from obstacle.crater_analyzer import analyze_crater
from obstacle.uxo_analyzer import UXO_CLASSES, analyze_uxo


ObstacleResult = dict[str, object]
ZoneLookup = Callable[[list[float]], str]


def _attach_zone(result: ObstacleResult, zone_lookup: ZoneLookup | None) -> ObstacleResult:
    if zone_lookup is None:
        return result
    result = dict(result)
    result["zone"] = zone_lookup(result["position"])  # type: ignore[arg-type]
    return result


def analyze_obstacles(
    detections: Iterable[Mapping[str, Any]],
    frame_to_map_H: object,
    *,
    zone_lookup: ZoneLookup | None = None,
) -> dict[str, list[ObstacleResult]]:
    """Analyze normalized YOLO detections with a frame-to-map homography."""
    crater_results: list[ObstacleResult] = []
    uxo_results: list[ObstacleResult] = []

    for detection in detections:
        class_name = detection.get("class")
        if class_name == "crater":
            crater_results.append(_attach_zone(analyze_crater(detection, frame_to_map_H), zone_lookup))
        elif class_name in UXO_CLASSES:
            uxo_results.append(_attach_zone(analyze_uxo(detection, frame_to_map_H), zone_lookup))

    return {
        "craters": crater_results,
        "uxos": uxo_results,
    }
