"""Map localized world coordinates to named mission zones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

UNKNOWN_ZONE = "UNKNOWN"


@dataclass(frozen=True)
class ZoneRegion:
    """Named polygonal map region in world centimeters."""

    name: str
    polygon_cm: tuple[tuple[float, float], ...]

    def contains(self, point_cm: Sequence[float]) -> bool:
        """Return whether ``point_cm`` is inside or on this polygon."""
        x = float(point_cm[0])
        y = float(point_cm[1])
        points = self.polygon_cm
        if len(points) < 3:
            return False

        inside = False
        previous_x, previous_y = points[-1]
        for current_x, current_y in points:
            if _point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
                return True
            crosses = (current_y > y) != (previous_y > y)
            if crosses:
                x_intersection = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x <= x_intersection:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside


class ZoneRegionLookup:
    """Resolve arbitrary world coordinates to configured named zones."""

    def __init__(self, regions: Iterable[ZoneRegion]) -> None:
        self.regions = list(regions)

    def lookup(self, point_cm: Sequence[float]) -> str:
        """Return the containing region name, or ``UNKNOWN`` when outside all zones."""
        for region in self.regions:
            if region.contains(point_cm):
                return region.name
        return UNKNOWN_ZONE


class _SimpleZonesParser:
    """Small parser for the existing ``zones: name: polygon_cm`` YAML shape."""

    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()

    def parse(self) -> dict[str, Any]:
        zones: dict[str, dict[str, Any]] = {}
        in_zones = False
        current_name: str | None = None
        for raw_line in self.lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "zones:":
                in_zones = True
                continue
            if not in_zones:
                continue
            if raw_line.startswith("  ") and not raw_line.startswith("    ") and stripped.endswith(":"):
                current_name = stripped[:-1]
                zones[current_name] = {}
                continue
            if current_name is not None and stripped.startswith("polygon_cm:"):
                value = stripped.split(":", 1)[1].strip()
                zones[current_name]["polygon_cm"] = _parse_inline_points(value)
        return {"zones": zones}


def _point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance


def _parse_inline_points(value: str) -> list[list[float]]:
    import ast

    parsed = ast.literal_eval(value)
    return [[float(point[0]), float(point[1])] for point in parsed]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        return _SimpleZonesParser(text).parse()

    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Zone config must be a mapping: {path}")
    return data


def load_zone_regions(path: str | Path) -> ZoneRegionLookup:
    """Load named polygon regions from the configured zones YAML file."""
    zones_path = Path(path)
    data = _load_yaml(zones_path)
    zones = data.get("zones")
    if not isinstance(zones, Mapping):
        raise ValueError(f"Zone config missing 'zones' mapping: {zones_path}")

    regions: list[ZoneRegion] = []
    for name, info in zones.items():
        if not isinstance(info, Mapping):
            continue
        polygon = info.get("polygon_cm")
        if not isinstance(polygon, Sequence):
            continue
        points = tuple((float(point[0]), float(point[1])) for point in polygon)  # type: ignore[index]
        regions.append(ZoneRegion(str(name), points))
    return ZoneRegionLookup(regions)
