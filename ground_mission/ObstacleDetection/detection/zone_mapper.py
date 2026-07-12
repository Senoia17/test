"""Map real-world object coordinates to competition zones."""

from __future__ import annotations

from dataclasses import dataclass
import math

import config


@dataclass(frozen=True)
class ZoneLayout:
    """Description of a rectangular grid region in world coordinates."""

    name: str
    prefix: str
    origin_x_cm: float
    origin_y_cm: float
    cell_length_cm: float
    cell_width_cm: float
    rows: int
    cols: int

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneLayout":
        """Create a zone layout from config data."""
        origin_x_cm, origin_y_cm = data["origin_cm"]
        cell_length_cm, cell_width_cm = data["cell_cm"]
        return cls(
            name=str(data["name"]),
            prefix=str(data["prefix"]),
            origin_x_cm=float(origin_x_cm),
            origin_y_cm=float(origin_y_cm),
            cell_length_cm=float(cell_length_cm),
            cell_width_cm=float(cell_width_cm),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
        )

    @property
    def max_x_cm(self) -> float:
        """Return the maximum x coordinate covered by this layout."""
        return self.origin_x_cm + self.cell_length_cm * self.cols

    @property
    def max_y_cm(self) -> float:
        """Return the maximum y coordinate covered by this layout."""
        return self.origin_y_cm + self.cell_width_cm * self.rows

    def contains(self, x_cm: float, y_cm: float) -> bool:
        """Return whether a coordinate is inside this layout."""
        return (
            self.origin_x_cm <= x_cm < self.max_x_cm
            and self.origin_y_cm <= y_cm < self.max_y_cm
        )

    def zone_name(self, x_cm: float, y_cm: float) -> str:
        """Return the zone name for a coordinate inside this layout."""
        col = math.floor((x_cm - self.origin_x_cm) / self.cell_length_cm)
        row = math.floor((y_cm - self.origin_y_cm) / self.cell_width_cm)

        if self.rows == 1:
            return f"{self.prefix}-{col + 1:02d}"

        row_label = chr(ord("A") + row)
        return f"{self.prefix}-{row_label}{col + 1}"


class ZoneMapper:
    """Map detections with real-world coordinates to zone labels."""

    def __init__(self, layouts: tuple[dict, ...] = config.ZONE_LAYOUTS) -> None:
        """Initialize the mapper with configured competition zones."""
        self.layouts = [ZoneLayout.from_dict(layout) for layout in layouts]

    def map_zone(self, detection: dict) -> dict:
        """Attach a zone to a detection when real-world coordinates exist.

        Expected coordinate inputs, in priority order:
        - `center_cm`: [x_cm, y_cm]
        - `world_center_cm`: [x_cm, y_cm]
        - `world_bbox_cm`: [x1_cm, y1_cm, x2_cm, y2_cm]

        If homography has not yet produced these fields, the zone remains
        `UNKNOWN` so the current pipeline still runs.
        """
        mapped_detection = detection.copy()
        point_cm = self._extract_center_cm(mapped_detection)

        if point_cm is None:
            mapped_detection.setdefault("zone", config.UNKNOWN_ZONE)
            return mapped_detection

        x_cm, y_cm = point_cm
        mapped_detection["zone"] = self.get_zone(x_cm, y_cm)
        return mapped_detection

    def get_zone(self, x_cm: float, y_cm: float) -> str:
        """Return the configured zone label for a world coordinate."""
        for layout in self.layouts:
            if layout.contains(x_cm, y_cm):
                return layout.zone_name(x_cm, y_cm)
        return config.UNKNOWN_ZONE

    @staticmethod
    def _extract_center_cm(detection: dict) -> tuple[float, float] | None:
        """Extract the center point in centimeters from a detection."""
        point = detection.get("center_cm") or detection.get("world_center_cm")
        if point is not None and len(point) == 2:
            return float(point[0]), float(point[1])

        world_bbox = detection.get("world_bbox_cm")
        if world_bbox is not None and len(world_bbox) == 4:
            x1_cm, y1_cm, x2_cm, y2_cm = [float(value) for value in world_bbox]
            return (x1_cm + x2_cm) / 2.0, (y1_cm + y2_cm) / 2.0

        return None


def map_zone(detection: dict) -> dict:
    """Functional wrapper for zone mapping."""
    return ZoneMapper().map_zone(detection)
