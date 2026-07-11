"""Global map coordinate-system metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from mapping.map_homography import FIELD_CORNERS_CM


@dataclass(frozen=True)
class MapCoordinateSystem:
    """Serializable global map coordinate metadata."""

    resolution: str = "1 pixel per centimeter"
    origin: list[float] | None = None
    scale: float = 1.0
    markers: dict[str, list[float]] | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["origin"] = self.origin or [0.0, 0.0]
        data["markers"] = self.markers or {str(k): v for k, v in FIELD_CORNERS_CM.items()}
        return data

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MapCoordinateSystem":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            resolution=data.get("resolution", "1 pixel per centimeter"),
            origin=data.get("origin", [0.0, 0.0]),
            scale=float(data.get("scale", 1.0)),
            markers=data.get("markers"),
        )


def default_coordinate_system(marker_positions: Mapping[int, Sequence[float]] | None = None) -> MapCoordinateSystem:
    markers = marker_positions or FIELD_CORNERS_CM
    return MapCoordinateSystem(markers={str(k): [float(v[0]), float(v[1])] for k, v in markers.items()})


def save_map_info(path: str | Path, coordinate_system: MapCoordinateSystem | None = None) -> None:
    (coordinate_system or default_coordinate_system()).save(path)
