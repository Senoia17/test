"""Camera calibration parameter model and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraModel:
    """OpenCV camera calibration parameters."""

    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    image_size: list[int]
    reprojection_error: float | None = None

    def matrix(self):
        import numpy as np

        return np.array(self.camera_matrix, dtype=np.float64)

    def dist_coeffs(self):
        import numpy as np

        return np.array(self.distortion_coefficients, dtype=np.float64)

    @classmethod
    def load(cls, path: Path | str) -> "CameraModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(
                camera_matrix=data["camera_matrix"],
                distortion_coefficients=data["distortion_coefficients"],
                image_size=data["image_size"],
                reprojection_error=data.get("reprojection_error"),
            )
        except KeyError as exc:
            raise ValueError(f"Calibration JSON is missing key: {exc}") from exc

    def save(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# Backward-compatible alias for the name used by the previous helper module.
Calibration = CameraModel


def load_calibration(path: Path | str) -> CameraModel:
    return CameraModel.load(path)


def save_calibration(calibration: CameraModel, path: Path | str) -> None:
    calibration.save(path)
