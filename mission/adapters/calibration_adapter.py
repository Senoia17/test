"""Calibration adapter for optional camera undistortion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibration.camera_model import CameraModel
from calibration.undistort import undistort_frame


class CalibrationAdapter:
    """Hide CameraModel loading and undistortion details behind a stable API."""

    def __init__(self, calibration_path: str | Path | None = None) -> None:
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.camera_model: CameraModel | None = None
        self.error: str | None = None
        if self.calibration_path is not None:
            if self.calibration_path.exists():
                try:
                    self.camera_model = CameraModel.load(self.calibration_path)
                except Exception as exc:
                    self.error = f"Calibration load failed: {exc}"
            else:
                self.error = f"Calibration file not found: {self.calibration_path}"

    @property
    def enabled(self) -> bool:
        return self.camera_model is not None

    def undistort(self, frame: Any) -> Any:
        return undistort_frame(frame, self.camera_model)
