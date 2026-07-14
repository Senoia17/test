"""Shared Ultralytics YOLO execution runtime.

This module owns only model loading and prediction execution. It intentionally
avoids parsing boxes, classification probabilities, mission decisions, tracking,
or output formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.yolo_device import move_yolo_model, select_yolo_device


class YoloRuntime:
    """Thin wrapper around ``ultralytics.YOLO.predict``."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | int | None = None,
        imgsz: int = 640,
        conf: float | None = None,
        verbose: bool = False,
        model: Any | None = None,
    ) -> None:
        self.weights = Path(weights) if not isinstance(weights, Path) else weights
        self.device = self._default_device() if device is None else device
        self.imgsz = imgsz
        self.conf = conf
        self.verbose = verbose
        self.model = model if model is not None else self._load_model(self.weights)
        if model is None:
            self.device = move_yolo_model(self.model, str(self.device))

    @staticmethod
    def _default_device() -> str | int:
        return select_yolo_device()

    @staticmethod
    def _load_model(weights: Path) -> Any:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is required to load YOLO models. Install project AI dependencies before running inference."
            ) from exc
        return YOLO(str(weights))

    def predict(self, image: Any, **kwargs: Any) -> list[Any]:
        """Execute YOLO prediction and return raw Ultralytics results."""
        predict_kwargs: dict[str, Any] = {
            "source": image,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": self.verbose,
        }
        if self.conf is not None:
            predict_kwargs["conf"] = self.conf
        predict_kwargs.update(kwargs)
        return self.model.predict(**predict_kwargs)
