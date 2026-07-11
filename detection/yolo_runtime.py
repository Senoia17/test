"""Common Ultralytics YOLO runtime wrapper.

This module centralizes model loading and prediction configuration only. It does
not parse boxes, interpret classes/probabilities, make mission decisions, track
objects, or write JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class YoloRuntimeConfig:
    """Shared YOLO runtime configuration."""

    weights_path: str
    imgsz: int = 640
    conf: Optional[float] = None
    device: Optional[Any] = None
    verbose: bool = False


class YoloRuntime:
    """Thin wrapper around ``ultralytics.YOLO`` for shared prediction setup."""

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 640,
        conf: Optional[float] = None,
        device: Optional[Any] = None,
        verbose: bool = False,
        model: Optional[Any] = None,
    ) -> None:
        self.config = YoloRuntimeConfig(
            weights_path=str(weights_path),
            imgsz=int(imgsz),
            conf=conf,
            device=device,
            verbose=verbose,
        )
        self._model = model

    @property
    def model(self) -> Any:
        """Load and return the underlying Ultralytics model lazily."""

        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.weights_path)
        return self._model

    def predict(self, source: Any, **kwargs: Any) -> Any:
        """Run common YOLO prediction with overridable runtime settings."""

        predict_kwargs: dict[str, Any] = {
            "source": source,
            "imgsz": kwargs.pop("imgsz", self.config.imgsz),
            "verbose": kwargs.pop("verbose", self.config.verbose),
        }
        conf = kwargs.pop("conf", self.config.conf)
        if conf is not None:
            predict_kwargs["conf"] = conf
        device = kwargs.pop("device", self.config.device)
        if device is not None:
            predict_kwargs["device"] = device
        predict_kwargs.update(kwargs)
        return self.model.predict(**predict_kwargs)
