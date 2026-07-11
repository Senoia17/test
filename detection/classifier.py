"""YOLO classification wrapper for facility normal/damaged inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detection.postprocess import normalize_confidence
from detection.yolo_runtime import YoloRuntime


class YoloClassifier:
    """Parse YOLO classification probabilities into a stable dictionary."""

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        runtime: YoloRuntime | None = None,
        device: str | int | None = None,
        imgsz: int = 320,
        conf: float | None = None,
    ) -> None:
        if runtime is None and weights is None:
            raise ValueError("Either weights or runtime must be provided")
        self.runtime = runtime or YoloRuntime(weights, device=device, imgsz=imgsz, conf=conf)  # type: ignore[arg-type]

    def classify(self, image: Any) -> dict[str, object]:
        """Return top-1 label, confidence, and raw class probabilities."""
        results = self.runtime.predict(image)
        if not results:
            raise ValueError("YOLO classifier returned no results")
        result = results[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            raise ValueError("YOLO result does not contain classification probabilities")

        top1_idx = int(probs.top1)
        names = getattr(result, "names", {})
        label = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else str(top1_idx)
        probs_data = probs.data.detach().cpu().numpy() if hasattr(probs.data, "detach") else probs.data
        raw_probs = {name: float(probs_data[int(idx)]) for idx, name in dict(names).items()}
        return {
            "label": label,
            "confidence": normalize_confidence(probs.top1conf),
            "raw_probs": raw_probs,
        }

    __call__ = classify
