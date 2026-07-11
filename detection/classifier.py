"""YOLO classification wrapper for the facility mission."""

from __future__ import annotations

from typing import Any, Optional

from .yolo_runtime import YoloRuntime


class YoloClassifier:
    """Parse YOLO classification probabilities into a generic result."""

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 320,
        device: Optional[Any] = None,
        runtime: Optional[YoloRuntime] = None,
    ) -> None:
        self.runtime = runtime or YoloRuntime(
            weights_path=weights_path,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

    def classify(self, frame: Any) -> dict[str, object]:
        """Classify one image/frame and return label, confidence, and raw probabilities."""

        results = self.runtime.predict(frame)
        result = results[0]
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        names = result.names
        raw_values = probs.data.detach().cpu().numpy()
        raw_probs = {name: float(raw_values[int(idx)]) for idx, name in names.items()}
        return {
            "label": names[top1_idx],
            "confidence": top1_conf,
            "raw_probs": raw_probs,
        }
