import sys
from pathlib import Path

from ultralytics import YOLO


def _resolve_pretrained_weight() -> str | Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from main import load_config
    from mission.model_weights import resolve_pretrained_model_path

    pretrained = resolve_pretrained_model_path(load_config(project_root / "config.yaml"), "obstacle")
    if pretrained is None:
        raise ValueError("Missing models.obstacle.pretrained in config.yaml")
    return pretrained


def main():
    model = YOLO(_resolve_pretrained_weight())

    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=1280,
        batch=8,
        workers=4,
        device=0,
        patience=30,
        project="runs",
        name="ground_detector",
        exist_ok=True,
        degrees=5,
        translate=0.08,
        scale=0.25,
        perspective=0.0008,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
    )


if __name__ == "__main__":
    main()
