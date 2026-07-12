import sys
from pathlib import Path

from ultralytics import YOLO


def _resolve_pretrained_weight() -> str | Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from main import load_config

    config = load_config(project_root / "config.yaml")
    obstacle_model = config.get("models", {}).get("obstacle", {})
    directory = obstacle_model.get("directory") if isinstance(obstacle_model, dict) else None
    pretrained = obstacle_model.get("pretrained") if isinstance(obstacle_model, dict) else None
    if not pretrained:
        raise ValueError("Missing models.obstacle.pretrained in config.yaml")

    candidate = Path(directory) / pretrained if directory else Path(pretrained)
    return candidate if candidate.exists() else str(pretrained)


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
