"""YOLO11 training module."""

from __future__ import annotations

import logging

from ultralytics import YOLO

import config
from detection.utils import configure_logging, ensure_directories


def train() -> None:
    """Train a YOLO11 model using settings from config.py."""
    ensure_directories([config.OUTPUT_DIR, config.WEIGHTS_DIR, config.TRAIN_PROJECT])
    configure_logging(config.LOG_PATH)
    logger = logging.getLogger(__name__)

    if not config.DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config.DATA_YAML}")

    model_path = config.PRETRAINED_MODEL_PATH
    logger.info("Starting YOLO training with model: %s", model_path)

    model = YOLO(str(model_path))
    model.train(
        data=str(config.DATA_YAML),
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.TRAIN_BATCH_SIZE,
        device=config.DEVICE,
        project=str(config.TRAIN_PROJECT),
        name=config.TRAIN_NAME,
        workers=config.TRAIN_WORKERS,
        exist_ok=True,
    )
    logger.info("Training completed.")


if __name__ == "__main__":
    train()
