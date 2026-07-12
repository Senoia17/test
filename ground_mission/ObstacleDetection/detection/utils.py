"""Shared utilities for logging, paths, drawing, and geometry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import config


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_path: Path) -> None:
    """Configure console and file logging for the application."""
    ensure_directories([log_path.parent])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def xyxy_to_xywh(bbox: list[float]) -> list[float]:
    """Convert a bounding box from xyxy to center xywh format."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return [center_x, center_y, width, height]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert a bounding box from center xywh to xyxy format."""
    center_x, center_y, width, height = bbox
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return [x1, y1, x2, y2]


def draw_detection(frame: np.ndarray, detection: dict) -> None:
    """Draw one tracked detection on a video frame in place."""
    x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
    label = (
        f'#{detection["id"]} {detection["class"]} '
        f'{detection["confidence"]:.2f}'
    )
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        config.ANNOTATION_THICKNESS,
    )
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.ANNOTATION_FONT_SCALE,
        (0, 255, 0),
        config.ANNOTATION_THICKNESS,
        cv2.LINE_AA,
    )
