"""Shared device selection for YOLO inference."""

from __future__ import annotations

from typing import Any

import torch


def select_yolo_device() -> str:
    """Return the best available YOLO inference device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def move_yolo_model(model: Any, device: str | None = None) -> str:
    """Move a YOLO model to its inference device and log the selection."""
    selected_device = device or select_yolo_device()
    model.to(selected_device)
    print(f"Using device: {selected_device}")
    return selected_device
