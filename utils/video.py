"""Shared video IO utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


def open_video(video_path: str | Path) -> Any:
    """Open a video file with OpenCV and validate it is readable."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return capture


def iter_video_frames(video_path: str | Path, *, sample_interval: int = 1) -> Iterator[tuple[int, Any]]:
    """Yield ``(frame_index, frame)`` pairs from a video."""
    capture = open_video(video_path)
    frame_index = 0
    sample_interval = max(1, int(sample_interval))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_interval == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def sample_video_frames(video_path: str | Path, *, sample_interval: int = 30) -> list[tuple[int, Any]]:
    """Return sampled video frames as a list."""
    return list(iter_video_frames(video_path, sample_interval=sample_interval))
