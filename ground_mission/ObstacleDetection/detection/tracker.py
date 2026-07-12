"""Video tracking module using detector output and Ultralytics ByteTrack."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

import config
from detection.detector import YoloDetector
from detection.utils import draw_detection, ensure_directories, xyxy_to_xywh


class ByteTrackDetectionAdapter:
    """Adapter exposing detector results in the shape ByteTrack expects."""

    def __init__(self, detections: list[dict], class_id: int) -> None:
        """Build tensor fields consumed by Ultralytics BYTETracker.update()."""
        self._detections = detections
        self.conf = np.asarray(
            [item["confidence"] for item in detections],
            dtype=np.float32,
        )
        self.xywh = np.asarray(
            [xyxy_to_xywh(item["bbox"]) for item in detections],
            dtype=np.float32,
        ).reshape((-1, 4))
        self.cls = np.full(
            shape=(len(detections),),
            fill_value=class_id,
            dtype=np.float32,
        )

    def __len__(self) -> int:
        """Return number of detections."""
        return len(self._detections)


class VideoTracker:
    """Track objects across a video and keep one best detection per track."""

    def __init__(
        self,
        detector: YoloDetector,
        tracker_config_path: Path,
        class_names: dict[int, str],
    ) -> None:
        """Initialize per-class ByteTrack trackers."""
        self.detector = detector
        self.class_names = class_names
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tracker_args = self._load_tracker_args(tracker_config_path)
        self.trackers = {
            class_id: BYTETracker(args=self.tracker_args, frame_rate=30)
            for class_id in class_names
        }
        self.global_track_ids: dict[tuple[int, int], int] = {}
        self.next_global_track_id = 1

    def track_video(
        self,
        video_path: Path,
        annotated_video_path: Path | None = None,
    ) -> list[dict]:
        """Track all objects in a video and return one result per object."""
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        writer = self._create_video_writer(capture, annotated_video_path)
        best_by_track: dict[int, dict] = {}
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % config.VIDEO_FRAME_STRIDE != 0:
                    continue

                detections = self.detector.detect_frame(frame)
                active_tracks = self._update_trackers(detections, frame)

                for tracked_detection in active_tracks:
                    track_id = tracked_detection["id"]
                    current_best = best_by_track.get(track_id)
                    if (
                        current_best is None
                        or tracked_detection["confidence"]
                        > current_best["confidence"]
                    ):
                        best_by_track[track_id] = tracked_detection

                if writer is not None:
                    for tracked_detection in active_tracks:
                        draw_detection(frame, tracked_detection)
                    writer.write(frame)

            self.logger.info("Processed %s frames.", frame_index)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        return sorted(best_by_track.values(), key=lambda item: item["id"])

    def _update_trackers(
        self,
        detections: list[dict],
        frame: np.ndarray,
    ) -> list[dict]:
        """Update each class tracker with detections from the current frame."""
        tracked_detections: list[dict] = []

        for class_id, class_name in self.class_names.items():
            class_detections = [
                item for item in detections if item.get("class_id") == class_id
            ]
            adapter = ByteTrackDetectionAdapter(class_detections, class_id)
            tracks = self.trackers[class_id].update(adapter, frame)

            for track in tracks:
                parsed = self._parse_track(track, class_name, class_id)
                if parsed is not None:
                    tracked_detections.append(parsed)

        return tracked_detections

    def _parse_track(
        self,
        track: Any,
        class_name: str,
        class_id: int,
    ) -> dict | None:
        """Convert a ByteTrack output row/object into a detection dictionary."""
        values = self._track_to_array(track)
        if values.size < config.MIN_TRACK_OUTPUT_LENGTH:
            self.logger.debug("Skipping malformed track output: %s", values)
            return None

        x1, y1, x2, y2 = [float(value) for value in values[:4]]
        raw_track_id = int(values[4])
        confidence = float(values[5]) if values.size > 5 else 0.0
        track_id = self._get_global_track_id(class_id, raw_track_id)

        return {
            "id": track_id,
            "class": class_name,
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
        }

    def _get_global_track_id(self, class_id: int, raw_track_id: int) -> int:
        """Return a stable compact global ID for a per-class tracker ID."""
        key = (class_id, raw_track_id)
        if key not in self.global_track_ids:
            self.global_track_ids[key] = self.next_global_track_id
            self.next_global_track_id += 1
        return self.global_track_ids[key]

    @staticmethod
    def _track_to_array(track: Any) -> np.ndarray:
        """Normalize ByteTrack track output to a one-dimensional array."""
        if hasattr(track, "result"):
            return np.asarray(track.result, dtype=float).reshape(-1)
        return np.asarray(track, dtype=float).reshape(-1)

    @staticmethod
    def _load_tracker_args(tracker_config_path: Path) -> Namespace:
        """Load ByteTrack YAML settings as an argparse Namespace."""
        if not tracker_config_path.exists():
            raise FileNotFoundError(
                f"Tracker config not found: {tracker_config_path}"
            )

        with tracker_config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return Namespace(**data)

    @staticmethod
    def _create_video_writer(
        capture: cv2.VideoCapture,
        output_path: Path | None,
    ) -> cv2.VideoWriter | None:
        """Create an annotated-video writer when requested."""
        if output_path is None:
            return None

        ensure_directories([output_path.parent])
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or config.DEFAULT_VIDEO_FPS
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

        return writer
