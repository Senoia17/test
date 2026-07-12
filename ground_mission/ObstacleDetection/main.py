"""Entry point for the obstacle detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import config
from detection.crater_classifier import CraterClassifier
from detection.detector import YoloDetector
from detection.json_builder import JsonBuilder
from detection.tracker import VideoTracker
from detection.utils import configure_logging, ensure_directories
from detection.zone_mapper import ZoneMapper


def run_pipeline(video_path: Path = config.INPUT_VIDEO_PATH) -> None:
    """Run detection, tracking, post-processing, and JSON export."""
    ensure_directories([config.OUTPUT_DIR])
    configure_logging(config.LOG_PATH)

    logger = logging.getLogger(__name__)
    logger.info("Starting obstacle detection pipeline.")

    detector = YoloDetector(
        model_path=config.MODEL_PATH,
        confidence=config.CONFIDENCE,
        image_size=config.IMAGE_SIZE,
        device=config.DEVICE,
    )
    tracker = VideoTracker(
        detector=detector,
        tracker_config_path=config.TRACKER_CONFIG_PATH,
        class_names=config.CLASS_NAMES,
    )

    detections = tracker.track_video(
        video_path=video_path,
        annotated_video_path=config.ANNOTATED_VIDEO_PATH
        if config.SAVE_ANNOTATED_VIDEO
        else None,
    )

    zone_mapper = ZoneMapper()
    crater_classifier = CraterClassifier()

    processed_detections = []
    for detection in detections:
        mapped_detection = zone_mapper.map_zone(detection)
        classified_detection = crater_classifier.classify_size(mapped_detection)
        processed_detections.append(classified_detection)

    JsonBuilder().write_json_files(
        detections=processed_detections,
        crater_json_path=config.CRATER_JSON_PATH,
        uxo_json_path=config.UXO_JSON_PATH,
    )

    logger.info("Pipeline completed. Total objects: %s", len(processed_detections))


def main() -> None:
    """CLI wrapper for the pipeline."""
    try:
        run_pipeline()
    except Exception:
        logging.getLogger(__name__).exception("Pipeline failed.")
        raise


if __name__ == "__main__":
    main()
