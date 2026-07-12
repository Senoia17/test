"""Build final crater and UXO JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import config
from detection.utils import ensure_directories


class JsonBuilder:
    """Create final JSON outputs expected by the competition server."""

    def __init__(self) -> None:
        """Initialize a JSON builder."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self, detections: list[dict]) -> tuple[dict, dict]:
        """Build crater and UXO JSON payloads from final detections."""
        crater_items: list[dict] = []
        uxo_items: list[dict] = []

        for detection in detections:
            class_name = detection.get("class")
            zone = detection.get("zone", config.UNKNOWN_ZONE)

            if class_name == config.CRATER_CLASS_NAME:
                crater_items.append(
                    {
                        "zone": zone,
                        "size": detection.get("size", config.UNKNOWN_SIZE),
                    }
                )
            elif class_name in config.UXO_CLASS_NAMES:
                uxo_items.append(
                    {
                        "zone": zone,
                        "type": class_name,
                    }
                )

        return {"crater_detect": crater_items}, {"UXO_detect": uxo_items}

    def write_json_files(
        self,
        detections: list[dict],
        crater_json_path: Path,
        uxo_json_path: Path,
    ) -> None:
        """Write crater and UXO JSON payloads to disk."""
        ensure_directories([crater_json_path.parent, uxo_json_path.parent])
        crater_payload, uxo_payload = self.build(detections)

        self._write_json(crater_json_path, crater_payload)
        self._write_json(uxo_json_path, uxo_payload)

        self.logger.info("Wrote crater JSON: %s", crater_json_path)
        self.logger.info("Wrote UXO JSON: %s", uxo_json_path)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        """Write a dictionary as pretty UTF-8 JSON."""
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
