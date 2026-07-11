"""Obstacle mission adapter.

Phase 1 keeps the existing obstacle implementation intact and delegates to it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_OBSTACLE_DIR = REPO_ROOT / "ground_mission" / "ObstacleDetection"


def run_obstacle_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run the existing runway/taxiway obstacle detection pipeline."""
    if dry_run:
        print("[DRY-RUN] mission=obstacle adapter=ground_mission/ObstacleDetection")
        return {"mission": "obstacle", "adapter": "ground_mission/ObstacleDetection"}

    if str(LEGACY_OBSTACLE_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_OBSTACLE_DIR))

    legacy_main_path = LEGACY_OBSTACLE_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("legacy_obstacle_main", legacy_main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy obstacle pipeline: {legacy_main_path}")
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    legacy_run_pipeline = legacy_module.run_pipeline

    video_path = input_path
    if video_path is None:
        mission_config = config.get("paths", {})
        configured_input = mission_config.get("input")
        video_path = Path(configured_input) if configured_input else None

    if video_path is None:
        legacy_run_pipeline()
    else:
        legacy_run_pipeline(video_path=Path(video_path))

    return None
