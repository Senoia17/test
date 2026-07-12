"""Top-level mission router for the offline drone AI mission system.

This module intentionally contains no computer-vision logic. It only parses
arguments, loads configuration, selects a mission, and delegates execution to a
mission pipeline adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


from facility.run import run_facility_pipeline
from mission.map_pipeline import run_map_pipeline
from mission.obstacle_pipeline import run_obstacle_pipeline
from mission.pipeline import run_ground_pipeline


MISSION_CHOICES = ("ground", "facility", "mapping", "obstacle")


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    return value.strip('"\'')


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """Parse the simple two-level config.yaml used by Phase 1.

    PyYAML is used when installed; this fallback keeps the mission router
    importable in minimal environments without introducing CV dependencies.
    """
    data: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue
        if current_section and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current_section][key.strip()] = _parse_scalar(value.strip())
    return data


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration for mission adapters."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_yaml_fallback(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone AI mission router")
    parser.add_argument("--mission", choices=MISSION_CHOICES, default="ground")
    parser.add_argument("--input", type=Path, help="Mission input video/image/crop path")
    parser.add_argument("--output", type=Path, help="Mission output path or directory")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate routing without invoking CV dependencies")
    parser.add_argument("--debug", action="store_true", help="Save debug frames for the integrated ground pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.mission == "ground":
        run_ground_pipeline(args.input, args.output, config, debug=args.debug, dry_run=args.dry_run)
    elif args.mission == "facility":
        run_facility_pipeline(args.input, args.output, config, dry_run=args.dry_run)
    elif args.mission == "mapping":
        run_map_pipeline(args.input, args.output, config, dry_run=args.dry_run)
    elif args.mission == "obstacle":
        run_obstacle_pipeline(args.input, args.output, config, dry_run=args.dry_run)
    else:  # argparse choices prevent this branch.
        raise ValueError(f"Unsupported mission: {args.mission}")


if __name__ == "__main__":
    main()
