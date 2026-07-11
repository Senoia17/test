"""Global map generation mission adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mapping.map_builder import build_global_map


def run_map_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run video-based global map generation."""
    if dry_run:
        print("[DRY-RUN] mission=mapping adapter=mapping.map_builder.build_global_map")
        return {"mission": "mapping", "adapter": "mapping.map_builder.build_global_map"}

    paths_config = config.get("paths", {})
    if input_path is None:
        configured_input = paths_config.get("mapping_video") or paths_config.get("input") or paths_config.get("map_image")
        input_path = Path(configured_input) if configured_input else None
    if input_path is None:
        raise ValueError("Mapping mission requires --input or paths.mapping_video in config.yaml")

    if output_path is None:
        configured_output = paths_config.get("map_output_dir") or Path(paths_config.get("global_map", "data/map/global_map.jpg")).parent
        output_path = Path(configured_output)

    calibration_path = paths_config.get("calibration")
    result = build_global_map(
        input_path,
        output_dir=output_path,
        calibration_path=calibration_path if calibration_path else None,
    )
    print(f"Saved global map artifacts to {output_path}")
    return result
