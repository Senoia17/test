"""Facility mission pipeline entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_facility_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run the existing facility state analyzer for a provided crop/image."""
    if dry_run:
        print("[DRY-RUN] mission=facility adapter=facility.run")
        return {"mission": "facility", "adapter": "facility.run"}

    paths_config = config.get("paths", {})
    if input_path is None:
        configured_input = paths_config.get("facility_video") or paths_config.get("input")
        input_path = Path(configured_input) if configured_input else None
    if input_path is None:
        raise ValueError("Facility mission requires --input or paths.facility_video in config.yaml")
    if not input_path.exists():
        raise FileNotFoundError(f"Facility video not found: {input_path}")

    calibration_path = paths_config.get("calibration")
    if not calibration_path:
        raise ValueError("Facility mission requires paths.calibration in config.yaml")
    if not Path(calibration_path).exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    model_path = config.get("models", {}).get("facility_damage_weights")
    if not model_path:
        raise ValueError("Missing models.facility_damage_weights in config.yaml")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Facility model weights not found: {model_path}")

    from calibration.camera_model import CameraModel
    from facility.state_infer import analyze_facility_video

    result = analyze_facility_video(
        video_path=str(input_path),
        model_path=model_path,
        camera_model=CameraModel.load(calibration_path),
        sample_interval=int(config.get("missions", {}).get("facility", {}).get("sample_interval", 30)),
        fa_id=None,
    )

    if output_path is not None:
        from mission.json_writer import write_json

        write_json(output_path, result)

    return result
