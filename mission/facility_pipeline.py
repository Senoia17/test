"""Facility inspection mission adapter.

Phase 1 exposes a pipeline-shaped entry point around the existing facility crop
analysis script. Full video/frame orchestration is intentionally deferred.
"""

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
        print("[DRY-RUN] mission=facility adapter=facility_state_infer.py")
        return {"mission": "facility", "adapter": "facility_state_infer.py"}

    if input_path is None:
        raise ValueError("Facility mission currently requires --input pointing to an existing facility crop/image")

    model_path = config.get("models", {}).get("facility_damage_weights")
    if not model_path:
        raise ValueError("Missing models.facility_damage_weights in config.yaml")

    from facility_state_infer import analyze_single_crop

    result = analyze_single_crop(
        image_path=str(input_path),
        model_path=model_path,
        fa_id=None,
        save_debug=False,
    )

    if output_path is not None:
        from mission.json_writer import write_json

        write_json(output_path, result)

    return result
