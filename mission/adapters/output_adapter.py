"""Output adapter for mission JSON results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mission.json_writer import write_json


def resolve_output_path(output: str | Path | None, default_name: str = "ground_mission_results.json") -> Path:
    if output is None:
        output_path = Path("outputs") / default_name
    else:
        output_path = Path(output)
        if output_path.suffix.lower() != ".json":
            output_path = output_path / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_mission_output(payload: Any, output: str | Path | None, default_name: str = "ground_mission_results.json") -> Path:
    output_path = resolve_output_path(output, default_name=default_name)
    write_json(payload, output_path)
    return output_path
