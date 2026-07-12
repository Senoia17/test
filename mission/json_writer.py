"""Mission-level flexible JSON writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: Any, output_path: Path | str | None = None) -> None:
    """Write mission data to JSON.

    The preferred Phase 8 call style is ``write_json(data, output_path)``. For
    backward compatibility with Phase 1 adapters, ``write_json(output_path,
    data)`` is also accepted when the first argument is path-like.
    """
    if output_path is None:
        raise ValueError("write_json requires an output path")

    if isinstance(data, (str, Path)) and not isinstance(output_path, (str, Path)):
        data, output_path = output_path, data

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
