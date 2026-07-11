"""Mission-level JSON writer facade for Phase 1 adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path | str, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
