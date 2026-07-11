"""Global map generation mission adapter.

Phase 1 delegates to the existing ArUco homography helper without changing its
math or marker assumptions.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_SRC_DIR = REPO_ROOT / "ground_mission" / "src"


def run_map_pipeline(
    input_path: Path | None,
    output_path: Path | None,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run the existing ArUco homography generation path."""
    if dry_run:
        print("[DRY-RUN] mission=mapping adapter=ground_mission/src/aruco_homography.py")
        return {"mission": "mapping", "adapter": "ground_mission/src/aruco_homography.py"}

    if input_path is None:
        configured_input = config.get("paths", {}).get("map_image")
        input_path = Path(configured_input) if configured_input else None
    if input_path is None:
        raise ValueError("Mapping mission requires --input or paths.map_image in config.yaml")

    if output_path is None:
        configured_output = config.get("paths", {}).get("homography")
        output_path = Path(configured_output) if configured_output else Path("configs/homography.pkl")

    if str(GROUND_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(GROUND_SRC_DIR))

    import cv2  # Imported only inside the mapping pipeline adapter.
    from aruco_homography import compute_homography, detect_aruco_centers  # type: ignore[import-not-found]

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(input_path)

    centers = detect_aruco_centers(image)
    homography = compute_homography(centers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(homography, file)

    print(f"Saved homography to {output_path}")
    return None
