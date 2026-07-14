"""Backward-compatible calibration wrapper.

The Phase 2 calibration implementation lives in the top-level ``calibration``
package. This module preserves the old ``ground_mission/src/calibration.py``
entry point and function names without duplicating the implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = _REPO_ROOT / "calibration"
_ALIAS = "_drone_calibration_package"


def _load_calibration_package():
    package = sys.modules.get(_ALIAS)
    if package is not None:
        return package

    spec = importlib.util.spec_from_file_location(
        _ALIAS,
        _PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(_PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration package from {_PACKAGE_DIR}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = package
    spec.loader.exec_module(package)
    return package


_pkg = _load_calibration_package()

CameraModel = _pkg.CameraModel
Calibration = _pkg.Calibration
calibrate_camera = _pkg.calibrate_camera
estimate_aruco_calibration = _pkg.estimate_aruco_calibration
load_calibration_config = _pkg.load_calibration_config
load_calibration = _pkg.load_calibration
save_calibration = _pkg.save_calibration
undistort_frame = _pkg.undistort_frame
undistort_image = _pkg.undistort_image
def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate drone camera calibration from four ArUco marker videos.")
    parser.add_argument("--videos", type=Path, nargs=4, required=True, metavar=("VIDEO_1", "VIDEO_2", "VIDEO_3", "VIDEO_4"))
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=Path("configs/calibration.json"))
    args = parser.parse_args()

    calibration = estimate_aruco_calibration(args.videos, load_calibration_config(args.config))
    save_calibration(calibration, args.output)
    print(f"Saved calibration to {args.output} with reprojection error {calibration.reprojection_error:.4f}")


if __name__ == "__main__":
    main()
