"""Focused tests for ArUco calibration configuration and geometry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.calibrator import (
    MIN_CALIBRATION_OBSERVATIONS,
    ArucoCalibrationConfig,
    load_calibration_config,
)
from calibration.camera_model import CameraModel


def test_load_calibration_config_reuses_mapping_dictionary(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """mapping:
  aruco:
    dictionary: DICT_5X5_50
calibration:
  aruco:
    marker_ids: [4, 3, 1, 2]
    marker_length: 12.5
    sample_interval: 10
""",
        encoding="utf-8",
    )

    config = load_calibration_config(config_path)

    assert config == ArucoCalibrationConfig("DICT_5X5_50", (4, 3, 1, 2), 12.5, 10)


def test_marker_length_must_be_configured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """mapping:
  aruco:
    dictionary: DICT_5X5_50
calibration:
  aruco:
    marker_ids: [4, 3, 1, 2]
    marker_length: null
    sample_interval: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="marker_length"):
        load_calibration_config(config_path)


def test_camera_model_json_schema_is_unchanged(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    model = CameraModel(
        camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.1, -0.2, 0.0, 0.0, 0.01],
        image_size=[1280, 720],
        reprojection_error=0.4,
    )

    model.save(output)

    assert set(json.loads(output.read_text(encoding="utf-8"))) == {
        "camera_matrix",
        "distortion_coefficients",
        "image_size",
        "reprojection_error",
    }
    assert CameraModel.load(output) == model


def test_calibration_requires_thirty_observations() -> None:
    assert MIN_CALIBRATION_OBSERVATIONS == 30
