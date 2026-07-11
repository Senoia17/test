"""Localization quality checks and status constants."""

from __future__ import annotations

import math
from typing import Any


STATUS_SUCCESS = "SUCCESS"
STATUS_TOO_FEW_MATCHES = "FAILED_TOO_FEW_MATCHES"
STATUS_TOO_FEW_INLIERS = "FAILED_TOO_FEW_INLIERS"
STATUS_LOW_INLIER_RATIO = "FAILED_LOW_INLIER_RATIO"
STATUS_INVALID_HOMOGRAPHY = "FAILED_INVALID_HOMOGRAPHY"


def _as_3x3(homography: Any) -> list[list[float]] | None:
    if hasattr(homography, "tolist"):
        homography = homography.tolist()
    if not isinstance(homography, (list, tuple)) or len(homography) != 3:
        return None
    matrix: list[list[float]] = []
    for row in homography:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        matrix.append([float(value) for value in row])
    return matrix


def _det3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def validate_homography_matrix(homography: Any) -> tuple[bool, str]:
    """Validate that a homography is finite, 3x3, and not singular."""
    matrix = _as_3x3(homography)
    if matrix is None:
        return False, STATUS_INVALID_HOMOGRAPHY
    if not all(math.isfinite(value) for row in matrix for value in row):
        return False, STATUS_INVALID_HOMOGRAPHY
    if abs(_det3(matrix)) < 1e-9:
        return False, STATUS_INVALID_HOMOGRAPHY
    return True, STATUS_SUCCESS


def validate_match_quality(
    *,
    homography: Any,
    num_matches: int,
    num_inliers: int,
    inlier_ratio: float,
    min_matches: int,
    min_inliers: int,
    min_inlier_ratio: float,
) -> tuple[bool, str]:
    """Validate feature-match homography quality against configured thresholds."""
    if num_matches < min_matches:
        return False, STATUS_TOO_FEW_MATCHES
    if num_inliers < min_inliers:
        return False, STATUS_TOO_FEW_INLIERS
    if inlier_ratio < min_inlier_ratio:
        return False, STATUS_LOW_INLIER_RATIO
    return validate_homography_matrix(homography)
