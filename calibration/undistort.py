"""Lens distortion correction helpers."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from calibration.camera_model import CameraModel


def undistort_frame(image: Any, camera_model: "CameraModel | None", crop: bool = False) -> Any:
    """Remove lens distortion while preserving frame size by default."""
    if camera_model is None:
        return image

    import cv2

    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_model.matrix(), camera_model.dist_coeffs(), (width, height), 1, (width, height))
    undistorted = cv2.undistort(image, camera_model.matrix(), camera_model.dist_coeffs(), None, new_matrix)
    if crop:
        x, y, w, h = roi
        if w > 0 and h > 0:
            return undistorted[y : y + h, x : x + w]
    return undistorted


# Backward-compatible alias for the previous helper function name.
undistort_image = undistort_frame
