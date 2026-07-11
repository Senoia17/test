"""Reusable ArUco marker detection for mapping and localization."""

from __future__ import annotations

from typing import Any


ARUCO_DICT_NAME = "DICT_4X4_50"


def _create_detector(dictionary_name: str = ARUCO_DICT_NAME) -> Any:
    import cv2

    aruco = cv2.aruco
    dictionary_id = getattr(aruco, dictionary_name)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_aruco_markers(image: Any, dictionary_name: str = ARUCO_DICT_NAME) -> list[dict[str, object]]:
    """Detect ArUco markers and return IDs with four image-space corners."""
    import cv2

    detector = _create_detector(dictionary_name)
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    if ids is None:
        return []

    markers: list[dict[str, object]] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]
        markers.append(
            {
                "id": int(marker_id),
                "corners": [[float(x), float(y)] for x, y in pts],
            }
        )
    return markers


def marker_centers(markers: list[dict[str, object]]) -> dict[int, list[float]]:
    """Return marker center points keyed by marker ID."""
    centers: dict[int, list[float]] = {}
    for marker in markers:
        corners = marker["corners"]
        x = sum(float(point[0]) for point in corners) / 4.0  # type: ignore[index]
        y = sum(float(point[1]) for point in corners) / 4.0  # type: ignore[index]
        centers[int(marker["id"])] = [x, y]
    return centers


def detect_aruco_centers(image: Any, dictionary_name: str = ARUCO_DICT_NAME) -> dict[int, list[float]]:
    """Backward-compatible helper returning only marker centers."""
    return marker_centers(detect_aruco_markers(image, dictionary_name=dictionary_name))
