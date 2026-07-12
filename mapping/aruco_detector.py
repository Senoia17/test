"""Reusable ArUco marker detection for mapping and localization."""

from __future__ import annotations

from typing import Any

from mapping.aruco_config import resolve_dictionary_name


def _create_detector(dictionary_name: str | None = None) -> Any:
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


def detect_aruco_markers(image: Any, dictionary_name: str | None = None) -> list[dict[str, object]]:
    """Detect ArUco markers using the shared marker schema.

    Each marker is returned as::

        {"id": int, "corners": [[x, y], ...], "center": [x, y]}

    ``corners`` and ``center`` are image-space pixel coordinates.
    """
    import cv2

    dictionary_name = resolve_dictionary_name(dictionary_name)
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
        corners_xy = [[float(x), float(y)] for x, y in pts]
        center = [
            sum(point[0] for point in corners_xy) / 4.0,
            sum(point[1] for point in corners_xy) / 4.0,
        ]
        markers.append(
            {
                "id": int(marker_id),
                "corners": corners_xy,
                "center": center,
            }
        )
    return markers


def marker_centers(markers: list[dict[str, object]]) -> dict[int, list[float]]:
    """Return marker center points keyed by marker ID."""
    centers: dict[int, list[float]] = {}
    for marker in markers:
        center = marker.get("center")
        if center is not None:
            centers[int(marker["id"])] = [float(center[0]), float(center[1])]  # type: ignore[index]
            continue

        # Compatibility fallback for marker dictionaries produced before the
        # shared schema included an explicit center field.
        corners = marker["corners"]
        x = sum(float(point[0]) for point in corners) / 4.0  # type: ignore[index]
        y = sum(float(point[1]) for point in corners) / 4.0  # type: ignore[index]
        centers[int(marker["id"])] = [x, y]
    return centers


def detect_aruco_centers(image: Any, dictionary_name: str | None = None) -> dict[int, list[float]]:
    """Backward-compatible helper returning only marker centers."""
    return marker_centers(detect_aruco_markers(image, dictionary_name=dictionary_name))
