"""Crater size classification placeholder module."""

from __future__ import annotations


class CraterClassifier:
    """Classify crater size after real-world measurement is available.

    Homography-based size estimation is intentionally not implemented yet.
    """

    def classify_size(self, detection: dict) -> dict:
        """Return the detection unchanged until size classification is added."""
        return detection


def classify_size(detection: dict) -> dict:
    """Functional wrapper for dummy crater size classification."""
    return CraterClassifier().classify_size(detection)
