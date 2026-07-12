CRATER_SIZE_MM = {
    "small": (102.5, 99.0),
    "medium": (159.0, 150.0),
    "large": (179.0, 200.0),
}


def classify_crater_size(width_mm: float, height_mm: float) -> str:
    """Classify crater size by closest physical top-view area."""
    area = width_mm * height_mm

    distances = {}
    for name, (ref_w, ref_h) in CRATER_SIZE_MM.items():
        distances[name] = abs(area - (ref_w * ref_h))

    return min(distances, key=distances.get)

