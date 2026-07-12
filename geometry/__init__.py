"""Shared geometry layer for coordinate transforms, projections, and measurement."""

from geometry.measurement import physical_size, polygon_dimensions, polygon_size_mm
from geometry.projection import image_points_to_world, image_to_map, map_to_world
from geometry.transform import apply_homography, bbox_center, bbox_center_xyxy, bbox_corners, bbox_corners_xyxy

__all__ = [
    "apply_homography",
    "bbox_center",
    "bbox_center_xyxy",
    "bbox_corners",
    "bbox_corners_xyxy",
    "image_points_to_world",
    "image_to_map",
    "map_to_world",
    "physical_size",
    "polygon_dimensions",
    "polygon_size_mm",
]
