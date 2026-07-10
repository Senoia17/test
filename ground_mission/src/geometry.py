import cv2
import numpy as np


def image_points_to_world(points_px, homography):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, homography)
    return world.reshape(-1, 2)


def bbox_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bbox_corners_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def polygon_size_mm(world_corners_cm):
    pts = np.array(world_corners_cm, dtype=np.float32)

    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])

    width_cm = (w1 + w2) / 2.0
    height_cm = (h1 + h2) / 2.0

    return width_cm * 10.0, height_cm * 10.0

