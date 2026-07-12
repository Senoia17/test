"""
Airfield global-map + TW/RW close-up localization pipeline
==========================================================

TW/RW-only optimized version:
  - ArUco 5:4 map rectification is preserved from the previous version.
  - FA(1).mp4 and FA(2).mp4 are not processed.
  - FA regions are not precomputed or drawn, reducing unnecessary feature matching work.
  - Strict route gating is kept to prevent visually similar zones from being selected out of order.

Input videos are supplied by the project mission router or config, not by hardcoded filenames.

Outputs:
  outputs/
    global_map.jpg
    global_map_with_regions.jpg
    index.html
    TWA/
      localization.csv
      trajectory.jpg
      localization.mp4
      debug_frames/*.jpg
    RW/
      ...
    TWB/
      ...

Current route direction:
  - TWA: TW-A5 -> TW-A4 -> TW-A3 -> TW-A2 -> TW-A1
  - RW : RW-01 -> RW-02 -> ... -> RW-10
  - TWB: TW-B5 -> TW-B4 -> TW-B3 -> TW-B2 -> TW-B1
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mapping.aruco_config import resolve_corner_ids, resolve_dictionary_name


# ============================================================
# 0. Global constants / configs
# ============================================================

REAL_FIELD_W_M = 3000.0
REAL_FIELD_H_M = 2400.0


@dataclass
class MapBuildConfig:
    # Sampling / speed
    frame_stride: int = 10
    max_keyframes: int = 140
    resize_width: int = 960

    # Feature matching
    prefer_sift: bool = False       # ORB is much faster for map building.
    orb_features: int = 5000
    sift_features: int = 4000
    max_match_candidates: int = 500
    min_good_matches: int = 45
    min_inliers: int = 35
    min_inlier_ratio: float = 0.20
    ransac_reproj_thr: float = 5.0

    # Mosaic quality / safety
    max_canvas_side: int = 5200     # Avoid exploding canvas size.
    crop_black_border: bool = True
    save_keyframe_debug: bool = False

    # ArUco-based final map rectification
    # The printed airfield map is known to be a 5:4 rectangle.
    # After building a rough mosaic, detect the four corner ArUco markers and
    # warp the map to this fixed aspect ratio so region splitting becomes stable.
    use_aruco_rectification: bool = True
    # Loaded from config.yaml when not provided explicitly.
    aruco_dictionary: Optional[str] = None
    # Optional marker IDs in FINAL MAP order: top-left, top-right, bottom-right, bottom-left.
    # Leave None to infer from marker positions.
    aruco_corner_ids: Optional[Tuple[int, ...]] = None
    aruco_min_markers: int = 4
    rectified_map_width: int = 1500
    rectified_map_height: int = 1200
    # The mapping video may be portrait, while the required global map is landscape 5:4.
    # When the detected ArUco quadrilateral is portrait, rotate the source corner order
    # before warping. "ccw" matches the expected portrait mapping-video orientation.
    aruco_rotate_portrait_to_landscape: bool = True
    aruco_portrait_rotation: str = "ccw"  # "ccw" or "cw"
    # If the stitched mosaic cannot detect all four markers, scan the provided video and
    # rectify the best single frame that contains four markers.
    aruco_best_frame_fallback: bool = True
    aruco_best_frame_stride: int = 5
    aruco_debug: bool = True


@dataclass
class LocalizerConfig:
    # ROI matching
    prefer_sift: bool = True
    roi_pad_ratio: float = 0.20
    # FA matching should include nearby boundary lines/roads because printed FA objects
    # may look very different from the real model seen in close-up frames.
    fa_roi_pad_ratio: float = 0.40

    # Preprocessing mode for feature matching.
    #   gray   : CLAHE-enhanced grayscale. Best for TW/RW line-rich regions.
    #   hybrid : gray + Canny edges. Better for FA appearance gap.
    #   edge   : mostly edges. Useful when color/texture differs heavily.
    match_preprocess_mode: str = "gray"

    min_good_matches: int = 18
    min_inliers_feature: int = 12
    min_inlier_ratio_feature: float = 0.23
    max_reproj_error_feature: float = 8.0
    ransac_reproj_thr: float = 5.0

    # Homography sanity checks
    min_area_ratio: float = 0.00015
    max_area_ratio: float = 0.42
    max_side_ratio: float = 38.0

    # Previous-transform prior
    soft_jump_m: float = 550.0
    hard_jump_m: float = 1400.0

    # Temporal tracking fallback
    use_temporal: bool = True
    min_temporal_tracks: int = 28
    min_temporal_inlier_ratio: float = 0.38
    max_temporal_reproj_error: float = 5.0

    # Speed/log control. If direct map feature matching succeeds, temporal tracking can
    # be skipped. This reduces warnings like too_few_temporal_inliers.
    skip_temporal_if_feature_ok: bool = True

    # FA safety net. When matching fails, fill the trace using known route order.
    # This does not create a true homography; it only provides an approximate center point
    # for trajectory/CSV continuity.
    enable_route_prior_fallback: bool = False


@dataclass
class PipelineConfig:
    input_dir: Path = Path(".")
    output_dir: Path = Path("outputs")

    top_view_video: Optional[str] = None
    global_map_name: str = "global_map.jpg"

    # Process every Nth frame in close-up videos.
    # 5 is a good first-run tradeoff. Use 1 if you need dense frame-by-frame output.
    localize_frame_stride: int = 5

    # Optional per-video override. FA videos usually need denser sampling because
    # texture/appearance matching is harder.
    per_video_frame_stride: Optional[Dict[str, int]] = None

    # Save one debug jpg every N processed frames. Set 1 to save every processed frame.
    debug_every_n_processed: int = 5

    # Output visualization video width. Larger = clearer but slower/heavier.
    vis_video_width: int = 1280
    vis_video_fps: float = 8.0

    # If True, rebuild global map even if outputs/global_map.jpg already exists.
    force_rebuild_global_map: bool = True


@dataclass
class VideoJob:
    video_file: str
    name: str
    route: List[str]
    start_route_idx: int = 0
    backtrack: int = 1
    lookahead: int = 2

    # Important:
    # If this is True, the first successful frame can match ANY zone in the route.
    # That is dangerous for TWA/TWB/RW because many zones look similar.
    # Keep False to force the first match to start near route[0].
    initial_wide_search: bool = False

    # "default" for TW/RW, "fa" for FA appearance-gap handling.
    localizer_profile: str = "default"


@dataclass
class Region:
    name: str
    bbox_norm: Tuple[float, float, float, float]

    def bbox_px(
        self,
        map_w: int,
        map_h: int,
        pad_ratio: float = 0.0,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_norm
        px1 = int(round(x1 * map_w))
        py1 = int(round(y1 * map_h))
        px2 = int(round(x2 * map_w))
        py2 = int(round(y2 * map_h))

        bw = max(px2 - px1, 1)
        bh = max(py2 - py1, 1)
        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))

        return (
            max(0, px1 - pad_x),
            max(0, py1 - pad_y),
            min(map_w, px2 + pad_x),
            min(map_h, py2 + pad_y),
        )


@dataclass
class MatchResult:
    ok: bool
    source: str = "none"  # feature / temporal / none
    zone: Optional[str] = None
    route_idx: Optional[int] = None
    H_frame_to_global: Optional[np.ndarray] = None
    center_px: Optional[Tuple[float, float]] = None
    center_m: Optional[Tuple[float, float]] = None
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_error: float = 9999.0
    area_ratio: float = 0.0
    score: float = -1e9
    reason: str = ""


# ============================================================
# 1. Route / region definitions
# ============================================================

def make_default_airfield_regions() -> Dict[str, Region]:
    """
    TW/RW-only normalized layout on the rectified 5:4 global map.

    FA bands are intentionally omitted in this version because FA(1)/FA(2)
    videos are no longer processed. This makes feature precomputation faster
    and keeps global_map_with_regions.jpg focused on the actually-used zones.

    IMPORTANT:
      Check outputs/global_map_with_regions.jpg after the first run.
      If the global map is cropped/rotated differently, tune y ranges below.
    """
    regions: Dict[str, Region] = {}

    # Slight overlaps reduce boundary failures.
    tw_overlap_x = 0.025
    rw_overlap_x = 0.015

    # Rectified map is 5:4. These bands match the current paper-map layout.
    tw_a_y = (0.18, 0.42)
    rw_y = (0.36, 0.64)
    tw_b_y = (0.58, 0.82)

    for i in range(5):
        x1 = max(0.0, i / 5.0 - tw_overlap_x)
        x2 = min(1.0, (i + 1) / 5.0 + tw_overlap_x)
        name = f"TW-A{i + 1}"
        regions[name] = Region(name, (x1, tw_a_y[0], x2, tw_a_y[1]))

    for i in range(10):
        x1 = max(0.0, i / 10.0 - rw_overlap_x)
        x2 = min(1.0, (i + 1) / 10.0 + rw_overlap_x)
        name = f"RW-{i + 1:02d}"
        regions[name] = Region(name, (x1, rw_y[0], x2, rw_y[1]))

    for i in range(5):
        x1 = max(0.0, i / 5.0 - tw_overlap_x)
        x2 = min(1.0, (i + 1) / 5.0 + tw_overlap_x)
        name = f"TW-B{i + 1}"
        regions[name] = Region(name, (x1, tw_b_y[0], x2, tw_b_y[1]))

    return regions


def make_default_video_jobs(rw_reverse: bool = False) -> List[VideoJob]:
    """
    TW/RW-only video jobs.

    Current default route direction:
      TW-A: TW-A5 -> TW-A4 -> TW-A3 -> TW-A2 -> TW-A1
      RW  : RW-01 -> RW-02 -> ... -> RW-10 by default
      TW-B: TW-B5 -> TW-B4 -> TW-B3 -> TW-B2 -> TW-B1

    Video filenames must be supplied by callers; placeholder names are labels only.
    """
    rw_route = [f"RW-{i:02d}" for i in range(1, 11)]
    if rw_reverse:
        rw_route = list(reversed(rw_route))

    return [
        # Strict route gating prevents the first frame of reverse routes such as
        # TWA/TWB from being incorrectly locked to a visually similar end zone.
        VideoJob(
            "", "TWA",
            [f"TW-A{i}" for i in range(5, 0, -1)],
            backtrack=0, lookahead=2,
            initial_wide_search=False,
            localizer_profile="default",
        ),
        VideoJob(
            "", "RW",
            rw_route,
            backtrack=0, lookahead=2,
            initial_wide_search=False,
            localizer_profile="default",
        ),
        VideoJob(
            "", "TWB",
            [f"TW-B{i}" for i in range(5, 0, -1)],
            backtrack=0, lookahead=2,
            initial_wide_search=False,
            localizer_profile="default",
        ),
    ]


# ============================================================
# 2. Utility functions
# ============================================================

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_keep_width(img: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    if target_width is None:
        return img
    h, w = img.shape[:2]
    if w <= target_width:
        return img
    scale = target_width / float(w)
    return cv2.resize(img, (target_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def resize_keep_width_with_scale(img: np.ndarray, target_width: Optional[int]) -> Tuple[np.ndarray, float]:
    if target_width is None:
        return img, 1.0
    h, w = img.shape[:2]
    if w == target_width:
        return img, 1.0
    scale = target_width / float(w)
    return cv2.resize(img, (target_width, int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def enhance_gray(img: np.ndarray) -> np.ndarray:
    """Backward-compatible grayscale enhancement used by map stitching."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_for_matching(img: np.ndarray, mode: str = "gray") -> np.ndarray:
    """
    Convert a frame/map ROI into a matching-friendly single-channel image.

    Modes:
      - gray:   CLAHE-enhanced grayscale. Stable and fast for RW/TW.
      - hybrid: sharpened gray + edges. Recommended for FA where the real model
                appearance differs from the printed black/white global map.
      - edge:   edge-only representation. More aggressive; useful for debugging FA.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    mode = (mode or "gray").lower()

    # CLAHE helps both printed-map shadows and close-up exposure changes.
    clip = 2.8 if mode in {"hybrid", "edge"} else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if mode == "gray":
        return gray

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)

    # Adaptive thresholds for Canny: more robust across dark/bright frames.
    med = float(np.median(sharp))
    low = int(max(20, 0.66 * med))
    high = int(min(220, 1.33 * med + 20))
    if high <= low:
        high = low + 40
    edges = cv2.Canny(sharp, low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    if mode == "edge":
        return edges

    # hybrid keeps gradient/texture while giving line structure extra weight.
    return cv2.addWeighted(sharp, 0.68, edges, 0.32, 0)


def pixel_to_meter(xy_px: Tuple[float, float], map_w: int, map_h: int) -> Tuple[float, float]:
    x_px, y_px = xy_px
    return float(x_px / map_w * REAL_FIELD_W_M), float(y_px / map_h * REAL_FIELD_H_M)


def pixel_dist_to_meter(dx_px: float, dy_px: float, map_w: int, map_h: int) -> float:
    dx_m = dx_px / map_w * REAL_FIELD_W_M
    dy_m = dy_px / map_h * REAL_FIELD_H_M
    return float(math.sqrt(dx_m * dx_m + dy_m * dy_m))


def transform_points_to_global(points_xy: np.ndarray, H_frame_to_global: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_frame_to_global)
    return out.reshape(-1, 2)


def read_sampled_video_frames(
    video_path: Path | str,
    stride: int,
    max_frames: Optional[int],
    resize_width: Optional[int],
) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0
    stride = max(1, int(stride))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            frame = resize_keep_width(frame, resize_width)
            frames.append((frame_idx, frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def put_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - baseline - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def order_points_clockwise(points_xy: np.ndarray) -> np.ndarray:
    """Return points in tl, tr, br, bl order."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _get_aruco_dictionary(dictionary_name: str):
    """OpenCV ArUco dictionary loader with clear error messages."""
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is unavailable. Install OpenCV contrib build: "
            "python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python "
            "opencv-contrib-python-headless && "
            "python -m pip install opencv-contrib-python-headless numpy"
        )

    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        available = [name for name in dir(aruco) if name.startswith("DICT_")]
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}. Available examples: {available[:10]}")

    dict_id = getattr(aruco, dictionary_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def detect_aruco_markers(
    image_bgr: np.ndarray,
    dictionary_name: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Detect ArUco markers and return dictionaries with id, corners and center.
    Compatible with both newer and older OpenCV ArUco APIs.
    """
    dictionary_name = resolve_dictionary_name(dictionary_name)
    aruco = cv2.aruco
    dictionary = _get_aruco_dictionary(dictionary_name)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    markers: List[Dict[str, object]] = []
    if ids is None:
        return markers

    for marker_id, marker_corners in zip(ids.reshape(-1).tolist(), corners):
        pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        markers.append({
            "id": int(marker_id),
            "corners": pts.astype(float).tolist(),
            "center": center.astype(float).tolist(),
        })
    return markers


def _marker_dicts_ordered_by_centers(markers: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return marker dicts in spatial TL, TR, BR, BL order using centers."""
    centers = np.array([m["center"] for m in markers], dtype=np.float32)
    ordered_centers = order_points_clockwise(centers)

    ordered: List[Dict[str, object]] = []
    used = set()
    for target in ordered_centers:
        best_idx = None
        best_dist = 1e18
        for i, m in enumerate(markers):
            if i in used:
                continue
            d = float(np.linalg.norm(np.asarray(m["center"], dtype=np.float32) - target))
            if d < best_dist:
                best_dist = d
                best_idx = i
        assert best_idx is not None
        used.add(best_idx)
        ordered.append(markers[best_idx])
    return ordered


def _quad_is_portrait_from_markers(markers_tl_tr_br_bl: List[Dict[str, object]]) -> bool:
    """Return True when the detected source quad is taller than it is wide."""
    c = np.array([m["center"] for m in markers_tl_tr_br_bl], dtype=np.float32)
    tl, tr, br, bl = c
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return bool(height > width * 1.05)


def _choose_four_corner_markers(
    markers: List[Dict[str, object]],
    corner_ids: Optional[Tuple[int, ...]] = None,
    rotate_portrait_to_landscape: bool = True,
    portrait_rotation: str = "ccw",
) -> List[Dict[str, object]]:
    """
    Select and order four corner markers in FINAL OUTPUT MAP order: TL, TR, BR, BL.

    If corner_ids is given, it must already be in final map order: TL, TR, BR, BL.
    If corner_ids is None, the markers are first ordered spatially in the source image.
    When the source image is portrait but the output map is landscape 5:4, the source
    order is rotated before warping so RW-01~RW-10 becomes horizontal.
    """
    if len(markers) < 4:
        raise ValueError(f"Need at least 4 ArUco markers, detected {len(markers)}.")

    if corner_ids is not None:
        by_id = {int(m["id"]): m for m in markers}
        missing = [int(mid) for mid in corner_ids if int(mid) not in by_id]
        if missing:
            raise ValueError(f"Configured ArUco corner IDs not detected: {missing}")
        return [by_id[int(mid)] for mid in corner_ids]

    # Keep four spatially extreme markers when more than four are visible.
    centers = np.array([m["center"] for m in markers], dtype=np.float32)
    extreme_targets = order_points_clockwise(centers)

    selected: List[Dict[str, object]] = []
    used = set()
    for target in extreme_targets:
        best_idx = None
        best_dist = 1e18
        for i, m in enumerate(markers):
            if i in used:
                continue
            d = float(np.linalg.norm(np.asarray(m["center"], dtype=np.float32) - target))
            if d < best_dist:
                best_dist = d
                best_idx = i
        assert best_idx is not None
        used.add(best_idx)
        selected.append(markers[best_idx])

    spatial = _marker_dicts_ordered_by_centers(selected)  # source-image TL, TR, BR, BL

    if rotate_portrait_to_landscape and _quad_is_portrait_from_markers(spatial):
        rot = portrait_rotation.lower().strip()
        if rot == "ccw":
            # Source right side becomes final top side.
            # This matches portrait source videos where the ID text side with FA-01~03 becomes top.
            return [spatial[1], spatial[2], spatial[3], spatial[0]]
        if rot == "cw":
            # Source left side becomes final top side.
            return [spatial[3], spatial[0], spatial[1], spatial[2]]
        raise ValueError("portrait_rotation must be 'ccw' or 'cw'")

    return spatial


def estimate_map_corners_from_aruco(
    image_bgr: np.ndarray,
    dictionary_name: Optional[str] = None,
    corner_ids: Optional[Tuple[int, ...]] = None,
    rotate_portrait_to_landscape: bool = True,
    portrait_rotation: str = "ccw",
) -> Tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Estimate true map quadrilateral from the four corner ArUco markers.

    Returns:
        src_corners: source-image points in FINAL MAP TL, TR, BR, BL order.
        selected_markers: selected markers in FINAL MAP TL, TR, BR, BL order.
        all_markers: all detected markers.
    """
    dictionary_name = resolve_dictionary_name(dictionary_name)
    corner_ids = resolve_corner_ids(corner_ids)

    markers = detect_aruco_markers(image_bgr, dictionary_name=dictionary_name)
    selected = _choose_four_corner_markers(
        markers,
        corner_ids=corner_ids,
        rotate_portrait_to_landscape=rotate_portrait_to_landscape,
        portrait_rotation=portrait_rotation,
    )

    marker_centers = np.array([m["center"] for m in selected], dtype=np.float32)
    map_center = marker_centers.mean(axis=0)

    outer_corners = []
    for marker in selected:
        pts = np.asarray(marker["corners"], dtype=np.float32).reshape(4, 2)
        # The map-corner side is the ArUco corner farthest away from the whole-map center.
        idx = int(np.argmax(np.linalg.norm(pts - map_center.reshape(1, 2), axis=1)))
        outer_corners.append(pts[idx])

    # Do NOT reorder again here. selected is already in final output TL, TR, BR, BL order.
    src_corners = np.array(outer_corners, dtype=np.float32)
    return src_corners, selected, markers


def _markers_to_jsonable_in_rectified(
    markers: List[Dict[str, object]],
    H_src_to_rectified: np.ndarray,
    roles: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Transform marker corners/centers to rectified-map coordinates for report/overlay."""
    out: List[Dict[str, object]] = []
    for i, m in enumerate(markers):
        pts = np.asarray(m["corners"], dtype=np.float32).reshape(-1, 1, 2)
        center = np.asarray(m["center"], dtype=np.float32).reshape(1, 1, 2)
        pts_r = cv2.perspectiveTransform(pts, H_src_to_rectified).reshape(-1, 2)
        center_r = cv2.perspectiveTransform(center, H_src_to_rectified).reshape(2)
        item = {
            "id": int(m["id"]),
            "center": [float(round(center_r[0], 3)), float(round(center_r[1], 3))],
            "corners": np.round(pts_r.astype(float), 3).tolist(),
        }
        if roles is not None and i < len(roles):
            item["role"] = roles[i]
        out.append(item)
    return out


def rectify_map_with_aruco_corners(
    image_bgr: np.ndarray,
    out_size: Tuple[int, int] = (1500, 1200),
    dictionary_name: Optional[str] = None,
    corner_ids: Optional[Tuple[int, ...]] = None,
    rotate_portrait_to_landscape: bool = True,
    portrait_rotation: str = "ccw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Perspective-warp mosaic/frame to a fixed 5:4 rectangle using ArUco corner markers.
    out_size is (width, height), e.g. 1500x1200.
    """
    out_w, out_h = out_size
    src_corners, selected_markers, all_markers = estimate_map_corners_from_aruco(
        image_bgr,
        dictionary_name=dictionary_name,
        corner_ids=corner_ids,
        rotate_portrait_to_landscape=rotate_portrait_to_landscape,
        portrait_rotation=portrait_rotation,
    )
    dst_corners = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    rectified = cv2.warpPerspective(image_bgr, H, (out_w, out_h))
    return rectified, H, src_corners, selected_markers, all_markers


def find_best_aruco_frame_in_video(
    video_path: Path | str,
    dictionary_name: Optional[str] = None,
    frame_stride: int = 5,
    min_markers: int = 4,
    max_scan_frames: Optional[int] = None,
) -> Tuple[np.ndarray, int, List[Dict[str, object]]]:
    """Find a top-view frame with the strongest four-corner ArUco detection."""
    dictionary_name = resolve_dictionary_name(dictionary_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video for ArUco fallback: {video_path}")

    best_frame = None
    best_idx = -1
    best_markers: List[Dict[str, object]] = []
    best_score = -1.0

    idx = 0
    scanned = 0
    stride = max(1, int(frame_stride))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride != 0:
            idx += 1
            continue

        markers = detect_aruco_markers(frame, dictionary_name=dictionary_name)
        if len(markers) >= min_markers:
            centers = np.array([m["center"] for m in markers], dtype=np.float32)
            ordered = order_points_clockwise(centers)
            area = abs(cv2.contourArea(ordered.astype(np.float32)))
            score = len(markers) * 1_000_000.0 + float(area)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_idx = idx
                best_markers = markers

        scanned += 1
        if max_scan_frames is not None and scanned >= max_scan_frames:
            break
        idx += 1

    cap.release()

    if best_frame is None:
        raise ValueError(f"No frame with at least {min_markers} ArUco markers was found in {video_path}")

    return best_frame, best_idx, best_markers


def draw_aruco_rectification_debug(
    image_bgr: np.ndarray,
    src_corners: np.ndarray,
    selected_markers: List[Dict[str, object]],
    all_markers: Optional[List[Dict[str, object]]] = None,
) -> np.ndarray:
    """Create debug image showing detected marker IDs and selected crop polygon."""
    debug = image_bgr.copy()

    if all_markers is None:
        all_markers = selected_markers

    selected_ids = {int(m["id"]) for m in selected_markers}
    for m in all_markers:
        pts = np.asarray(m["corners"], dtype=np.int32).reshape(-1, 1, 2)
        is_selected = int(m["id"]) in selected_ids
        color = (0, 255, 255) if is_selected else (180, 180, 180)
        cv2.polylines(debug, [pts], True, color, 2)
        cx, cy = np.asarray(m["center"], dtype=np.float32)
        cv2.circle(debug, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        put_label(debug, f"ID {m['id']}", (int(cx) + 8, int(cy) - 8), font_scale=0.6)

    poly = np.asarray(src_corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [poly], True, (0, 0, 255), 4)
    names = ["TL", "TR", "BR", "BL"]
    for name, p in zip(names, src_corners):
        cv2.circle(debug, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
        put_label(debug, name, (int(p[0]) + 8, int(p[1]) + 8), font_scale=0.7, bg_color=(0, 0, 180))
    return debug


# ============================================================
# 3. Feature backend
# ============================================================

class FeatureBackend:
    def __init__(
        self,
        prefer_sift: bool = True,
        orb_features: int = 8000,
        sift_features: int = 6000,
    ):
        self.kind: str
        self.detector: cv2.Feature2D
        self.norm: int
        self.ratio: float
        self.use_crosscheck: bool

        if prefer_sift and hasattr(cv2, "SIFT_create"):
            self.kind = "SIFT"
            self.detector = cv2.SIFT_create(nfeatures=sift_features)
            self.norm = cv2.NORM_L2
            self.ratio = 0.75
            self.use_crosscheck = False
        else:
            self.kind = "ORB"
            self.detector = cv2.ORB_create(
                nfeatures=orb_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                fastThreshold=8,
            )
            self.norm = cv2.NORM_HAMMING
            self.ratio = 0.82
            self.use_crosscheck = False

        self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

    def detect(self, gray: np.ndarray, mask: Optional[np.ndarray] = None):
        return self.detector.detectAndCompute(gray, mask)

    def ratio_match(self, des_query, des_train) -> List[cv2.DMatch]:
        if des_query is None or des_train is None:
            return []
        raw = self.matcher.knnMatch(des_query, des_train, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good


# ============================================================
# 4. Fast top-view mosaic builder
# ============================================================

class FastTopViewMosaicBuilder:
    """
    Faster than repeatedly matching every new frame to the full growing mosaic.

    It estimates pairwise transforms:
      current_frame -> previous_accepted_frame
    and accumulates them into:
      current_frame -> first_frame_reference

    Then it warps all accepted keyframes once onto a single canvas.
    """

    def __init__(self, config: MapBuildConfig):
        self.cfg = config
        if self.cfg.aruco_dictionary is None:
            self.cfg.aruco_dictionary = resolve_dictionary_name(None)
        if self.cfg.aruco_corner_ids is None:
            self.cfg.aruco_corner_ids = resolve_corner_ids(None)
        self.backend = FeatureBackend(
            prefer_sift=config.prefer_sift,
            orb_features=config.orb_features,
            sift_features=config.sift_features,
        )

    def _compute_homography_pair(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        kp_prev, des_prev = self.backend.detect(prev_gray)
        kp_curr, des_curr = self.backend.detect(curr_gray)

        if des_prev is None or des_curr is None or len(kp_prev) < 4 or len(kp_curr) < 4:
            return None, {"matches": 0, "inliers": 0, "inlier_ratio": 0.0, "reproj_error": 9999.0}

        # Query=current, Train=previous, so output H maps current -> previous.
        good = self.backend.ratio_match(des_curr, des_prev)
        good = sorted(good, key=lambda m: m.distance)[: self.cfg.max_match_candidates]

        if len(good) < self.cfg.min_good_matches:
            return None, {"matches": len(good), "inliers": 0, "inlier_ratio": 0.0, "reproj_error": 9999.0}

        src = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_reproj_thr)
        if H is None or mask is None:
            return None, {"matches": len(good), "inliers": 0, "inlier_ratio": 0.0, "reproj_error": 9999.0}

        mask_bool = mask.reshape(-1).astype(bool)
        inliers = int(mask_bool.sum())
        inlier_ratio = inliers / max(len(good), 1)

        if inliers < self.cfg.min_inliers or inlier_ratio < self.cfg.min_inlier_ratio:
            return None, {"matches": len(good), "inliers": inliers, "inlier_ratio": inlier_ratio, "reproj_error": 9999.0}

        proj = cv2.perspectiveTransform(src[mask_bool], H).reshape(-1, 2)
        dst_in = dst[mask_bool].reshape(-1, 2)
        reproj_error = float(np.median(np.linalg.norm(proj - dst_in, axis=1)))

        H = H / H[2, 2]
        return H, {
            "matches": len(good),
            "inliers": inliers,
            "inlier_ratio": float(inlier_ratio),
            "reproj_error": reproj_error,
        }

    def _estimate_canvas(
        self,
        frames: Sequence[np.ndarray],
        H_to_ref: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, int, int, Tuple[int, int, int, int]]:
        all_corners = []
        for frame, H in zip(frames, H_to_ref):
            h, w = frame.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            all_corners.append(warped)

        all_corners_arr = np.concatenate(all_corners, axis=0)
        x_min, y_min = np.floor(all_corners_arr.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners_arr.max(axis=0)).astype(int)

        canvas_w = int(x_max - x_min)
        canvas_h = int(y_max - y_min)

        if canvas_w <= 0 or canvas_h <= 0:
            raise RuntimeError("Invalid mosaic canvas size.")

        max_side = max(canvas_w, canvas_h)
        scale = 1.0
        if max_side > self.cfg.max_canvas_side:
            scale = self.cfg.max_canvas_side / float(max_side)
            canvas_w = int(round(canvas_w * scale))
            canvas_h = int(round(canvas_h * scale))

        T = np.array(
            [
                [scale, 0, -x_min * scale],
                [0, scale, -y_min * scale],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        return T, canvas_w, canvas_h, (int(x_min), int(y_min), int(x_max), int(y_max))

    def _crop_valid_area(self, mosaic: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if not self.cfg.crop_black_border:
            return mosaic
        mask = weight > 0
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return mosaic
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        return mosaic[y1:y2, x1:x2]

    def build(self, video_path: Path | str, output_path: Path | str, debug_dir: Optional[Path | str] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        video_path = Path(video_path)
        output_path = Path(output_path)
        if debug_dir is not None:
            ensure_dir(debug_dir)

        sampled = read_sampled_video_frames(
            video_path=video_path,
            stride=self.cfg.frame_stride,
            max_frames=self.cfg.max_keyframes,
            resize_width=self.cfg.resize_width,
        )
        if len(sampled) < 2:
            raise ValueError(f"Not enough sampled frames from {video_path}")

        accepted_frames: List[np.ndarray] = []
        accepted_source_indices: List[int] = []
        H_to_ref: List[np.ndarray] = []
        pair_logs: List[Dict[str, object]] = []

        first_idx, first_frame = sampled[0]
        accepted_frames.append(first_frame)
        accepted_source_indices.append(first_idx)
        H_to_ref.append(np.eye(3, dtype=np.float64))

        prev_gray = enhance_gray(first_frame)
        prev_H_to_ref = np.eye(3, dtype=np.float64)
        prev_source_idx = first_idx

        print(f"[MAP] Feature={self.backend.kind} | sampled={len(sampled)} | first_frame={first_idx}")

        for source_idx, frame in sampled[1:]:
            curr_gray = enhance_gray(frame)
            H_curr_to_prev, info = self._compute_homography_pair(prev_gray, curr_gray)
            info_log = {"source_frame": source_idx, "prev_source_frame": prev_source_idx, **info}

            if H_curr_to_prev is None:
                info_log["status"] = "skip"
                pair_logs.append(info_log)
                print(
                    f"[MAP/SKIP] frame={source_idx} prev={prev_source_idx} "
                    f"matches={info['matches']} inliers={info['inliers']} ratio={info['inlier_ratio']:.2f}"
                )
                continue

            H_curr_to_ref = prev_H_to_ref @ H_curr_to_prev
            H_curr_to_ref = H_curr_to_ref / H_curr_to_ref[2, 2]

            accepted_frames.append(frame)
            accepted_source_indices.append(source_idx)
            H_to_ref.append(H_curr_to_ref)
            pair_logs.append({**info_log, "status": "ok"})

            prev_gray = curr_gray
            prev_H_to_ref = H_curr_to_ref
            prev_source_idx = source_idx

            print(
                f"[MAP/OK] frame={source_idx} accepted={len(accepted_frames)} "
                f"inliers={info['inliers']} ratio={info['inlier_ratio']:.2f} err={info['reproj_error']:.2f}"
            )

        if len(accepted_frames) < 2:
            raise RuntimeError("Too few accepted keyframes. Try lowering frame_stride or using prefer_sift=True.")

        T, canvas_w, canvas_h, bounds = self._estimate_canvas(accepted_frames, H_to_ref)
        print(f"[MAP] canvas={canvas_w}x{canvas_h} accepted_keyframes={len(accepted_frames)}")

        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i, (frame, H) in enumerate(zip(accepted_frames, H_to_ref)):
            M = T @ H
            warped = cv2.warpPerspective(frame, M, (canvas_w, canvas_h))
            mask_src = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask_src, M, (canvas_w, canvas_h))
            mask = (warped_mask > 0).astype(np.float32)

            accum += warped.astype(np.float32) * mask[..., None]
            weight += mask

            if self.cfg.save_keyframe_debug and debug_dir is not None:
                cv2.imwrite(str(Path(debug_dir) / f"warped_keyframe_{i:03d}_src_{accepted_source_indices[i]:06d}.jpg"), warped)

        safe_weight = np.maximum(weight, 1e-6)
        raw_mosaic = (accum / safe_weight[..., None]).clip(0, 255).astype(np.uint8)
        raw_mosaic = self._crop_valid_area(raw_mosaic, weight)

        ensure_dir(output_path.parent)
        if debug_dir is not None:
            cv2.imwrite(str(Path(debug_dir) / "raw_mosaic_before_aruco_rectification.jpg"), raw_mosaic)

        mosaic = raw_mosaic
        aruco_report: Dict[str, object] = {"enabled": bool(self.cfg.use_aruco_rectification), "status": "not_run"}

        if self.cfg.use_aruco_rectification:
            try:
                rectified, H_aruco, src_corners, selected_markers, all_markers = rectify_map_with_aruco_corners(
                    raw_mosaic,
                    out_size=(self.cfg.rectified_map_width, self.cfg.rectified_map_height),
                    dictionary_name=self.cfg.aruco_dictionary,
                    corner_ids=self.cfg.aruco_corner_ids,
                    rotate_portrait_to_landscape=self.cfg.aruco_rotate_portrait_to_landscape,
                    portrait_rotation=self.cfg.aruco_portrait_rotation,
                )
                mosaic = rectified

                if debug_dir is not None and self.cfg.aruco_debug:
                    debug_img = draw_aruco_rectification_debug(raw_mosaic, src_corners, selected_markers, all_markers)
                    cv2.imwrite(str(Path(debug_dir) / "aruco_selected_corners_on_raw_mosaic.jpg"), debug_img)
                    cv2.imwrite(str(Path(debug_dir) / "rectified_5x4_global_map.jpg"), mosaic)

                aruco_report = {
                    "enabled": True,
                    "status": "ok",
                    "source": "raw_mosaic",
                    "dictionary": self.cfg.aruco_dictionary,
                    "rectified_size": {
                        "width": int(self.cfg.rectified_map_width),
                        "height": int(self.cfg.rectified_map_height),
                        "aspect_ratio": float(self.cfg.rectified_map_width / self.cfg.rectified_map_height),
                    },
                    "all_detected_marker_ids": [int(m["id"]) for m in all_markers],
                    "selected_marker_ids_tl_tr_br_bl": [int(m["id"]) for m in selected_markers],
                    "source_corners_tl_tr_br_bl": src_corners.astype(float).round(3).tolist(),
                    "rectified_selected_markers_tl_tr_br_bl": _markers_to_jsonable_in_rectified(selected_markers, H_aruco, roles=["TL", "TR", "BR", "BL"]),
                    "H_source_to_rectified": H_aruco.astype(float).reshape(-1).tolist(),
                }
                print(
                    "[MAP/ARUCO] rectified with markers "
                    f"{aruco_report['selected_marker_ids_tl_tr_br_bl']} -> "
                    f"{self.cfg.rectified_map_width}x{self.cfg.rectified_map_height}"
                )
            except Exception as e:
                print(f"[MAP/ARUCO/WARN] raw mosaic rectification failed: {e}")
                if self.cfg.aruco_best_frame_fallback:
                    try:
                        best_frame, best_idx, best_markers0 = find_best_aruco_frame_in_video(
                            video_path=video_path,
                            dictionary_name=self.cfg.aruco_dictionary,
                            frame_stride=self.cfg.aruco_best_frame_stride,
                            min_markers=self.cfg.aruco_min_markers,
                        )
                        rectified, H_aruco, src_corners, selected_markers, all_markers = rectify_map_with_aruco_corners(
                            best_frame,
                            out_size=(self.cfg.rectified_map_width, self.cfg.rectified_map_height),
                            dictionary_name=self.cfg.aruco_dictionary,
                            corner_ids=self.cfg.aruco_corner_ids,
                            rotate_portrait_to_landscape=self.cfg.aruco_rotate_portrait_to_landscape,
                            portrait_rotation=self.cfg.aruco_portrait_rotation,
                        )
                        mosaic = rectified
                        if debug_dir is not None and self.cfg.aruco_debug:
                            cv2.imwrite(str(Path(debug_dir) / f"aruco_best_frame_{best_idx:06d}.jpg"), best_frame)
                            debug_img = draw_aruco_rectification_debug(best_frame, src_corners, selected_markers, all_markers)
                            cv2.imwrite(str(Path(debug_dir) / "aruco_selected_corners_on_best_frame.jpg"), debug_img)
                            cv2.imwrite(str(Path(debug_dir) / "rectified_5x4_global_map.jpg"), mosaic)

                        aruco_report = {
                            "enabled": True,
                            "status": "ok",
                            "source": "best_video_frame",
                            "best_frame_index": int(best_idx),
                            "dictionary": self.cfg.aruco_dictionary,
                            "rectified_size": {
                                "width": int(self.cfg.rectified_map_width),
                                "height": int(self.cfg.rectified_map_height),
                                "aspect_ratio": float(self.cfg.rectified_map_width / self.cfg.rectified_map_height),
                            },
                            "all_detected_marker_ids": [int(m["id"]) for m in all_markers],
                            "selected_marker_ids_tl_tr_br_bl": [int(m["id"]) for m in selected_markers],
                            "source_corners_tl_tr_br_bl": src_corners.astype(float).round(3).tolist(),
                            "rectified_selected_markers_tl_tr_br_bl": _markers_to_jsonable_in_rectified(selected_markers, H_aruco, roles=["TL", "TR", "BR", "BL"]),
                            "H_source_to_rectified": H_aruco.astype(float).reshape(-1).tolist(),
                        }
                        print(
                            "[MAP/ARUCO] rectified from best video frame "
                            f"{best_idx} with markers {aruco_report['selected_marker_ids_tl_tr_br_bl']} -> "
                            f"{self.cfg.rectified_map_width}x{self.cfg.rectified_map_height}"
                        )
                    except Exception as e2:
                        aruco_report = {
                            "enabled": True,
                            "status": "failed_fallback_to_raw_mosaic",
                            "dictionary": self.cfg.aruco_dictionary,
                            "raw_mosaic_error": str(e),
                            "best_frame_error": str(e2),
                        }
                        print(f"[MAP/ARUCO/WARN] best-frame fallback failed: {e2}")
                        print("[MAP/ARUCO/WARN] Fallback: using unrectified raw mosaic.")
                else:
                    aruco_report = {
                        "enabled": True,
                        "status": "failed_fallback_to_raw_mosaic",
                        "dictionary": self.cfg.aruco_dictionary,
                        "error": str(e),
                    }
                    print("[MAP/ARUCO/WARN] Fallback: using unrectified raw mosaic.")

        cv2.imwrite(str(output_path), mosaic)

        report = {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "feature_backend": self.backend.kind,
            "sampled_frames": len(sampled),
            "accepted_keyframes": len(accepted_frames),
            "accepted_source_indices": accepted_source_indices,
            "canvas_before_crop": {"width": canvas_w, "height": canvas_h, "bounds_ref": bounds},
            "raw_mosaic_shape": {"height": int(raw_mosaic.shape[0]), "width": int(raw_mosaic.shape[1])},
            "mosaic_shape": {"height": int(mosaic.shape[0]), "width": int(mosaic.shape[1])},
            "aruco_rectification": aruco_report,
            "pair_logs": pair_logs,
            "config": asdict(self.cfg),
        }
        with open(output_path.parent / "global_map_build_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[MAP/DONE] saved: {output_path}")
        return mosaic, report


# ============================================================
# 5. Close-up localizer
# ============================================================

class RouteState:
    def __init__(
        self,
        route: List[str],
        start_idx: int = 0,
        backtrack: int = 1,
        lookahead: int = 2,
        initial_wide_search: bool = True,
    ):
        self.route = route
        self.idx = start_idx
        self.backtrack = backtrack
        self.lookahead = lookahead
        self.initial_wide_search = initial_wide_search
        self.locked = False
        self.zone_to_idx = {z: i for i, z in enumerate(route)}

    def candidates(self) -> List[str]:
        if self.initial_wide_search and not self.locked:
            return self.route
        lo = max(0, self.idx - self.backtrack)
        hi = min(len(self.route), self.idx + self.lookahead + 1)
        return self.route[lo:hi]

    def update(self, zone: Optional[str]) -> None:
        if zone is None or zone not in self.zone_to_idx:
            return
        new_idx = self.zone_to_idx[zone]
        if not self.locked:
            self.idx = new_idx
            self.locked = True
        else:
            self.idx = max(self.idx, new_idx)

    def route_idx_of(self, zone: Optional[str]) -> Optional[int]:
        if zone is None:
            return None
        return self.zone_to_idx.get(zone)


class CloseupToGlobalLocalizer:
    def __init__(
        self,
        global_map_bgr: np.ndarray,
        regions: Dict[str, Region],
        config: Optional[LocalizerConfig] = None,
    ):
        self.global_map_bgr = global_map_bgr
        self.config = config or LocalizerConfig()
        self.global_map_match = preprocess_for_matching(global_map_bgr, self.config.match_preprocess_mode)
        self.map_h, self.map_w = self.global_map_match.shape[:2]
        self.regions = regions
        self.backend = FeatureBackend(prefer_sift=self.config.prefer_sift)
        self.region_features: Dict[str, Dict[str, object]] = {}
        self._precompute_region_features()

        print(
            f"[LOC] Feature={self.backend.kind} | map={self.map_w}x{self.map_h} "
            f"| preprocess={self.config.match_preprocess_mode} | regions={len(self.region_features)}"
        )

    def _region_pad_ratio(self, region_name: str) -> float:
        if region_name.startswith("FA-"):
            return self.config.fa_roi_pad_ratio
        return self.config.roi_pad_ratio

    def _precompute_region_features(self) -> None:
        for name, region in self.regions.items():
            pad = self._region_pad_ratio(name)
            x1, y1, x2, y2 = region.bbox_px(self.map_w, self.map_h, pad_ratio=pad)
            crop = self.global_map_match[y1:y2, x1:x2]
            kp, des = self.backend.detect(crop)
            self.region_features[name] = {"bbox": (x1, y1, x2, y2), "kp": kp, "des": des, "pad": pad}

    def _project_frame_corners(self, H_frame_to_global: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(corners, H_frame_to_global).reshape(-1, 2)

    def _validate_projected_polygon(
        self,
        H_frame_to_global: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[bool, str, Tuple[float, float], float]:
        poly = self._project_frame_corners(H_frame_to_global, frame_shape)
        center_frame = np.array([[[frame_shape[1] / 2.0, frame_shape[0] / 2.0]]], dtype=np.float32)
        center_global = cv2.perspectiveTransform(center_frame, H_frame_to_global)
        cx, cy = center_global.reshape(-1, 2)[0]

        if not (0 <= cx < self.map_w and 0 <= cy < self.map_h):
            return False, "center_out_of_map", (float(cx), float(cy)), 0.0

        margin = max(self.map_w, self.map_h) * 0.06
        if not (
            np.all(poly[:, 0] >= -margin)
            and np.all(poly[:, 0] <= self.map_w + margin)
            and np.all(poly[:, 1] >= -margin)
            and np.all(poly[:, 1] <= self.map_h + margin)
        ):
            return False, "projected_polygon_out_of_map", (float(cx), float(cy)), 0.0

        area = abs(cv2.contourArea(poly.astype(np.float32)))
        area_ratio = area / float(self.map_w * self.map_h)
        if area_ratio < self.config.min_area_ratio:
            return False, "projected_area_too_small", (float(cx), float(cy)), float(area_ratio)
        if area_ratio > self.config.max_area_ratio:
            return False, "projected_area_too_large", (float(cx), float(cy)), float(area_ratio)

        side_lengths = [float(np.linalg.norm(poly[i] - poly[(i + 1) % 4])) for i in range(4)]
        min_side, max_side = min(side_lengths), max(side_lengths)
        if min_side < 3:
            return False, "projected_polygon_degenerate", (float(cx), float(cy)), float(area_ratio)
        if max_side / max(min_side, 1e-6) > self.config.max_side_ratio:
            return False, "projected_polygon_too_distorted", (float(cx), float(cy)), float(area_ratio)

        return True, "ok", (float(cx), float(cy)), float(area_ratio)

    def _zone_at_point(self, center_px: Tuple[float, float], allowed_zones: Sequence[str]) -> Optional[str]:
        cx, cy = center_px
        for zone in allowed_zones:
            region = self.regions.get(zone)
            if region is None:
                continue
            x1, y1, x2, y2 = region.bbox_px(self.map_w, self.map_h, pad_ratio=0.0)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone
        return None

    def _nearest_zone(self, center_px: Tuple[float, float], allowed_zones: Sequence[str]) -> Optional[str]:
        cx, cy = center_px
        best_zone = None
        best_dist = 1e18
        for zone in allowed_zones:
            region = self.regions.get(zone)
            if region is None:
                continue
            x1, y1, x2, y2 = region.bbox_px(self.map_w, self.map_h, pad_ratio=0.0)
            zx, zy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = (cx - zx) ** 2 + (cy - zy) ** 2
            if d < best_dist:
                best_dist = d
                best_zone = zone
        return best_zone

    def _apply_previous_prior(self, result: MatchResult, prev_center_px: Optional[Tuple[float, float]]) -> MatchResult:
        if not result.ok or prev_center_px is None or result.center_px is None:
            return result
        dx = result.center_px[0] - prev_center_px[0]
        dy = result.center_px[1] - prev_center_px[1]
        jump_m = pixel_dist_to_meter(dx, dy, self.map_w, self.map_h)
        if jump_m > self.config.soft_jump_m:
            result.score -= (jump_m - self.config.soft_jump_m) * 0.15
        if jump_m > self.config.hard_jump_m:
            result.ok = False
            result.reason = f"jump_too_large_{jump_m:.1f}m"
        return result

    def _match_to_region(
        self,
        frame_gray: np.ndarray,
        frame_kp,
        frame_des,
        region_name: str,
        route_state: RouteState,
        prev_center_px: Optional[Tuple[float, float]],
    ) -> MatchResult:
        data = self.region_features.get(region_name)
        if data is None:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="unknown_region")

        region_kp = data["kp"]
        region_des = data["des"]
        x1, y1, _, _ = data["bbox"]

        if frame_des is None or region_des is None:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="no_descriptor")
        if len(frame_kp) < 4 or len(region_kp) < 4:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="too_few_keypoints")

        good = self.backend.ratio_match(frame_des, region_des)
        if len(good) < self.config.min_good_matches:
            return MatchResult(ok=False, source="feature", zone=region_name, matches=len(good), reason="too_few_good_matches")

        src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_crop = np.float32([region_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
        dst_global = (dst_crop + np.array([x1, y1], dtype=np.float32)).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_global, cv2.RANSAC, self.config.ransac_reproj_thr)
        if H is None or mask is None:
            return MatchResult(ok=False, source="feature", zone=region_name, matches=len(good), reason="homography_failed")
        H = H / H[2, 2]

        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(len(good), 1)

        if inliers < self.config.min_inliers_feature:
            return MatchResult(ok=False, source="feature", zone=region_name, matches=len(good), inliers=inliers, inlier_ratio=inlier_ratio, reason="too_few_inliers")
        if inlier_ratio < self.config.min_inlier_ratio_feature:
            return MatchResult(ok=False, source="feature", zone=region_name, matches=len(good), inliers=inliers, inlier_ratio=inlier_ratio, reason="low_inlier_ratio")

        proj = cv2.perspectiveTransform(src_pts[inlier_mask], H).reshape(-1, 2)
        dst_in = dst_global[inlier_mask].reshape(-1, 2)
        reproj_error = float(np.median(np.linalg.norm(proj - dst_in, axis=1)))
        if reproj_error > self.config.max_reproj_error_feature:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
                reason="high_reprojection_error",
            )

        valid, reason, center_px, area_ratio = self._validate_projected_polygon(H, frame_gray.shape[:2])
        if not valid:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
                area_ratio=area_ratio,
                reason=reason,
            )

        center_m = pixel_to_meter(center_px, self.map_w, self.map_h)
        score = 2.0 * inliers + 70.0 * inlier_ratio - 3.0 * reproj_error
        result = MatchResult(
            ok=True,
            source="feature",
            zone=region_name,
            route_idx=route_state.route_idx_of(region_name),
            H_frame_to_global=H,
            center_px=center_px,
            center_m=center_m,
            matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
            area_ratio=area_ratio,
            score=score,
            reason="ok",
        )
        return self._apply_previous_prior(result, prev_center_px)

    def _estimate_temporal_transform(
        self,
        prev_gray: Optional[np.ndarray],
        curr_gray: np.ndarray,
        prev_H_frame_to_global: Optional[np.ndarray],
        route_state: RouteState,
        allowed_zones: Sequence[str],
        prev_center_px: Optional[Tuple[float, float]],
    ) -> MatchResult:
        if prev_gray is None or prev_H_frame_to_global is None:
            return MatchResult(ok=False, source="temporal", reason="no_previous_transform")
        if prev_gray.shape[:2] != curr_gray.shape[:2]:
            return MatchResult(ok=False, source="temporal", reason="frame_size_changed")

        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=900, qualityLevel=0.01, minDistance=8, blockSize=7)
        if p0 is None or len(p0) < self.config.min_temporal_tracks:
            return MatchResult(ok=False, source="temporal", reason="too_few_tracking_points")

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or st is None:
            return MatchResult(ok=False, source="temporal", reason="optical_flow_failed")

        st_bool = st.reshape(-1).astype(bool)
        prev_pts = p0.reshape(-1, 2)[st_bool]
        curr_pts = p1.reshape(-1, 2)[st_bool]
        if len(curr_pts) < self.config.min_temporal_tracks:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), reason="too_few_tracked_points")

        H_curr_to_prev, mask = cv2.findHomography(curr_pts.reshape(-1, 1, 2), prev_pts.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
        if H_curr_to_prev is None or mask is None:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), reason="temporal_homography_failed")

        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(len(curr_pts), 1)
        if inliers < self.config.min_temporal_tracks:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), inliers=inliers, inlier_ratio=inlier_ratio, reason="too_few_temporal_inliers")
        if inlier_ratio < self.config.min_temporal_inlier_ratio:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), inliers=inliers, inlier_ratio=inlier_ratio, reason="low_temporal_inlier_ratio")

        proj_prev = cv2.perspectiveTransform(curr_pts[inlier_mask].reshape(-1, 1, 2).astype(np.float32), H_curr_to_prev).reshape(-1, 2)
        prev_in = prev_pts[inlier_mask]
        reproj_error = float(np.median(np.linalg.norm(proj_prev - prev_in, axis=1)))
        if reproj_error > self.config.max_temporal_reproj_error:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), inliers=inliers, inlier_ratio=inlier_ratio, reproj_error=reproj_error, reason="high_temporal_reprojection_error")

        H_curr_to_global = prev_H_frame_to_global @ H_curr_to_prev
        H_curr_to_global = H_curr_to_global / H_curr_to_global[2, 2]

        valid, reason, center_px, area_ratio = self._validate_projected_polygon(H_curr_to_global, curr_gray.shape[:2])
        if not valid:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), inliers=inliers, inlier_ratio=inlier_ratio, reproj_error=reproj_error, area_ratio=area_ratio, reason=reason)

        zone = self._zone_at_point(center_px, allowed_zones)
        if zone is None:
            zone = self._nearest_zone(center_px, allowed_zones)
        if zone is None:
            return MatchResult(ok=False, source="temporal", matches=len(curr_pts), inliers=inliers, inlier_ratio=inlier_ratio, reproj_error=reproj_error, area_ratio=area_ratio, reason="temporal_center_not_in_route_candidates")

        center_m = pixel_to_meter(center_px, self.map_w, self.map_h)
        score = 1.5 * inliers + 50.0 * inlier_ratio - 5.0 * reproj_error
        result = MatchResult(
            ok=True,
            source="temporal",
            zone=zone,
            route_idx=route_state.route_idx_of(zone),
            H_frame_to_global=H_curr_to_global,
            center_px=center_px,
            center_m=center_m,
            matches=len(curr_pts),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
            area_ratio=area_ratio,
            score=score,
            reason="ok",
        )
        return self._apply_previous_prior(result, prev_center_px)

    def localize_frame(
        self,
        frame_bgr: np.ndarray,
        route_state: RouteState,
        prev_gray: Optional[np.ndarray] = None,
        prev_H_frame_to_global: Optional[np.ndarray] = None,
        prev_center_px: Optional[Tuple[float, float]] = None,
    ) -> Tuple[MatchResult, np.ndarray]:
        frame_gray = preprocess_for_matching(frame_bgr, self.config.match_preprocess_mode)
        allowed_zones = route_state.candidates()

        frame_kp, frame_des = self.backend.detect(frame_gray)
        feature_results: List[MatchResult] = []
        for zone in allowed_zones:
            res = self._match_to_region(frame_gray, frame_kp, frame_des, zone, route_state, prev_center_px)
            if res.ok:
                feature_results.append(res)

        best_feature = max(feature_results, key=lambda r: r.score) if feature_results else None

        temporal = MatchResult(ok=False, source="temporal", reason="disabled")
        if self.config.use_temporal and not (self.config.skip_temporal_if_feature_ok and best_feature is not None and best_feature.ok):
            temporal = self._estimate_temporal_transform(prev_gray, frame_gray, prev_H_frame_to_global, route_state, allowed_zones, prev_center_px)

        candidates = [r for r in [best_feature, temporal] if r is not None and r.ok]
        if not candidates:
            feature_reason = "no_valid_feature_match" if best_feature is None else best_feature.reason
            return MatchResult(ok=False, source="none", reason=f"feature:{feature_reason}; temporal:{temporal.reason}"), frame_gray

        best = max(candidates, key=lambda r: r.score)
        return best, frame_gray

    def draw_current_result_on_map(self, frame_bgr: np.ndarray, result: MatchResult, title: str = "") -> np.ndarray:
        canvas = self.global_map_bgr.copy()
        if result.ok and result.H_frame_to_global is not None:
            poly = self._project_frame_corners(result.H_frame_to_global, frame_bgr.shape[:2]).astype(np.int32)
            cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=3)
            if result.center_px is not None:
                cx, cy = result.center_px
                cv2.circle(canvas, (int(round(cx)), int(round(cy))), 7, (255, 0, 0), -1)
            text = f"{title} | {result.source} | {result.zone} | score={result.score:.1f}"
        else:
            text = f"{title} | SKIP | {result.reason[:80]}"
        put_label(canvas, text, (24, 38), font_scale=0.8)
        return canvas


# ============================================================
# 6. Result serialization / visualization
# ============================================================

def result_to_row(frame_id: str, result: MatchResult) -> Dict[str, str]:
    """
    Serialize one localization result.

    status meanings:
      - ok    : image-derived localization; source is usually feature or temporal.
      - prior : route-prior fallback only. It gives an approximate center point,
                but has no real homography/inliers/reprojection error. Do not use
                it as a high-confidence defect coordinate transform.
      - skip  : no usable localization.
    """
    if not result.ok or result.center_px is None:
        return {
            "frame_id": frame_id,
            "status": "skip",
            "source": result.source,
            "zone": "",
            "route_idx": "",
            "center_x_px": "",
            "center_y_px": "",
            "center_x_m": "",
            "center_y_m": "",
            "matches": str(result.matches),
            "inliers": str(result.inliers),
            "inlier_ratio": f"{result.inlier_ratio:.4f}",
            "reproj_error": f"{result.reproj_error:.4f}",
            "area_ratio": f"{result.area_ratio:.6f}",
            "score": f"{result.score:.4f}",
            "reason": result.reason,
            "homography": "",
        }

    if result.source == "route_prior":
        return {
            "frame_id": frame_id,
            "status": "prior",
            "source": result.source,
            "zone": result.zone or "",
            "route_idx": "" if result.route_idx is None else str(result.route_idx),
            "center_x_px": f"{result.center_px[0]:.3f}",
            "center_y_px": f"{result.center_px[1]:.3f}",
            "center_x_m": f"{result.center_m[0]:.3f}",
            "center_y_m": f"{result.center_m[1]:.3f}",
            "matches": "0",
            "inliers": "0",
            "inlier_ratio": "",
            "reproj_error": "",
            "area_ratio": "",
            "score": f"{result.score:.4f}",
            "reason": result.reason or "route_prior_fallback",
            "homography": "",
        }

    H_flat = [] if result.H_frame_to_global is None else result.H_frame_to_global.reshape(-1).tolist()
    return {
        "frame_id": frame_id,
        "status": "ok",
        "source": result.source,
        "zone": result.zone or "",
        "route_idx": "" if result.route_idx is None else str(result.route_idx),
        "center_x_px": f"{result.center_px[0]:.3f}",
        "center_y_px": f"{result.center_px[1]:.3f}",
        "center_x_m": f"{result.center_m[0]:.3f}",
        "center_y_m": f"{result.center_m[1]:.3f}",
        "matches": str(result.matches),
        "inliers": str(result.inliers),
        "inlier_ratio": f"{result.inlier_ratio:.4f}",
        "reproj_error": f"{result.reproj_error:.4f}",
        "area_ratio": f"{result.area_ratio:.6f}",
        "score": f"{result.score:.4f}",
        "reason": result.reason,
        "homography": json.dumps(H_flat),
    }


def write_csv(rows: List[Dict[str, str]], out_csv: Path | str) -> None:
    if not rows:
        return
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_regions_on_map(
    global_map: np.ndarray,
    regions: Dict[str, Region],
    out_path: Path | str,
    aruco_report: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    canvas = global_map.copy()
    h, w = canvas.shape[:2]

    # 1) Region overlay
    for name, region in regions.items():
        x1, y1, x2, y2 = region.bbox_px(w, h, pad_ratio=0.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)
        put_label(canvas, name, (x1 + 5, max(y1 + 25, 25)), font_scale=0.55, color=(255, 255, 255), bg_color=(0, 80, 120), thickness=1)

    # 2) ArUco overlay: use the transformed marker coordinates used for rectification.
    status_text = "ArUco: not available"
    if aruco_report is not None:
        status = str(aruco_report.get("status", "unknown"))
        source = str(aruco_report.get("source", "unknown"))
        ids = aruco_report.get("selected_marker_ids_tl_tr_br_bl", [])
        status_text = f"ArUco {status} | src={source} | TL,TR,BR,BL IDs={ids}"

        markers = aruco_report.get("rectified_selected_markers_tl_tr_br_bl", [])
        if isinstance(markers, list):
            for m in markers:
                try:
                    marker_id = int(m.get("id"))
                    role = str(m.get("role", ""))
                    corners = np.asarray(m.get("corners", []), dtype=np.float32).reshape(-1, 2)
                    center = np.asarray(m.get("center", []), dtype=np.float32).reshape(2)
                    if len(corners) == 4:
                        pts = corners.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)
                    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
                    cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), -1)
                    put_label(canvas, f"Aruco {role} ID {marker_id}", (cx + 10, cy + 10), font_scale=0.55, bg_color=(0, 100, 0), thickness=1)
                except Exception:
                    continue

    put_label(canvas, status_text, (20, h - 20), font_scale=0.6, bg_color=(20, 20, 20), thickness=1)
    cv2.imwrite(str(out_path), canvas)
    return canvas


def draw_trajectory(
    localizer: CloseupToGlobalLocalizer,
    records: List[Tuple[str, MatchResult]],
    job_name: str,
    out_path: Path | str,
) -> np.ndarray:
    canvas = localizer.global_map_bgr.copy()
    ok_records = [(fid, r) for fid, r in records if r.ok and r.center_px is not None]

    pts = []
    for _, result in ok_records:
        x, y = result.center_px
        pts.append((int(round(x)), int(round(y))))

    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], isClosed=False, color=(255, 0, 255), thickness=3)

    for i, (frame_id, result) in enumerate(ok_records):
        cx, cy = result.center_px
        center = (int(round(cx)), int(round(cy)))
        # feature/temporal: blue, route-prior fallback: orange.
        dot_color = (255, 0, 0) if result.source != "route_prior" else (0, 140, 255)
        cv2.circle(canvas, center, 4, dot_color, -1)
        if i == 0:
            put_label(canvas, "START", (center[0] + 8, center[1] - 8), font_scale=0.55, bg_color=(0, 80, 0), thickness=1)
        elif i == len(ok_records) - 1:
            put_label(canvas, "END", (center[0] + 8, center[1] - 8), font_scale=0.55, bg_color=(80, 0, 0), thickness=1)

    ok_count = len(ok_records)
    total = len(records)
    success_rate = 0.0 if total == 0 else ok_count / total * 100.0
    prior_count = sum(1 for _, r in ok_records if r.source == "route_prior")
    put_label(canvas, f"{job_name} trajectory | ok={ok_count}/{total} ({success_rate:.1f}%) | route_prior={prior_count}", (24, 38), font_scale=0.8)
    cv2.imwrite(str(out_path), canvas)
    return canvas


def make_video_writer(out_path: Path | str, frame_shape: Tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {out_path}")
    return writer


def resize_for_visualization(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    trusted_rows = [r for r in rows if r["status"] == "ok"]
    prior_rows = [r for r in rows if r["status"] == "prior"]
    trace_rows = trusted_rows + prior_rows

    zones: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for r in trace_rows:
        zones[r["zone"]] = zones.get(r["zone"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    return {
        "total_processed_frames": total,
        # image-derived localization only: feature / temporal
        "localized_frames": len(trusted_rows),
        "success_rate": 0.0 if total == 0 else len(trusted_rows) / total,
        # route_prior-inclusive trace continuity
        "trace_frames": len(trace_rows),
        "trace_rate": 0.0 if total == 0 else len(trace_rows) / total,
        "prior_frames": len(prior_rows),
        "zones": zones,
        "sources": sources,
    }


# ============================================================
# 7. Per-video processing
# ============================================================

def make_localizer_config_for_profile(base: LocalizerConfig, profile: str) -> LocalizerConfig:
    """Return profile-specific localization parameters without mutating the base config."""
    data = asdict(base)
    profile = (profile or "default").lower()

    if profile == "fa":
        # FA objects in the real model can look very different from the printed map.
        # We therefore match more on structural boundaries/edges and use a wider ROI
        # that includes roads and white boundary lines around the facility area.
        data.update(
            {
                "prefer_sift": True,
                "match_preprocess_mode": "hybrid",
                "fa_roi_pad_ratio": 0.48,
                "roi_pad_ratio": 0.28,
                "min_good_matches": 10,
                "min_inliers_feature": 7,
                "min_inlier_ratio_feature": 0.16,
                "max_reproj_error_feature": 10.0,
                "ransac_reproj_thr": 6.5,
                "min_temporal_tracks": 18,
                "min_temporal_inlier_ratio": 0.25,
                "max_temporal_reproj_error": 6.5,
                "skip_temporal_if_feature_ok": True,
                "enable_route_prior_fallback": True,
            }
        )

    return LocalizerConfig(**data)


def region_center_px(region: Region, map_w: int, map_h: int) -> Tuple[float, float]:
    x1, y1, x2, y2 = region.bbox_px(map_w, map_h, pad_ratio=0.0)
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def interpolate_route_center(
    route: Sequence[str],
    regions: Dict[str, Region],
    map_w: int,
    map_h: int,
    progress: float,
) -> Tuple[Tuple[float, float], str, int]:
    """Approximate center point along a known route order. Used only as FA fallback."""
    valid_route = [z for z in route if z in regions]
    if not valid_route:
        return (float(map_w / 2.0), float(map_h / 2.0)), "", 0
    if len(valid_route) == 1:
        return region_center_px(regions[valid_route[0]], map_w, map_h), valid_route[0], 0

    progress = float(np.clip(progress, 0.0, 1.0))
    seg_pos = progress * (len(valid_route) - 1)
    i0 = int(math.floor(seg_pos))
    i1 = min(i0 + 1, len(valid_route) - 1)
    alpha = seg_pos - i0

    p0 = np.array(region_center_px(regions[valid_route[i0]], map_w, map_h), dtype=np.float32)
    p1 = np.array(region_center_px(regions[valid_route[i1]], map_w, map_h), dtype=np.float32)
    p = (1.0 - alpha) * p0 + alpha * p1
    nearest_idx = i0 if alpha < 0.5 else i1
    return (float(p[0]), float(p[1])), valid_route[nearest_idx], nearest_idx


def make_route_prior_result(
    job: VideoJob,
    regions: Dict[str, Region],
    map_w: int,
    map_h: int,
    processed_idx: int,
    total_processed: int,
    original_reason: str,
) -> MatchResult:
    """
    Fill a failed FA frame with an approximate route-based center.

    This is intentionally marked as source='route_prior'. It is useful for stable
    high-level tracing when visual matching fails due to FA appearance mismatch, but
    it should not be treated as a true image-derived homography.
    """
    denom = max(total_processed - 1, 1)
    progress = processed_idx / float(denom)
    center_px, zone, route_idx = interpolate_route_center(job.route, regions, map_w, map_h, progress)
    center_m = pixel_to_meter(center_px, map_w, map_h)
    return MatchResult(
        ok=True,
        source="route_prior",
        zone=zone,
        route_idx=route_idx,
        H_frame_to_global=None,
        center_px=center_px,
        center_m=center_m,
        matches=0,
        inliers=0,
        inlier_ratio=0.0,
        reproj_error=9999.0,
        area_ratio=0.0,
        score=-999.0,
        reason=f"route_prior_fallback_after:{original_reason[:120]}",
    )


def process_one_closeup_video(
    job: VideoJob,
    localizer: CloseupToGlobalLocalizer,
    pipeline_cfg: PipelineConfig,
) -> Dict[str, object]:
    video_path = pipeline_cfg.input_dir / job.video_file
    if not video_path.exists():
        print(f"[VIDEO/SKIP] missing: {video_path}")
        return {"job": job.name, "video": str(video_path), "status": "missing"}

    out_dir = pipeline_cfg.output_dir / job.name
    debug_dir = out_dir / "debug_frames"
    ensure_dir(out_dir)
    ensure_dir(debug_dir)

    route_state = RouteState(
        route=job.route,
        start_idx=job.start_route_idx,
        backtrack=job.backtrack,
        lookahead=job.lookahead,
        initial_wide_search=job.initial_wide_search,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    rows: List[Dict[str, str]] = []
    records: List[Tuple[str, MatchResult]] = []

    prev_gray: Optional[np.ndarray] = None
    prev_H: Optional[np.ndarray] = None
    prev_center_px: Optional[Tuple[float, float]] = None

    writer: Optional[cv2.VideoWriter] = None
    vis_video_path = out_dir / "localization.mp4"

    frame_idx = 0
    processed_idx = 0
    per_video = pipeline_cfg.per_video_frame_stride or {}
    stride = max(1, int(per_video.get(job.name, pipeline_cfg.localize_frame_stride)))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    estimated_processed_total = max(1, int(math.ceil(total_frames / stride))) if total_frames > 0 else 1

    print(
        f"\n[VIDEO] {job.name} | file={job.video_file} | route={job.route} | "
        f"stride={stride} | profile={job.localizer_profile}"
    )
    print(
        f"[ROUTE-GATE] initial_wide_search={job.initial_wide_search} | "
        f"start_candidates={RouteState(job.route, job.start_route_idx, job.backtrack, job.lookahead, job.initial_wide_search).candidates()}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        result, curr_gray = localizer.localize_frame(
            frame_bgr=frame,
            route_state=route_state,
            prev_gray=prev_gray,
            prev_H_frame_to_global=prev_H,
            prev_center_px=prev_center_px,
        )

        if (not result.ok) and localizer.config.enable_route_prior_fallback:
            result = make_route_prior_result(
                job=job,
                regions=localizer.regions,
                map_w=localizer.map_w,
                map_h=localizer.map_h,
                processed_idx=processed_idx,
                total_processed=max(estimated_processed_total, processed_idx + 1),
                original_reason=result.reason,
            )

        frame_id = f"{frame_idx:06d}"
        rows.append(result_to_row(frame_id, result))
        records.append((frame_id, result))

        vis = localizer.draw_current_result_on_map(frame, result, title=f"{job.name} frame={frame_id}")
        vis_small = resize_for_visualization(vis, pipeline_cfg.vis_video_width)
        if writer is None:
            writer = make_video_writer(vis_video_path, vis_small.shape, pipeline_cfg.vis_video_fps)
        writer.write(vis_small)

        if pipeline_cfg.debug_every_n_processed > 0 and processed_idx % pipeline_cfg.debug_every_n_processed == 0:
            cv2.imwrite(str(debug_dir / f"debug_{frame_id}.jpg"), vis)

        if result.ok:
            route_state.update(result.zone)

            if result.source == "route_prior":
                # Route-prior is only an approximate center estimate. It has no real
                # homography, so never use it as the next temporal-tracking reference.
                # Keeping prev_H/prev_gray unchanged prevents fake priors from
                # contaminating later feature/temporal localization.
                prev_center_px = result.center_px
                print(
                    f"[PRIOR] {job.name} frame={frame_id} "
                    f"zone={result.zone} "
                    f"m=({result.center_m[0]:.1f},{result.center_m[1]:.1f}) "
                    f"reason={result.reason[:80]}"
                )
            else:
                prev_gray = curr_gray
                prev_H = result.H_frame_to_global
                prev_center_px = result.center_px
                print(
                    f"[OK] {job.name} frame={frame_id} "
                    f"src={result.source} zone={result.zone} "
                    f"m=({result.center_m[0]:.1f},{result.center_m[1]:.1f}) "
                    f"inliers={result.inliers} err={result.reproj_error:.2f}"
                )
        else:
            print(f"[SKIP] {job.name} frame={frame_id} reason={result.reason}")

        frame_idx += 1
        processed_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    csv_path = out_dir / "localization.csv"
    write_csv(rows, csv_path)
    trajectory_path = out_dir / "trajectory.jpg"
    draw_trajectory(localizer, records, job.name, trajectory_path)

    summary = summarize_rows(rows)
    summary.update(
        {
            "job": job.name,
            "video_file": job.video_file,
            "route": job.route,
            "csv": str(csv_path),
            "trajectory": str(trajectory_path),
            "visualization_video": str(vis_video_path),
            "debug_dir": str(debug_dir),
        }
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[VIDEO/DONE] {job.name} | csv={csv_path} | trajectory={trajectory_path} | video={vis_video_path}")
    return summary


# ============================================================
# 8. HTML / notebook-friendly result viewer
# ============================================================

def create_html_index(output_dir: Path, summaries: List[Dict[str, object]]) -> Path:
    html_path = output_dir / "index.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Airfield Localization Results</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.45} img{max-width:100%;border:1px solid #ddd;margin:8px 0} .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0} code{background:#f6f6f6;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>Airfield Localization Results</h1>",
        "<h2>Global map</h2>",
        "<p><a href='global_map.jpg'>global_map.jpg</a> / <a href='global_map_with_regions.jpg'>global_map_with_regions.jpg</a></p>",
        "<img src='global_map_with_regions.jpg'>",
    ]

    for s in summaries:
        name = s.get("job", "unknown")
        if s.get("status") == "missing":
            parts.append(f"<div class='card'><h2>{name}</h2><p>Missing video: <code>{s.get('video')}</code></p></div>")
            continue
        rel_dir = name
        success = float(s.get("success_rate", 0.0)) * 100.0
        parts.append("<div class='card'>")
        parts.append(f"<h2>{name}</h2>")
        trace = float(s.get("trace_rate", 0.0)) * 100.0
        parts.append(f"<p>Trusted localized: <b>{s.get('localized_frames')}</b> / {s.get('total_processed_frames')} ({success:.1f}%)</p>")
        parts.append(f"<p>Trace including route-prior: <b>{s.get('trace_frames', s.get('localized_frames'))}</b> / {s.get('total_processed_frames')} ({trace:.1f}%), route-prior frames: {s.get('prior_frames', 0)}</p>")
        parts.append(f"<p><a href='{rel_dir}/localization.csv'>CSV</a> | <a href='{rel_dir}/summary.json'>summary.json</a> | <a href='{rel_dir}/localization.mp4'>localization.mp4</a></p>")
        parts.append(f"<img src='{rel_dir}/trajectory.jpg'>")
        parts.append("</div>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def display_results_in_notebook(output_dir: Path | str = "outputs") -> None:
    """Optional helper for Jupyter Notebook."""
    try:
        from IPython.display import HTML, Image, display
    except Exception:
        print("IPython display is not available. Open outputs/index.html manually.")
        return

    output_dir = Path(output_dir)
    index_path = output_dir / "index.html"
    if index_path.exists():
        display(HTML(index_path.read_text(encoding="utf-8")))
    else:
        if (output_dir / "global_map_with_regions.jpg").exists():
            display(Image(filename=str(output_dir / "global_map_with_regions.jpg")))
        for p in sorted(output_dir.glob("*/trajectory.jpg")):
            display(Image(filename=str(p)))


# ============================================================
# 9. Main pipeline
# ============================================================

# Use the folder where this .py file exists, not the current terminal folder.
# This is safer in GitHub Codespaces when you run:
#   python yhj/airfield_global_localization_pipeline.py
SCRIPT_DIR = Path(__file__).resolve().parent

PIPELINE = PipelineConfig(
    input_dir=SCRIPT_DIR,
    output_dir=SCRIPT_DIR / "outputs",
    top_view_video=None,
    localize_frame_stride=5,
    per_video_frame_stride={"TWA": 5, "RW": 5, "TWB": 5},
    debug_every_n_processed=5,
    vis_video_width=1280,
    vis_video_fps=8.0,
    force_rebuild_global_map=True,
)

MAP_BUILDER = MapBuildConfig(
    frame_stride=10,
    max_keyframes=140,
    resize_width=960,
    prefer_sift=False,
    orb_features=5000,
    min_good_matches=45,
    min_inliers=35,
    min_inlier_ratio=0.20,
    max_canvas_side=5200,
    crop_black_border=True,
    use_aruco_rectification=True,
    aruco_dictionary=None,
    # If you know the exact marker IDs, set them in FINAL MAP order: TL, TR, BR, BL.
    # For portrait mapping videos, automatic portrait->landscape ordering usually works.
    aruco_corner_ids=None,
    rectified_map_width=1500,
    rectified_map_height=1200,
    aruco_rotate_portrait_to_landscape=True,
    aruco_portrait_rotation="ccw",
    aruco_best_frame_fallback=True,
    aruco_best_frame_stride=5,
    aruco_debug=True,
)

LOCALIZER = LocalizerConfig(
    prefer_sift=True,
    roi_pad_ratio=0.20,
    fa_roi_pad_ratio=0.40,
    match_preprocess_mode="gray",
    min_good_matches=18,
    min_inliers_feature=12,
    min_inlier_ratio_feature=0.23,
    max_reproj_error_feature=8.0,
    use_temporal=True,
    skip_temporal_if_feature_ok=True,
    enable_route_prior_fallback=False,
)

# Current filming direction: RW-01 -> RW-10. Change to True only if RW was filmed RW-10 -> RW-01.
RW_REVERSE = False


def run_pipeline(
    pipeline_cfg: PipelineConfig = PIPELINE,
    map_cfg: MapBuildConfig = MAP_BUILDER,
    localizer_cfg: LocalizerConfig = LOCALIZER,
    rw_reverse: bool = RW_REVERSE,
) -> List[Dict[str, object]]:
    ensure_dir(pipeline_cfg.output_dir)

    top_view_path = pipeline_cfg.input_dir / pipeline_cfg.top_view_video
    global_map_path = pipeline_cfg.output_dir / pipeline_cfg.global_map_name

    map_report: Optional[Dict[str, object]] = None
    if pipeline_cfg.force_rebuild_global_map or not global_map_path.exists():
        builder = FastTopViewMosaicBuilder(map_cfg)
        global_map, map_report = builder.build(
            video_path=top_view_path,
            output_path=global_map_path,
            debug_dir=pipeline_cfg.output_dir / "map_debug",
        )
    else:
        global_map = cv2.imread(str(global_map_path))
        if global_map is None:
            raise FileNotFoundError(f"Cannot read existing global map: {global_map_path}")
        report_path = pipeline_cfg.output_dir / "global_map_build_report.json"
        if report_path.exists():
            try:
                map_report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                map_report = None
        print(f"[MAP] using existing: {global_map_path}")

    regions = make_default_airfield_regions()
    aruco_report = None if map_report is None else map_report.get("aruco_rectification")
    draw_regions_on_map(global_map, regions, pipeline_cfg.output_dir / "global_map_with_regions.jpg", aruco_report=aruco_report)

    jobs = make_default_video_jobs(rw_reverse=rw_reverse)

    # Build one localizer for the TW/RW default profile. Feature caches are reused across TWA/RW/TWB.
    localizer_cache: Dict[str, CloseupToGlobalLocalizer] = {}
    summaries: List[Dict[str, object]] = []
    for job in jobs:
        profile = job.localizer_profile
        if profile not in localizer_cache:
            cfg = make_localizer_config_for_profile(localizer_cfg, profile)
            localizer_cache[profile] = CloseupToGlobalLocalizer(
                global_map_bgr=global_map,
                regions=regions,
                config=cfg,
            )
        summaries.append(process_one_closeup_video(job, localizer_cache[profile], pipeline_cfg))

    with open(pipeline_cfg.output_dir / "all_video_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    html_path = create_html_index(pipeline_cfg.output_dir, summaries)
    print(f"\n[ALL DONE] Open this file to inspect results: {html_path}")
    return summaries


if __name__ == "__main__":
    run_pipeline()
