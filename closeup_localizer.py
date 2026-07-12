import os
import cv2
import csv
import json
import glob
import math
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable


# ============================================================
# 0. 실제 경기장 크기 설정
# ============================================================

REAL_FIELD_W_M = 3000.0   # 실제 가로 3km
REAL_FIELD_H_M = 2400.0   # 실제 세로 2.4km


# ============================================================
# 1. 데이터 구조
# ============================================================

@dataclass
class Region:
    """
    global map 상의 구역.
    bbox_norm = (x1, y1, x2, y2), 0~1 정규화 좌표
    """
    name: str
    bbox_norm: Tuple[float, float, float, float]

    def bbox_px(self, map_w: int, map_h: int, pad_ratio: float = 0.0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_norm

        px1 = int(x1 * map_w)
        py1 = int(y1 * map_h)
        px2 = int(x2 * map_w)
        py2 = int(y2 * map_h)

        bw = px2 - px1
        bh = py2 - py1

        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        px1 = max(0, px1 - pad_x)
        py1 = max(0, py1 - pad_y)
        px2 = min(map_w, px2 + pad_x)
        py2 = min(map_h, py2 + pad_y)

        return px1, py1, px2, py2


@dataclass
class MatchResult:
    ok: bool
    source: str = "none"          # "feature" or "temporal"
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


@dataclass
class LocalizerConfig:
    # feature matching
    roi_pad_ratio: float = 0.15
    min_good_matches: int = 18
    min_inliers_feature: int = 12
    min_inlier_ratio_feature: float = 0.25
    max_reproj_error_feature: float = 7.0
    ransac_reproj_thr: float = 5.0

    # projected close-up frame polygon validation
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.35
    max_side_ratio: float = 35.0

    # previous transform prior
    soft_jump_m: float = 500.0
    hard_jump_m: float = 1300.0

    # temporal tracking
    use_temporal: bool = True
    min_temporal_tracks: int = 30
    min_temporal_inlier_ratio: float = 0.40
    max_temporal_reproj_error: float = 4.5


# ============================================================
# 2. 지도 구역 / 비행 경로 정의
# ============================================================

def make_default_regions() -> Dict[str, Region]:
    """
    업로드한 레이아웃을 기준으로 한 대략적 구역 분할.

    지도는 가로 500cm, 세로 400cm 구조.
    실제 사용 시 global map crop 상태에 따라 y 범위는 조금 조정 가능.

    대략:
    - 상단 유도로 A구역: TW-A1 ~ TW-A5
    - 중앙 활주로 구역: RW-01 ~ RW-10
    - 하단 유도로 B구역: TW-B1 ~ TW-B5
    """

    regions = {}

    # 구역 간 경계에서 매칭이 끊기지 않도록 약간 overlap
    x_overlap_tw = 0.025
    x_overlap_rw = 0.015

    # y 범위는 지도 crop 상태에 따라 조정 가능
    # global map이 정면 보정된 상태에서 위쪽부터 아래쪽으로:
    tw_a_y = (0.20, 0.42)
    rw_y   = (0.38, 0.62)
    tw_b_y = (0.58, 0.80)

    # TW-A1 ~ TW-A5
    for i in range(5):
        x1 = max(0.0, i / 5.0 - x_overlap_tw)
        x2 = min(1.0, (i + 1) / 5.0 + x_overlap_tw)
        name = f"TW-A{i + 1}"
        regions[name] = Region(name, (x1, tw_a_y[0], x2, tw_a_y[1]))

    # RW-01 ~ RW-10
    for i in range(10):
        x1 = max(0.0, i / 10.0 - x_overlap_rw)
        x2 = min(1.0, (i + 1) / 10.0 + x_overlap_rw)
        name = f"RW-{i + 1:02d}"
        regions[name] = Region(name, (x1, rw_y[0], x2, rw_y[1]))

    # TW-B1 ~ TW-B5
    for i in range(5):
        x1 = max(0.0, i / 5.0 - x_overlap_tw)
        x2 = min(1.0, (i + 1) / 5.0 + x_overlap_tw)
        name = f"TW-B{i + 1}"
        regions[name] = Region(name, (x1, tw_b_y[0], x2, tw_b_y[1]))

    return regions


def make_flight_route() -> List[str]:
    """
    시간대 기준이 아니라 비행 구역 순서 기준.

    요청 반영:
    TW-A1 ~ A5 -> RW-10 ~ RW-01 -> TW-B1 ~ B5

    만약 실제 비행이 RW-01 -> RW-10 방향이면 아래 RW 부분만 뒤집으면 됨.
    """
    route = []

    route += [f"TW-A{i}" for i in range(1, 6)]
    route += [f"RW-{i:02d}" for i in range(10, 0, -1)]
    route += [f"TW-B{i}" for i in range(1, 6)]

    return route


class RouteState:
    """
    현재 비행이 어느 구역 근처에 있는지를 관리.
    첫 프레임에서는 전체 구역을 넓게 탐색하고,
    한 번 매칭이 성공하면 이후에는 현재 구역 주변만 탐색.
    """

    def __init__(
        self,
        route: List[str],
        start_idx: int = 0,
        backtrack: int = 1,
        lookahead: int = 2,
        initial_wide_search: bool = True
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
        if zone is None:
            return
        if zone not in self.zone_to_idx:
            return

        new_idx = self.zone_to_idx[zone]

        # 비행 순서는 기본적으로 앞으로만 진행한다고 가정
        if not self.locked:
            self.idx = new_idx
            self.locked = True
        else:
            self.idx = max(self.idx, new_idx)

    def route_idx_of(self, zone: Optional[str]) -> Optional[int]:
        if zone is None:
            return None
        return self.zone_to_idx.get(zone, None)


# ============================================================
# 3. 기본 유틸
# ============================================================

def enhance_gray(img: np.ndarray) -> np.ndarray:
    """
    조명 변화에 조금 더 강하게 만들기 위한 gray + CLAHE.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    입력 corner를 tl, tr, br, bl 순서로 정렬.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_global_map_from_corners(
    raw_map_bgr: np.ndarray,
    corners_xy: List[Tuple[float, float]],
    out_size: Tuple[int, int] = (1500, 1200)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    종이 사진처럼 지도 영역이 기울어져 있을 때,
    지도 사각형 4개 꼭짓점을 이용해 정면 보정.

    out_size = (width, height)
    1500x1200으로 보정하면 실제 3000m x 2400m이므로 1px = 2m가 됨.
    """
    out_w, out_h = out_size

    src = order_corners_clockwise(np.array(corners_xy, dtype=np.float32))
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(raw_map_bgr, H, (out_w, out_h))

    return rectified, H


def pixel_to_meter(
    xy_px: Tuple[float, float],
    map_w: int,
    map_h: int
) -> Tuple[float, float]:
    x_px, y_px = xy_px
    x_m = x_px / map_w * REAL_FIELD_W_M
    y_m = y_px / map_h * REAL_FIELD_H_M
    return float(x_m), float(y_m)


def pixel_dist_to_meter(
    dx_px: float,
    dy_px: float,
    map_w: int,
    map_h: int
) -> float:
    dx_m = dx_px / map_w * REAL_FIELD_W_M
    dy_m = dy_px / map_h * REAL_FIELD_H_M
    return float(math.sqrt(dx_m * dx_m + dy_m * dy_m))


def transform_points_to_global(
    points_xy: np.ndarray,
    H_frame_to_global: np.ndarray
) -> np.ndarray:
    """
    close-up frame 내의 임의 점들을 global map pixel 좌표로 변환.
    points_xy: shape (N, 2)
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_frame_to_global)
    return out.reshape(-1, 2)


# ============================================================
# 4. Feature detector backend
# ============================================================

class FeatureBackend:
    """
    SIFT 우선 사용.
    환경에 따라 SIFT가 없으면 ORB fallback.
    """

    def __init__(self, prefer_sift: bool = True):
        self.kind = None
        self.detector = None
        self.norm = None
        self.ratio = None

        if prefer_sift and hasattr(cv2, "SIFT_create"):
            self.kind = "SIFT"
            self.detector = cv2.SIFT_create(nfeatures=6000)
            self.norm = cv2.NORM_L2
            self.ratio = 0.75
        else:
            self.kind = "ORB"
            self.detector = cv2.ORB_create(
                nfeatures=8000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                fastThreshold=8
            )
            self.norm = cv2.NORM_HAMMING
            self.ratio = 0.82

        self.matcher = cv2.BFMatcher(self.norm)

    def detect(self, gray: np.ndarray):
        return self.detector.detectAndCompute(gray, None)


# ============================================================
# 5. Main localizer
# ============================================================

class CloseupToGlobalLocalizer:
    def __init__(
        self,
        global_map_bgr: np.ndarray,
        regions: Dict[str, Region],
        route: List[str],
        config: Optional[LocalizerConfig] = None,
        prefer_sift: bool = True,
    ):
        self.global_map_bgr = global_map_bgr
        self.global_map_gray = enhance_gray(global_map_bgr)

        self.map_h, self.map_w = self.global_map_gray.shape[:2]

        self.regions = regions
        self.route = route
        self.config = config or LocalizerConfig()
        self.backend = FeatureBackend(prefer_sift=prefer_sift)

        self.route_idx = {z: i for i, z in enumerate(route)}

        self.region_features = {}
        self._precompute_region_features()

        print(f"[INFO] Feature detector: {self.backend.kind}")
        print(f"[INFO] Global map size: {self.map_w} x {self.map_h}")
        print(f"[INFO] Regions: {len(self.region_features)}")

    def _precompute_region_features(self) -> None:
        """
        global map의 각 구역 ROI feature를 미리 계산.
        close-up frame마다 전체 지도와 매칭하지 않고,
        route candidate 구역과만 매칭하기 위함.
        """
        for name, region in self.regions.items():
            x1, y1, x2, y2 = region.bbox_px(
                self.map_w,
                self.map_h,
                pad_ratio=self.config.roi_pad_ratio
            )

            crop = self.global_map_gray[y1:y2, x1:x2]
            kp, des = self.backend.detect(crop)

            self.region_features[name] = {
                "bbox": (x1, y1, x2, y2),
                "kp": kp,
                "des": des,
            }

    def _project_frame_corners(
        self,
        H_frame_to_global: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        h, w = frame_shape[:2]

        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32).reshape(-1, 1, 2)

        projected = cv2.perspectiveTransform(corners, H_frame_to_global)
        return projected.reshape(-1, 2)

    def _validate_projected_polygon(
        self,
        H_frame_to_global: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[bool, str, Tuple[float, float], float]:
        """
        Homography가 말이 되는지 검사.
        - close-up frame을 global map 위로 투영했을 때 지도 밖으로 심하게 나가지 않는가
        - 면적이 너무 작거나 너무 크지 않은가
        - 사각형이 지나치게 찌그러지지 않았는가
        """
        poly = self._project_frame_corners(H_frame_to_global, frame_shape)

        center_frame = np.array([[
            [frame_shape[1] / 2.0, frame_shape[0] / 2.0]
        ]], dtype=np.float32)

        center_global = cv2.perspectiveTransform(center_frame, H_frame_to_global)
        cx, cy = center_global.reshape(-1, 2)[0]

        margin = max(self.map_w, self.map_h) * 0.05

        inside_poly = (
            np.all(poly[:, 0] >= -margin) and
            np.all(poly[:, 0] <= self.map_w + margin) and
            np.all(poly[:, 1] >= -margin) and
            np.all(poly[:, 1] <= self.map_h + margin)
        )

        inside_center = (
            0 <= cx < self.map_w and
            0 <= cy < self.map_h
        )

        if not inside_center:
            return False, "center_out_of_map", (float(cx), float(cy)), 0.0

        if not inside_poly:
            return False, "projected_polygon_out_of_map", (float(cx), float(cy)), 0.0

        area = abs(cv2.contourArea(poly.astype(np.float32)))
        map_area = float(self.map_w * self.map_h)
        area_ratio = area / map_area

        if area_ratio < self.config.min_area_ratio:
            return False, "projected_area_too_small", (float(cx), float(cy)), area_ratio

        if area_ratio > self.config.max_area_ratio:
            return False, "projected_area_too_large", (float(cx), float(cy)), area_ratio

        side_lengths = []
        for i in range(4):
            p1 = poly[i]
            p2 = poly[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p1 - p2))

        min_side = min(side_lengths)
        max_side = max(side_lengths)

        if min_side < 3:
            return False, "projected_polygon_degenerate", (float(cx), float(cy)), area_ratio

        if max_side / max(min_side, 1e-6) > self.config.max_side_ratio:
            return False, "projected_polygon_too_distorted", (float(cx), float(cy)), area_ratio

        return True, "ok", (float(cx), float(cy)), float(area_ratio)

    def _zone_at_point(self, center_px: Tuple[float, float]) -> Optional[str]:
        """
        center point가 어느 route 구역에 들어가는지 찾음.
        """
        cx, cy = center_px

        for zone in self.route:
            if zone not in self.regions:
                continue

            x1, y1, x2, y2 = self.regions[zone].bbox_px(
                self.map_w,
                self.map_h,
                pad_ratio=0.0
            )

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone

        return None

    def _nearest_zone(
        self,
        center_px: Tuple[float, float],
        allowed_zones: List[str]
    ) -> Optional[str]:
        """
        center가 bbox 경계에 애매하게 걸린 경우,
        허용된 candidate 중 가장 가까운 zone을 선택.
        """
        cx, cy = center_px
        best_zone = None
        best_dist = 1e18

        for zone in allowed_zones:
            if zone not in self.regions:
                continue

            x1, y1, x2, y2 = self.regions[zone].bbox_px(
                self.map_w,
                self.map_h,
                pad_ratio=0.0
            )

            zx = (x1 + x2) / 2.0
            zy = (y1 + y2) / 2.0
            d = (cx - zx) ** 2 + (cy - zy) ** 2

            if d < best_dist:
                best_dist = d
                best_zone = zone

        return best_zone

    def _apply_previous_prior(
        self,
        result: MatchResult,
        prev_center_px: Optional[Tuple[float, float]]
    ) -> MatchResult:
        """
        직전 성공 프레임의 위치와 너무 멀리 튀는 결과는 감점 또는 reject.
        """
        if not result.ok or prev_center_px is None or result.center_px is None:
            return result

        dx = result.center_px[0] - prev_center_px[0]
        dy = result.center_px[1] - prev_center_px[1]
        jump_m = pixel_dist_to_meter(dx, dy, self.map_w, self.map_h)

        if jump_m > self.config.soft_jump_m:
            result.score -= (jump_m - self.config.soft_jump_m) * 0.15

        if jump_m > self.config.hard_jump_m:
            result.ok = False
            result.reason = f"jump_too_large_from_previous_transform_{jump_m:.1f}m"

        return result

    def _match_to_region(
        self,
        frame_gray: np.ndarray,
        frame_kp,
        frame_des,
        region_name: str,
        route_state: RouteState,
        prev_center_px: Optional[Tuple[float, float]]
    ) -> MatchResult:
        if region_name not in self.region_features:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="unknown_region")

        data = self.region_features[region_name]
        region_kp = data["kp"]
        region_des = data["des"]
        x1, y1, x2, y2 = data["bbox"]

        if frame_des is None or region_des is None:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="no_descriptor")

        if len(frame_kp) < 4 or len(region_kp) < 4:
            return MatchResult(ok=False, source="feature", zone=region_name, reason="too_few_keypoints")

        raw_matches = self.backend.matcher.knnMatch(frame_des, region_des, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.backend.ratio * n.distance:
                good.append(m)

        if len(good) < self.config.min_good_matches:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                reason="too_few_good_matches"
            )

        src_pts = np.float32([
            frame_kp[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        dst_pts_crop = np.float32([
            region_kp[m.trainIdx].pt for m in good
        ]).reshape(-1, 2)

        # ROI crop 좌표를 global map 좌표로 변환
        dst_pts_global = dst_pts_crop + np.array([x1, y1], dtype=np.float32)
        dst_pts_global = dst_pts_global.reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts_global,
            cv2.RANSAC,
            self.config.ransac_reproj_thr
        )

        if H is None or mask is None:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                reason="homography_failed"
            )

        H = H / H[2, 2]

        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(len(good), 1)

        if inliers < self.config.min_inliers_feature:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reason="too_few_inliers"
            )

        if inlier_ratio < self.config.min_inlier_ratio_feature:
            return MatchResult(
                ok=False,
                source="feature",
                zone=region_name,
                matches=len(good),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reason="low_inlier_ratio"
            )

        src_in = src_pts[inlier_mask]
        dst_in = dst_pts_global[inlier_mask].reshape(-1, 2)
        proj = cv2.perspectiveTransform(src_in, H).reshape(-1, 2)

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
                reason="high_reprojection_error"
            )

        valid, reason, center_px, area_ratio = self._validate_projected_polygon(
            H,
            frame_gray.shape[:2]
        )

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
                reason=reason
            )

        center_m = pixel_to_meter(center_px, self.map_w, self.map_h)
        route_idx = route_state.route_idx_of(region_name)

        score = (
            2.0 * inliers +
            70.0 * inlier_ratio -
            3.0 * reproj_error
        )

        result = MatchResult(
            ok=True,
            source="feature",
            zone=region_name,
            route_idx=route_idx,
            H_frame_to_global=H,
            center_px=center_px,
            center_m=center_m,
            matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
            area_ratio=area_ratio,
            score=score,
            reason="ok"
        )

        result = self._apply_previous_prior(result, prev_center_px)
        return result

    def _estimate_temporal_transform(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_H_frame_to_global: np.ndarray,
        route_state: RouteState,
        allowed_zones: List[str],
        prev_center_px: Optional[Tuple[float, float]]
    ) -> MatchResult:
        """
        직전 성공 프레임의 transform을 참조하는 부분.

        prev_H_frame_to_global:
            직전 close-up frame 좌표 -> global map 좌표

        Optical flow로 현재 frame -> 직전 frame 변환을 구한 뒤,
            현재 frame -> 직전 frame -> global map
        순서로 합성한다.
        """
        if prev_gray is None or prev_H_frame_to_global is None:
            return MatchResult(ok=False, source="temporal", reason="no_previous_transform")

        if prev_gray.shape[:2] != curr_gray.shape[:2]:
            return MatchResult(ok=False, source="temporal", reason="frame_size_changed")

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=900,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

        if p0 is None or len(p0) < self.config.min_temporal_tracks:
            return MatchResult(ok=False, source="temporal", reason="too_few_tracking_points")

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if p1 is None or st is None:
            return MatchResult(ok=False, source="temporal", reason="optical_flow_failed")

        st = st.reshape(-1).astype(bool)

        prev_pts = p0.reshape(-1, 2)[st]
        curr_pts = p1.reshape(-1, 2)[st]

        if len(curr_pts) < self.config.min_temporal_tracks:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                reason="too_few_tracked_points"
            )

        # 현재 frame 좌표 -> 직전 frame 좌표
        H_curr_to_prev, mask = cv2.findHomography(
            curr_pts.reshape(-1, 1, 2),
            prev_pts.reshape(-1, 1, 2),
            cv2.RANSAC,
            3.0
        )

        if H_curr_to_prev is None or mask is None:
            return MatchResult(ok=False, source="temporal", reason="temporal_homography_failed")

        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(len(curr_pts), 1)

        if inliers < self.config.min_temporal_tracks:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reason="too_few_temporal_inliers"
            )

        if inlier_ratio < self.config.min_temporal_inlier_ratio:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reason="low_temporal_inlier_ratio"
            )

        curr_in = curr_pts[inlier_mask].reshape(-1, 1, 2)
        prev_in = prev_pts[inlier_mask].reshape(-1, 2)

        proj_prev = cv2.perspectiveTransform(curr_in.astype(np.float32), H_curr_to_prev)
        proj_prev = proj_prev.reshape(-1, 2)

        reproj_error = float(np.median(np.linalg.norm(proj_prev - prev_in, axis=1)))

        if reproj_error > self.config.max_temporal_reproj_error:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
                reason="high_temporal_reprojection_error"
            )

        # 현재 frame -> global map
        H_curr_to_global = prev_H_frame_to_global @ H_curr_to_prev
        H_curr_to_global = H_curr_to_global / H_curr_to_global[2, 2]

        valid, reason, center_px, area_ratio = self._validate_projected_polygon(
            H_curr_to_global,
            curr_gray.shape[:2]
        )

        if not valid:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
                area_ratio=area_ratio,
                reason=reason
            )

        zone = self._zone_at_point(center_px)
        if zone is None or zone not in allowed_zones:
            zone = self._nearest_zone(center_px, allowed_zones)

        if zone is None:
            return MatchResult(
                ok=False,
                source="temporal",
                matches=len(curr_pts),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
                area_ratio=area_ratio,
                reason="temporal_center_not_in_route_candidates"
            )

        center_m = pixel_to_meter(center_px, self.map_w, self.map_h)
        route_idx = route_state.route_idx_of(zone)

        score = (
            1.5 * inliers +
            50.0 * inlier_ratio -
            5.0 * reproj_error
        )

        result = MatchResult(
            ok=True,
            source="temporal",
            zone=zone,
            route_idx=route_idx,
            H_frame_to_global=H_curr_to_global,
            center_px=center_px,
            center_m=center_m,
            matches=len(curr_pts),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
            area_ratio=area_ratio,
            score=score,
            reason="ok"
        )

        result = self._apply_previous_prior(result, prev_center_px)
        return result

    def localize_frame(
        self,
        frame_bgr: np.ndarray,
        route_state: RouteState,
        prev_gray: Optional[np.ndarray] = None,
        prev_H_frame_to_global: Optional[np.ndarray] = None,
        prev_center_px: Optional[Tuple[float, float]] = None,
    ) -> Tuple[MatchResult, np.ndarray]:
        """
        close-up frame 하나를 global map 좌표로 변환.
        """
        frame_gray = enhance_gray(frame_bgr)

        allowed_zones = route_state.candidates()

        frame_kp, frame_des = self.backend.detect(frame_gray)

        feature_results = []
        for zone in allowed_zones:
            res = self._match_to_region(
                frame_gray=frame_gray,
                frame_kp=frame_kp,
                frame_des=frame_des,
                region_name=zone,
                route_state=route_state,
                prev_center_px=prev_center_px
            )
            if res.ok:
                feature_results.append(res)

        best_feature = None
        if feature_results:
            best_feature = max(feature_results, key=lambda r: r.score)

        temporal_result = None
        if self.config.use_temporal:
            temporal_result = self._estimate_temporal_transform(
                prev_gray=prev_gray,
                curr_gray=frame_gray,
                prev_H_frame_to_global=prev_H_frame_to_global,
                route_state=route_state,
                allowed_zones=allowed_zones,
                prev_center_px=prev_center_px
            )

        candidates = []

        if best_feature is not None and best_feature.ok:
            candidates.append(best_feature)

        if temporal_result is not None and temporal_result.ok:
            candidates.append(temporal_result)

        if not candidates:
            reasons = []
            if best_feature is None:
                reasons.append("no_valid_feature_match")
            if temporal_result is not None:
                reasons.append(f"temporal:{temporal_result.reason}")

            return MatchResult(
                ok=False,
                source="none",
                reason=";".join(reasons)
            ), frame_gray

        best = max(candidates, key=lambda r: r.score)

        return best, frame_gray

    def draw_debug_on_map(
        self,
        frame_bgr: np.ndarray,
        result: MatchResult,
        out_path: str
    ) -> None:
        """
        global map 위에 현재 close-up frame이 매칭된 영역을 시각화.
        """
        canvas = self.global_map_bgr.copy()

        if result.ok and result.H_frame_to_global is not None:
            poly = self._project_frame_corners(
                result.H_frame_to_global,
                frame_bgr.shape[:2]
            ).astype(np.int32)

            cv2.polylines(
                canvas,
                [poly.reshape(-1, 1, 2)],
                isClosed=True,
                color=(0, 0, 255),
                thickness=3
            )

            if result.center_px is not None:
                cx, cy = result.center_px
                cv2.circle(canvas, (int(cx), int(cy)), 7, (255, 0, 0), -1)

            label = f"{result.source} | {result.zone} | score={result.score:.1f}"
            cv2.putText(
                canvas,
                label,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(out_path, canvas)


# ============================================================
# 6. 결과 저장
# ============================================================

def result_to_row(frame_id: str, result: MatchResult) -> Dict[str, str]:
    if not result.ok:
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

    H_flat = result.H_frame_to_global.reshape(-1).tolist()

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


def write_csv(rows: List[Dict[str, str]], out_csv: str) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 7. 비디오 / 프레임 폴더 처리
# ============================================================

def process_video(
    video_path: str,
    global_map_path: str,
    out_csv: str,
    debug_dir: Optional[str] = None,
    map_corners: Optional[List[Tuple[float, float]]] = None,
    rectified_map_size: Tuple[int, int] = (1500, 1200),
    start_route_idx: int = 0,
    frame_stride: int = 1,
) -> None:
    raw_map = cv2.imread(global_map_path)
    if raw_map is None:
        raise FileNotFoundError(f"Cannot read global map: {global_map_path}")

    if map_corners is not None:
        global_map, _ = rectify_global_map_from_corners(
            raw_map,
            map_corners,
            out_size=rectified_map_size
        )
    else:
        global_map = raw_map

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "rectified_or_loaded_global_map.jpg"), global_map)

    regions = make_default_regions()
    route = make_flight_route()

    config = LocalizerConfig()
    localizer = CloseupToGlobalLocalizer(
        global_map_bgr=global_map,
        regions=regions,
        route=route,
        config=config,
        prefer_sift=True
    )

    route_state = RouteState(
        route=route,
        start_idx=start_route_idx,
        backtrack=1,
        lookahead=2,
        initial_wide_search=True
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    rows = []

    prev_gray = None
    prev_H = None
    prev_center_px = None

    frame_idx = 0
    used_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        result, curr_gray = localizer.localize_frame(
            frame_bgr=frame,
            route_state=route_state,
            prev_gray=prev_gray,
            prev_H_frame_to_global=prev_H,
            prev_center_px=prev_center_px
        )

        frame_id = f"{frame_idx:06d}"
        rows.append(result_to_row(frame_id, result))

        if debug_dir is not None:
            debug_path = os.path.join(debug_dir, f"debug_{frame_id}.jpg")
            localizer.draw_debug_on_map(frame, result, debug_path)

        if result.ok:
            route_state.update(result.zone)

            # 직전 transform 업데이트는 성공 프레임 기준으로만 한다.
            prev_gray = curr_gray
            prev_H = result.H_frame_to_global
            prev_center_px = result.center_px

            print(
                f"[OK] frame={frame_id} "
                f"source={result.source} "
                f"zone={result.zone} "
                f"meter=({result.center_m[0]:.1f}, {result.center_m[1]:.1f}) "
                f"inliers={result.inliers} "
                f"err={result.reproj_error:.2f}"
            )
        else:
            # 불안정한 프레임은 skip.
            # prev_H, prev_gray는 업데이트하지 않음.
            print(f"[SKIP] frame={frame_id} reason={result.reason}")

        frame_idx += 1
        used_idx += 1

    cap.release()
    write_csv(rows, out_csv)

    print(f"[DONE] saved: {out_csv}")


def process_frame_folder(
    frame_dir: str,
    global_map_path: str,
    out_csv: str,
    debug_dir: Optional[str] = None,
    map_corners: Optional[List[Tuple[float, float]]] = None,
    rectified_map_size: Tuple[int, int] = (1500, 1200),
    start_route_idx: int = 0,
) -> None:
    raw_map = cv2.imread(global_map_path)
    if raw_map is None:
        raise FileNotFoundError(f"Cannot read global map: {global_map_path}")

    if map_corners is not None:
        global_map, _ = rectify_global_map_from_corners(
            raw_map,
            map_corners,
            out_size=rectified_map_size
        )
    else:
        global_map = raw_map

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "rectified_or_loaded_global_map.jpg"), global_map)

    regions = make_default_regions()
    route = make_flight_route()
    config = LocalizerConfig()

    localizer = CloseupToGlobalLocalizer(
        global_map_bgr=global_map,
        regions=regions,
        route=route,
        config=config,
        prefer_sift=True
    )

    route_state = RouteState(
        route=route,
        start_idx=start_route_idx,
        backtrack=1,
        lookahead=2,
        initial_wide_search=True
    )

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(frame_dir, ext)))

    image_paths = sorted(image_paths)

    rows = []

    prev_gray = None
    prev_H = None
    prev_center_px = None

    for i, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue

        frame_id = os.path.splitext(os.path.basename(path))[0]

        result, curr_gray = localizer.localize_frame(
            frame_bgr=frame,
            route_state=route_state,
            prev_gray=prev_gray,
            prev_H_frame_to_global=prev_H,
            prev_center_px=prev_center_px
        )

        rows.append(result_to_row(frame_id, result))

        if debug_dir is not None:
            debug_path = os.path.join(debug_dir, f"debug_{frame_id}.jpg")
            localizer.draw_debug_on_map(frame, result, debug_path)

        if result.ok:
            route_state.update(result.zone)

            prev_gray = curr_gray
            prev_H = result.H_frame_to_global
            prev_center_px = result.center_px

            print(
                f"[OK] {frame_id} "
                f"source={result.source} "
                f"zone={result.zone} "
                f"meter=({result.center_m[0]:.1f}, {result.center_m[1]:.1f})"
            )
        else:
            print(f"[SKIP] {frame_id} reason={result.reason}")

    write_csv(rows, out_csv)
    print(f"[DONE] saved: {out_csv}")


# ============================================================
# 8. 실행 예시
# ============================================================

if __name__ == "__main__":
    GLOBAL_MAP_PATH = "./global_map.jpg"
    CLOSEUP_VIDEO_PATH = "./closeup_video.mp4"

    OUT_CSV = "./closeup_global_coords.csv"
    DEBUG_DIR = "./debug_matching"

    # 이미 global map이 지도 부분만 정면 보정되어 있으면 None.
    MAP_CORNERS = None

    # 만약 지금 업로드한 사진처럼 종이 사진에서 지도 영역만 보정해야 한다면,
    # 지도 사각형의 꼭짓점 4개를 이미지 pixel 좌표로 입력.
    #
    # 순서는 아무렇게나 넣어도 order_corners_clockwise()가 정렬하지만,
    # 가능하면 좌상, 우상, 우하, 좌하 순서 권장.
    #
    # 예시:
    # MAP_CORNERS = [
    #     (x_top_left, y_top_left),
    #     (x_top_right, y_top_right),
    #     (x_bottom_right, y_bottom_right),
    #     (x_bottom_left, y_bottom_left),
    # ]

    process_video(
        video_path=CLOSEUP_VIDEO_PATH,
        global_map_path=GLOBAL_MAP_PATH,
        out_csv=OUT_CSV,
        debug_dir=DEBUG_DIR,
        map_corners=MAP_CORNERS,
        rectified_map_size=(1500, 1200),
        start_route_idx=0,
        frame_stride=1
    )

    # 프레임 폴더를 처리하고 싶으면 process_frame_folder 사용.
    #
    # process_frame_folder(
    #     frame_dir="./closeup_frames",
    #     global_map_path=GLOBAL_MAP_PATH,
    #     out_csv="./closeup_global_coords.csv",
    #     debug_dir="./debug_matching",
    #     map_corners=MAP_CORNERS,
    #     rectified_map_size=(1500, 1200),
    #     start_route_idx=0
    # )