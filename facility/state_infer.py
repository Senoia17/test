"""Facility state inference and video aggregation utilities."""

from detection.classifier import YoloClassifier


# =========================
# 1. 시설물 정보 하드코딩
# =========================
# 실제 설명자료 기준으로 type만 수정하면 됨.

FACILITY_INFO = {
    "FA-01": {"type": "control_tower"},
    "FA-02": {"type": "radar"},
    "FA-03": {"type": "hangar"},
    "FA-04": {"type": "building"},
    "FA-05": {"type": "bunker"},
    "FA-06": {"type": "warehouse"},
}


# =========================
# 2. 화재 감지 rule
# =========================

def detect_fire_with_contour(
    bgr_crop,
    min_fire_area_ratio=0.003,
    min_fire_pixel_ratio=0.006,
):
    """
    FA crop에서 화재 색상 영역을 감지한다.

    반환:
    {
        "is_fire": bool,
        "fire_score": float,
        "fire_pixel_ratio": float,
        "max_contour_ratio": float
    }
    """
    import cv2
    import numpy as np

    if bgr_crop is None or bgr_crop.size == 0:
        return {
            "is_fire": False,
            "fire_score": 0.0,
            "fire_pixel_ratio": 0.0,
            "max_contour_ratio": 0.0,
        }

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    # red range 1
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([15, 255, 255])

    # red range 2
    lower_red2 = np.array([165, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    # orange / yellow
    lower_orange = np.array([15, 80, 100])
    upper_orange = np.array([45, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    fire_mask = mask_red1 | mask_red2 | mask_orange

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    h, w = fire_mask.shape[:2]
    crop_area = h * w

    fire_pixel_ratio = float(np.count_nonzero(fire_mask)) / max(crop_area, 1)

    contours, _ = cv2.findContours(
        fire_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        max_area = max(max_area, area)

    max_contour_ratio = max_area / max(crop_area, 1)

    # contour와 전체 pixel ratio를 같이 사용
    is_fire = (
        max_contour_ratio >= min_fire_area_ratio
        and fire_pixel_ratio >= min_fire_pixel_ratio
    )

    # 점수는 경험적 값. 실제 영상으로 threshold 조정 필요.
    fire_score = min(
        0.99,
        0.5 + max_contour_ratio * 80 + fire_pixel_ratio * 20
    ) if is_fire else max_contour_ratio * 50 + fire_pixel_ratio * 10

    return {
        "is_fire": bool(is_fire),
        "fire_score": float(fire_score),
        "fire_pixel_ratio": float(fire_pixel_ratio),
        "max_contour_ratio": float(max_contour_ratio),
    }


# =========================
# 3. YOLO-cls 기반 normal/damaged classifier
# =========================

class FacilityStateClassifier:
    def __init__(
        self,
        damage_model_path,
        imgsz=320,
        device=None,
        damaged_threshold=0.65,
    ):
        """
        damage_model_path:
            configured Facility YOLO-CLS weight path

        damaged_threshold:
            damaged confidence가 이 값보다 낮으면 애매한 damaged로 처리.
        """

        self.classifier = YoloClassifier(
            damage_model_path,
            imgsz=imgsz,
            device=device,
        )
        self.damaged_threshold = damaged_threshold

    def classify_damage(self, bgr_crop):
        """
        화재가 아니라고 판단된 crop에 대해 normal/damaged 분류.

        반환:
        {
            "label": "normal" or "damaged",
            "confidence": float,
            "raw_probs": {...}
        }
        """

        return self.classifier.classify(bgr_crop)

    def analyze_crop(self, bgr_crop, fa_id=None):
        """
        최종 FA crop 분석:
        1. 화재 rule
        2. 화재 아니면 YOLO-cls normal/damaged

        반환:
        {
            "fa_id": ...,
            "facility_type": ...,
            "status": "fire" / "damaged" / "normal",
            "confidence": ...,
            "method": ...,
            ...
        }
        """

        fire = detect_fire_with_contour(bgr_crop)

        facility_type = None
        if fa_id in FACILITY_INFO:
            facility_type = FACILITY_INFO[fa_id]["type"]

        if fire["is_fire"]:
            return {
                "fa_id": fa_id,
                "facility_type": facility_type,
                "status": "fire",
                "confidence": fire["fire_score"],
                "method": "hsv_fire_rule",
                "fire_info": fire,
            }

        cls = self.classify_damage(bgr_crop)

        label = cls["label"]
        conf = cls["confidence"]

        # damaged가 낮은 confidence로 나온 경우는 불확실 처리 가능
        # 여기서는 status는 유지하되 confidence만 그대로 반환.
        if label == "damaged" and conf < self.damaged_threshold:
            method = "damage_yolo_cls_low_conf"
        else:
            method = "damage_yolo_cls"

        return {
            "fa_id": fa_id,
            "facility_type": facility_type,
            "status": label,
            "confidence": conf,
            "method": method,
            "raw_probs": cls["raw_probs"],
            "fire_info": fire,
        }


# =========================
# 4. 여러 프레임 결과 통합
# =========================

def fuse_facility_results(
    results,
    fire_min_count=2,
    fire_high_conf=0.80,
    damaged_min_conf=0.65,
):
    """
    같은 FA에 대해 여러 frame crop 결과를 통합한다.

    fire:
        - 2프레임 이상 fire면 fire
        - 또는 fire confidence가 매우 높으면 fire

    damaged:
        - fire가 아니고 damaged 점수가 normal보다 충분히 높으면 damaged

    normal:
        - 나머지는 normal
    """

    if not results:
        return {
            "status": "unknown",
            "confidence": 0.0,
            "reason": "no_results",
        }

    scores = {
        "fire": 0.0,
        "damaged": 0.0,
        "normal": 0.0,
    }

    counts = {
        "fire": 0,
        "damaged": 0,
        "normal": 0,
    }

    max_conf = {
        "fire": 0.0,
        "damaged": 0.0,
        "normal": 0.0,
    }

    for r in results:
        status = r["status"]
        conf = float(r["confidence"])

        if status not in scores:
            continue

        counts[status] += 1
        max_conf[status] = max(max_conf[status], conf)

        # 화재는 누락 방지 목적상 가중치 부여
        if status == "fire":
            scores[status] += conf * 1.35

        # damaged는 낮은 confidence면 약하게 반영
        elif status == "damaged":
            if conf >= damaged_min_conf:
                scores[status] += conf
            else:
                scores[status] += conf * 0.4

        else:
            scores[status] += conf

    # fire 우선 처리
    if counts["fire"] >= fire_min_count or max_conf["fire"] >= fire_high_conf:
        total = sum(scores.values()) + 1e-6
        return {
            "status": "fire",
            "confidence": min(0.99, scores["fire"] / total + 0.15),
            "scores": scores,
            "counts": counts,
            "reason": "fire_priority",
        }

    # 일반 점수 기반
    final_status = max(scores, key=scores.get)
    total = sum(scores.values()) + 1e-6
    final_conf = scores[final_status] / total

    return {
        "status": final_status,
        "confidence": float(final_conf),
        "scores": scores,
        "counts": counts,
        "reason": "score_voting",
    }


# =========================
# 5. 단일 이미지 테스트용
# =========================

def analyze_single_crop(
    image_path,
    model_path,
    fa_id=None,
    save_debug=False,
):
    import cv2

    crop = cv2.imread(image_path)

    if crop is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    analyzer = FacilityStateClassifier(model_path)
    result = analyzer.analyze_crop(crop, fa_id=fa_id)

    print(result)

    if save_debug:
        label = result["status"]
        conf = result["confidence"]

        vis = crop.copy()
        cv2.putText(
            vis,
            f"{fa_id} {label} {conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite("debug_facility_state.jpg", vis)

    return result


def analyze_facility_video(
    video_path,
    model_path,
    camera_model=None,
    sample_interval=30,
    fa_id=None,
):
    """Analyze sampled frames from one facility video and aggregate results.

    Policy: one video represents one facility. Fire in any sampled frame wins;
    otherwise status is majority vote and confidence is the average confidence of
    sampled frame results. Existing single-crop APIs remain unchanged.
    """
    from utils.video import iter_video_frames
    from calibration.undistort import undistort_frame

    analyzer = FacilityStateClassifier(model_path)
    frame_results = []

    for frame_index, frame in iter_video_frames(video_path, sample_interval=sample_interval):
        frame = undistort_frame(frame, camera_model)
        result = analyzer.analyze_crop(frame, fa_id=fa_id)
        result["frame_index"] = frame_index
        frame_results.append(result)

    if not frame_results:
        raise ValueError(f"No frames sampled from facility video: {video_path}")

    fire_results = [result for result in frame_results if result.get("status") == "fire"]
    if fire_results:
        confidence = sum(float(result.get("confidence", 0.0)) for result in fire_results) / len(fire_results)
        status = "fire"
        method = "video_fire_any_frame"
    else:
        counts = {}
        for result in frame_results:
            status_key = result.get("status", "unknown")
            counts[status_key] = counts.get(status_key, 0) + 1
        status = max(counts, key=counts.get)
        status_results = [result for result in frame_results if result.get("status") == status]
        confidence = sum(float(result.get("confidence", 0.0)) for result in status_results) / len(status_results)
        method = "video_majority_vote"

    return {
        "fa_id": fa_id,
        "status": status,
        "confidence": float(confidence),
        "method": method,
        "sampled_frames": len(frame_results),
        "frame_results": frame_results,
    }


if __name__ == "__main__":
    from pathlib import Path

    from main import load_config
    from mission.model_weights import resolve_model_weight_path

    CONFIG_PATH = Path("config.yaml")
    IMAGE_PATH = "FA-02_crop.jpg"
    MODEL_PATH = resolve_model_weight_path(load_config(CONFIG_PATH), "facility")
    if MODEL_PATH is None:
        raise ValueError("Missing models.facility.directory/version in config.yaml")

    analyze_single_crop(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        fa_id="FA-02",
        save_debug=True,
    )
