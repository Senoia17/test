import argparse
import pickle
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.yolo_device import move_yolo_model

from crater_size import classify_crater_size
from geometry import (
    bbox_center_xyxy,
    bbox_corners_xyxy,
    image_points_to_world,
    polygon_size_mm,
)
from json_writer import save_json
from zone_locator import find_zone, load_zones


CLASS_NAMES = {
    0: "crater",
    1: "missile",
    2: "cluster",
    3: "dumb",
}

UXO_CLASSES = {"missile", "cluster", "dumb"}


def _resolve_default_weights_path():
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from main import load_config
    from mission.model_weights import resolve_model_weight_path

    return resolve_model_weight_path(load_config(project_root / "config.yaml"), "obstacle")


def load_homography(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer(
    weights_path,
    source_path,
    homography_path,
    zones_path,
    output_dir,
    conf=0.3,
    imgsz=1280,
):
    model = YOLO(weights_path)
    device = move_yolo_model(model)
    homography = load_homography(homography_path)
    zones = load_zones(zones_path)

    crater_items = []
    uxo_items = []

    results = model.predict(
        source=source_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = CLASS_NAMES[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            center_px = bbox_center_xyxy(xyxy)
            center_cm = image_points_to_world([center_px], homography)[0]
            zone = find_zone(center_cm, zones)

            if cls_name == "crater":
                corners_px = bbox_corners_xyxy(xyxy)
                corners_cm = image_points_to_world(corners_px, homography)
                width_mm, height_mm = polygon_size_mm(corners_cm)
                size = classify_crater_size(width_mm, height_mm)

                crater_items.append(
                    {
                        "zone": zone,
                        "size": size,
                    }
                )

            elif cls_name in UXO_CLASSES:
                uxo_items.append(
                    {
                        "zone": zone,
                        "type": cls_name,
                    }
                )

    output_dir = Path(output_dir)

    save_json(
        output_dir / "crater_detect.json",
        {"crater_detect": crater_items},
    )

    save_json(
        output_dir / "uxo_detect.json",
        {"UXO_detect": uxo_items},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights")
    parser.add_argument("--source", required=True)
    parser.add_argument("--homography", default="configs/homography.pkl")
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else _resolve_default_weights_path()
    if weights_path is None:
        raise ValueError("Missing models.obstacle.directory/weights in config.yaml")

    infer(
        weights_path=weights_path,
        source_path=args.source,
        homography_path=args.homography,
        zones_path=args.zones,
        output_dir=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()

