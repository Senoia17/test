"""Train the facility damage classification model."""

from pathlib import Path

import torch
from ultralytics import YOLO


# =========================
# 1. 설정
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
FACILITY_DIR = Path(__file__).resolve().parent

DATA_DIR = FACILITY_DIR / "dataset" / "facility_damage"

PROJECT_DIR = REPO_ROOT / "runs" / "facility"
RUN_NAME = "facility_damage_cls"

IMG_SIZE = 320
EPOCHS = 60
BATCH = 32
PATIENCE = 15
WORKERS = 4


# =========================
# 2. config 기반 YOLO-cls pretrained weight 선택
# =========================

def find_yolo_cls_weight():
    """Return the configured Facility YOLO-CLS pretrained checkpoint."""
    from main import load_config
    from mission.model_weights import resolve_pretrained_model_path

    weight_path = resolve_pretrained_model_path(load_config(REPO_ROOT / "config.yaml"), "facility")
    if weight_path is None:
        raise ValueError("Missing models.facility.pretrained in config.yaml")
    print(f"[INFO] Using configured pretrained weight: {weight_path}")
    return str(weight_path)


def check_dataset_structure(data_dir):
    data_dir = Path(data_dir)

    required_dirs = [
        data_dir / "train" / "normal",
        data_dir / "train" / "damaged",
        data_dir / "val" / "normal",
        data_dir / "val" / "damaged",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing dataset directory: {d}")

    print("[INFO] Dataset structure OK")

    for split in ["train", "val"]:
        for cls in ["normal", "damaged"]:
            img_dir = data_dir / split / cls
            n = len([
                p for p in img_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ])
            print(f"[INFO] {split}/{cls}: {n} images")


def main():
    check_dataset_structure(DATA_DIR)

    weight_path = find_yolo_cls_weight()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = YOLO(weight_path)

    results = model.train(
        data=DATA_DIR,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,

        project=PROJECT_DIR,
        name=RUN_NAME,

        # 학습률
        lr0=0.001,

        # 드론 촬영 환경 대응용 augmentation
        degrees=8,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.30,
        hsv_v=0.25,

        # 저장
        save=True,
        verbose=True,
    )

    print("[INFO] Training finished")
    print(results)

    best_pt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    last_pt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"

    print(f"[INFO] best.pt: {best_pt}")
    print(f"[INFO] last.pt: {last_pt}")

    # 검증
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        val_results = best_model.val(
            data=DATA_DIR,
            imgsz=IMG_SIZE,
            device=device,
        )
        print("[INFO] Validation finished")
        print(val_results)


if __name__ == "__main__":
    main()