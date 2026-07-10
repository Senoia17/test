from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=1280,
        batch=8,
        workers=4,
        device=0,
        patience=30,
        project="runs",
        name="ground_detector",
        exist_ok=True,
        degrees=5,
        translate=0.08,
        scale=0.25,
        perspective=0.0008,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
    )


if __name__ == "__main__":
    main()

