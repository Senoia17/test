# Ground Mission YOLO Pipeline

This project trains and runs a YOLO detector for runway/taxiway ground objects:

- `crater`
- `missile`
- `cluster`
- `dumb`

Crater size is not learned by YOLO. The detector finds `crater`, then the
bounding box is projected to the generalized map coordinate system with a
homography matrix and classified as `small`, `medium`, or `large` by physical
size.

## Folder Layout

```text
ground_mission/
  configs/
    zones.yaml
  dataset/
    data.yaml
    images/
      train/
      val/
    labels/
      train/
      val/
  src/
    aruco_homography.py
    crater_size.py
    geometry.py
    infer_ground.py
    json_writer.py
    train_ground.py
    zone_locator.py
  weights/
  outputs/
```

## Install

```bash
pip install ultralytics opencv-python numpy pyyaml shapely
```

## Train

```bash
python src/train_ground.py
```

The trained model will be saved under `runs/ground_detector/weights/best.pt`.
Copy it to:

```text
../../models/obstacle/best.pt
```

## Create Homography

Use a high-altitude image where all four ArUco corner markers are visible.

```bash
python src/aruco_homography.py --image data/overview.jpg --output configs/homography.pkl
```

The script assumes four ArUco marker IDs:

- `0`: top-left
- `1`: top-right
- `2`: bottom-right
- `3`: bottom-left

These map to a 500 cm x 400 cm field.

## Inference

```bash
python src/infer_ground.py --source data/runway_video.mp4 --weights ../../models/obstacle/best.pt
```

Outputs:

```text
outputs/crater_detect.json
outputs/uxo_detect.json
```

