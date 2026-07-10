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
    calibration.py
    crater_size.py
    frame_alignment.py
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
weights/ground_best.pt
```

## Camera Calibration

Lens distortion is a camera/lens property, so estimate it once for the drone camera and reuse the JSON for every frame captured with the same lens/zoom setting.

```bash
python src/calibration.py --images-dir data/calibration --output configs/calibration.json
```

`calibration.py` exposes helpers used by inference and frame alignment. A calibration JSON has:

```json
{
  "camera_matrix": [[...], [...], [...]],
  "distortion_coefficients": [...],
  "image_size": [width, height],
  "reprojection_error": 0.42
}
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

## Sequential Frame Alignment

When the current drone frame is too different from the full map, estimate local homographies between neighboring frames and compose them into the map coordinate system. This supports the `local tracking + global correction` workflow and can also write warped preview frames.

```bash
python src/frame_alignment.py \
  --map-image data/overview.jpg \
  --frames-dir data/frames \
  --output-dir outputs/alignment \
  --calibration-json configs/calibration.json \
  --map-homography configs/homography.pkl
```

The metadata contains `homography_current_to_map_px` and, when `--map-homography` is supplied, `homography_current_to_world`.

## Inference

```bash
python src/infer_ground.py \
  --source data/runway_video.mp4 \
  --weights weights/ground_best.pt \
  --homography configs/homography.pkl \
  --calibration configs/calibration.json
```

Outputs:

```text
outputs/crater_detect.json
outputs/uxo_detect.json
```

