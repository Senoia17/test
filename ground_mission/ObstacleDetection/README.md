# ObstacleDetection

Drone runway obstacle detection pipeline for runway craters and UXO objects.

## Features

- YOLO11 training with Ultralytics
- Frame-level YOLO inference
- ByteTrack-based object tracking without `model.track()`
- Final crater and UXO JSON export
- Dummy zone mapping and crater size classification placeholders

## Quick Start

```powershell
cd ObstacleDetection
pip install -r requirements.txt
python main.py
```

Place the mission video at `input/mission_video.mp4` and trained weights at
`weights/best.pt`. The pipeline writes:

- `output/crater_detect.json`
- `output/uxo_detect.json`
- `output/annotated_video.mp4`
- `output/log.txt`

## Training

Prepare `dataset/data.yaml`, images, and labels, then run:

```powershell
python -m detection.train
```
