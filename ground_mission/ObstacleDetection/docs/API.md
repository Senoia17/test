# API

## `YoloDetector.detect_frame(frame)`

Returns frame-level detections:

```json
[
  {
    "class": "crater",
    "class_id": 0,
    "bbox": [10.0, 20.0, 100.0, 120.0],
    "confidence": 0.95
  }
]
```

## `VideoTracker.track_video(video_path, annotated_video_path=None)`

Returns one best detection per tracked object:

```json
[
  {
    "id": 1001,
    "class": "crater",
    "class_id": 0,
    "bbox": [10.0, 20.0, 100.0, 120.0],
    "confidence": 0.98
  }
]
```

## `JsonBuilder.write_json_files(...)`

Writes:

- `crater_detect.json`
- `uxo_detect.json`

## `ZoneMapper.get_zone(x_cm, y_cm)`

Uses the configured cell sizes:

- Facility zone: `160 x 80 cm`
- Taxiway zone: `100 x 80 cm`
- Runway zone: `50 x 80 cm`

Default labels are `FZ-01`, `TW-A2`, and `RW-01` style labels. Adjust
`ZONE_LAYOUTS` in `config.py` to match the official map origin and counts.
