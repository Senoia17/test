# Drone AI Ground Mission Pipeline

This repository contains a modular drone AI mission stack for the ground mission. The current integration keeps legacy mission code available while providing a central end-to-end pipeline built from reusable modules.

## Pipeline Architecture

```text
raw video frame / image sequence
  -> calibration adapter / undistortion
  -> YOLO detection adapter
  -> homography localization adapter
  -> pixel-to-global-map analysis
  -> mission JSON output
```

The integrated entry point is `mission.pipeline.GroundMissionPipeline`, exposed through `main.py`.

```bash
python main.py \
  --input data/video.mp4 \
  --config config.yaml \
  --output output/ \
  --debug
```

`--mission` is optional for the integrated ground pipeline and defaults to `ground`. Existing mission adapters are still available:

```bash
python main.py --mission facility --input data/facility.mp4 --output output/facility.json
python main.py --mission mapping --input data/mapping_video.mp4 --output data/map
python main.py --mission obstacle --input data/obstacle.mp4 --output output/obstacle.json
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `calibration/` | Camera model loading and frame undistortion. |
| `detection/` | YOLO runtime, object detection, classification, and generic detection post-processing. |
| `geometry/` | Coordinate transforms, homography projection, and physical measurement helpers. |
| `mapping/` | Video-based global map generation, ArUco detection, map homography, and map metadata artifacts. |
| `localization/` | Frame-to-map localization using ArUco, map matching, temporal frame matching, and homography selection. |
| `obstacle/` | Crater measurement/classification and UXO analysis from normalized detections and localization. |
| `facility/` | Facility mission inference, training entry point, and pipeline adapter. |
| `mission/adapters/` | Lightweight adapters that hide implementation details and expose stable pipeline contracts. |
| `mission/pipeline.py` | Central Ground Mission orchestration. |
| `mission/json_writer.py` | Flexible JSON output writer. |
| `utils/video.py` | Shared video frame iteration helpers. |

## Data Contracts

The central pipeline uses dataclasses in `mission/types.py`:

- `DetectionResult`
  - `class_name`
  - `confidence`
  - `bbox`
  - `center_pixel`
  - `class_id`
- `LocalizationResult`
  - `homography_matrix`
  - `confidence`
  - `method`
  - `localized`
  - `error`
- `MappedObject`
  - `object_info`
  - `global_coordinate`
  - `localized`

## Input / Output

### Input

The integrated ground pipeline accepts either:

- a video file path, or
- a directory containing image frames.

Optional config-driven inputs include:

- `paths.calibration`
- `paths.global_map`
- `paths.map_info`
- `models.obstacle.directory` + `models.obstacle.version`

If optional components are missing, the central pipeline records adapter errors and continues where safe. Mission-specific adapters may enforce stricter runtime policies.

### Output

If `--output` points to a directory, the central pipeline writes:

```text
output/ground_mission_results.json
```

If `--output` points to a `.json` file, that exact file is written.

The JSON includes:

- adapter status
- per-frame detections
- per-frame localization result
- crater results
- UXO results

## Debug Output

When `--debug` is enabled, debug images are saved under:

```text
output/debug/
  frame_000_original.jpg
  frame_000_undistorted.jpg
  frame_000_matches.jpg
  frame_000_warped.jpg
  frame_000_result.jpg
```

The matching visualization is currently a placeholder image because the existing localization APIs do not expose raw match visualizations yet.

## Legacy Compatibility

Legacy files are preserved, including:

- `ground_mission/src/infer_ground.py`
- `ground_mission/src/geometry.py`
- `ground_mission/src/aruco_homography.py`
- `ground_mission/ObstacleDetection/`
- `facility/state_infer.py`

Compatibility wrappers delegate reusable logic to the newer shared modules where safe.

## Remaining TODOs

- Expose raw feature matches from localization for real debug match visualizations.
- Add schema-specific competition JSON once the final output contract is fixed.
- Decide when legacy ground mission entry points can be retired.
- Add fixture-based integration tests with representative videos and map artifacts.
