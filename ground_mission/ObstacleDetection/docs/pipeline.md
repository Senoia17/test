# Pipeline

1. Load `input/mission_video.mp4`.
2. Read frames with OpenCV.
3. Run `YoloDetector.detect_frame()` for frame-level inference.
4. Pass detector outputs to per-class ByteTrack trackers.
5. Keep the highest-confidence bounding box for each track ID.
6. Apply zone mapping when homography/world coordinates are present, and apply
   dummy crater size classification.
7. Write `output/crater_detect.json` and `output/uxo_detect.json`.

Homography is intentionally deferred. `zone_mapper.py` already understands
world coordinates in centimeters through `center_cm`, `world_center_cm`, or
`world_bbox_cm`; without those fields it returns `UNKNOWN`.
