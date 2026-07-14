# 연속 드론 프레임 → 전체 map 좌표계 정렬

이 디렉토리는 이미 만들어진 전체 top-down map과, 시간 순서대로 저장된 드론 프레임 폴더를 이용해 각 프레임의 좌표 변환 행렬을 구하는 도구입니다.

핵심 순서는 다음과 같습니다.

```text
raw frame
  ↓
공통 렌즈 왜곡 보정(optional, 모든 프레임에 동일 적용)
  ↓
직전 이미지와 feature matching
  ↓
현재 프레임 → 직전 이미지 homography
  ↓
누적 합성으로 현재 프레임 → 전체 map homography
  ↓
전체 map canvas로 warp(optional)
```

첫 번째 프레임의 직전 이미지는 `--map-image`로 전달한 전체 map입니다. 두 번째 프레임부터는 바로 이전 프레임과 matching하고, 변환 행렬을 누적해서 전체 map 좌표계 기준 homography를 만듭니다.

## 중요 개념

- 렌즈 왜곡 보정은 카메라 렌즈 때문에 직선이 휘는 문제를 줄입니다.
- 렌즈 왜곡 보정값은 같은 카메라/렌즈/초점 설정이면 모든 프레임에 공통 적용할 수 있습니다.
- 하지만 렌즈 보정만으로 각 프레임이 top-view가 되지는 않습니다.
- 드론의 위치, 고도, yaw/pitch/roll 변화는 프레임마다 다르므로 homography는 프레임마다 새로 계산합니다.

## 기본 실행

```bash
python frame_homography_pipeline/process_frame_sequence.py \
  --map-image arena_map.png \
  --frames-dir drone_frames \
  --output-dir sequence_outputs \
  --map-size-mm 5000 4000
```

출력:

- `sequence_outputs/sequence_metadata.json`
  - 각 프레임의 `homography_current_to_previous`
  - 각 프레임의 `homography_current_to_map`
  - raw/good match 수, RANSAC inlier 수
- `sequence_outputs/warped_frames/`
  - 각 프레임을 전체 map canvas에 맞춰 warp한 이미지

## 렌즈 왜곡 보정값 사용

이미 카메라 calibration JSON이 있다면:

```bash
python frame_homography_pipeline/process_frame_sequence.py \
  --map-image arena_map.png \
  --frames-dir drone_frames \
  --output-dir sequence_outputs \
  --map-size-mm 5000 4000 \
  --calibration-json camera_calibration.json
```

JSON 형식:

```json
{
  "camera_matrix": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
  "distortion_coefficients": [0.01, -0.02, 0.0, 0.0, 0.0],
  "image_size": [1280, 720],
  "reprojection_error": 0.4
}
```

## 체스보드 이미지로 calibration 추정

첫 항공샷 하나만으로 일반적인 렌즈 왜곡 계수를 안정적으로 추정하기는 어렵습니다. 실제로는 같은 카메라로 촬영한 체스보드/캘리브레이션 패턴 이미지 여러 장을 사용하는 것이 표준입니다.

```bash
python frame_homography_pipeline/process_frame_sequence.py \
  --map-image arena_map.png \
  --frames-dir drone_frames \
  --output-dir sequence_outputs \
  --map-size-mm 5000 4000 \
  --calibration-videos marker_1.mp4 marker_2.mp4 marker_3.mp4 marker_4.mp4 \
  --config config.yaml \
  --save-calibration-json camera_calibration.json
```

이렇게 얻은 `camera_calibration.json`은 이후 같은 카메라로 찍은 모든 프레임에 재사용하면 됩니다.

## 주의사항

- feature matching 방식은 map과 프레임 사이에 충분한 텍스처/특징점이 있어야 안정적입니다.
- 연속 프레임 homography를 누적하면 drift가 생길 수 있습니다. 실전에서는 일정 간격으로 전체 map과 다시 matching하거나 ArUco가 보일 때마다 전역 보정을 추가하는 것이 좋습니다.
- `sequence_metadata.json`의 inlier 수가 급격히 낮아지는 프레임은 정렬 실패 가능성이 큽니다.
