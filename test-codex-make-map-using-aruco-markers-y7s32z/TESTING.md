# 테스트 방법

이 프로젝트는 드론 항공샷 이미지에서 대회장 네 꼭짓점 ArUco 마커를 검출한 뒤, 원근 보정을 통해 평면 지도를 만드는 스크립트를 제공합니다.

## 1. 의존성 설치

OpenCV의 ArUco 모듈은 `opencv-python`이 아니라 `opencv-contrib-python`에 포함되어 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 실제 촬영 이미지로 실행

네 꼭짓점 마커 ID를 알고 있다면 반드시 `--marker-ids`를 지정하는 것을 권장합니다. 순서는 최종 지도 기준 `좌상단 우상단 우하단 좌하단`입니다.

```bash
python aruco_diorama_mapper.py drone.jpg arena_map.png \
  --marker-ids 10 11 12 13 \
  --map-size-mm 5000 4000 \
  --pixels-per-meter 300 \
  --metadata arena_map.json \
  --debug-image detected_markers.png
```

출력 결과는 다음과 같습니다.

- `arena_map.png`: 원근 보정된 대회장 평면 지도(마커 중심이 아니라 각 마커의 가장 바깥쪽 꼭짓점을 전체 맵의 꼭짓점으로 사용)
- `arena_map.json`: 검출된 소스 좌표, 목적지 좌표, 호모그래피 행렬, 출력 크기
- `detected_markers.png`: 검출된 마커와 선택된 꼭짓점 순서를 확인하는 디버그 이미지

## 3. 합성 이미지 스모크 테스트

실제 드론 이미지가 없어도 합성 대회장 이미지를 생성해서 기본 동작을 확인할 수 있습니다.

```bash
python tests/synthetic_mapping_test.py
```

기본 실행은 pass/fail만 출력하고 이미지는 저장하지 않습니다. 결과 이미지를 직접 확인하려면 `--output-dir`를 지정하세요.

```bash
python tests/synthetic_mapping_test.py --output-dir test_outputs
```

성공하면 다음 파일이 생성됩니다.

- `test_outputs/01_expected_top_down.png`: 테스트가 기대하는 정답 평면 대회장 이미지
- `test_outputs/02_synthetic_drone_input.png`: 드론 촬영처럼 원근 변환한 입력 이미지
- `test_outputs/03_mapped_output.png`: 매퍼가 복원한 평면 지도 결과
- `test_outputs/04_detected_markers_debug.png`: 검출된 마커와 꼭짓점 순서 표시 이미지
- `test_outputs/metadata.json`: 검출된 바깥쪽 꼭짓점 좌표, 목적지 좌표, 호모그래피 행렬

성공하면 다음과 비슷한 메시지가 출력됩니다.

```text
Synthetic ArUco mapping smoke test passed. Artifacts saved to: test_outputs
```

## 4. 문법 검사

의존성이 설치되어 있지 않은 환경에서도 문법 검사는 가능합니다.

```bash
python -m py_compile aruco_diorama_mapper.py tests/synthetic_mapping_test.py
```

## 5. 맵 크기와 기준점

- 기본 실제 맵 크기는 `5000 4000` mm입니다. 즉 `--map-size-mm`을 생략하면 5 m × 4 m 지도로 생성됩니다.
- 크기를 바꾸려면 `--map-size-mm WIDTH_MM HEIGHT_MM`를 직접 지정하세요. 예: `--map-size-mm 3000 2000`
- 전체 map의 네 꼭짓점은 ArUco 마커의 중심점이 아니라 각 코너 마커의 가장 바깥쪽 꼭짓점입니다. 따라서 결과 map에는 네 개의 코너 마커 전체가 포함됩니다.
- 기존 방식과 호환이 필요하면 `--arena-size WIDTH_M HEIGHT_M`도 사용할 수 있지만, 새 사용법은 `--map-size-mm`입니다.

## 6. 테스트가 실패할 때 확인할 것

- `ModuleNotFoundError: No module named 'aruco_diorama_mapper'`: 예전 테스트 파일은 `tests/` 안에서 실행될 때 프로젝트 루트가 import 경로에 없어 발생할 수 있었습니다. 최신 테스트 파일은 프로젝트 루트를 자동으로 `sys.path`에 추가합니다. 최신 코드로 업데이트한 뒤 `/workspaces/test` 루트에서 다시 실행하세요.
- `ModuleNotFoundError: No module named 'cv2'`: `pip install -r requirements.txt`를 먼저 실행하세요.
- `Expected at least 4 ArUco markers`: 이미지에서 네 꼭짓점 마커가 모두 선명하게 보이는지 확인하세요.
- `Required corner marker IDs were not detected`: `--marker-ids`에 입력한 ID와 실제 출력/부착한 ArUco ID가 같은지 확인하세요.
- 결과 지도가 뒤집히거나 회전됨: `--marker-ids` 순서가 `좌상단 우상단 우하단 좌하단`인지 확인하세요.
