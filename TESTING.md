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
  --arena-size 4.0 2.5 \
  --pixels-per-meter 300 \
  --metadata arena_map.json \
  --debug-image detected_markers.png
```

출력 결과는 다음과 같습니다.

- `arena_map.png`: 원근 보정된 대회장 평면 지도
- `arena_map.json`: 검출된 소스 좌표, 목적지 좌표, 호모그래피 행렬, 출력 크기
- `detected_markers.png`: 검출된 마커와 선택된 꼭짓점 순서를 확인하는 디버그 이미지

## 3. 합성 이미지 스모크 테스트

실제 드론 이미지가 없어도 합성 대회장 이미지를 생성해서 기본 동작을 확인할 수 있습니다.

```bash
python tests/synthetic_mapping_test.py
```

성공하면 다음 메시지가 출력됩니다.

```text
Synthetic ArUco mapping smoke test passed.
```

## 4. 문법 검사

의존성이 설치되어 있지 않은 환경에서도 문법 검사는 가능합니다.

```bash
python -m py_compile aruco_diorama_mapper.py tests/synthetic_mapping_test.py
```

## 5. 테스트가 실패할 때 확인할 것

- `ModuleNotFoundError: No module named 'cv2'`: `pip install -r requirements.txt`를 먼저 실행하세요.
- `Expected at least 4 ArUco markers`: 이미지에서 네 꼭짓점 마커가 모두 선명하게 보이는지 확인하세요.
- `Required corner marker IDs were not detected`: `--marker-ids`에 입력한 ID와 실제 출력/부착한 ArUco ID가 같은지 확인하세요.
- 결과 지도가 뒤집히거나 회전됨: `--marker-ids` 순서가 `좌상단 우상단 우하단 좌하단`인지 확인하세요.
