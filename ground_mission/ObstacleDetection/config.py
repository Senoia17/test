"""Application configuration for the obstacle detection pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parents[1]

DATASET_DIR: Path = BASE_DIR / "dataset"
DATA_YAML: Path = DATASET_DIR / "data.yaml"

WEIGHTS_DIR: Path = PROJECT_ROOT / "models" / "obstacle"


def _load_project_config() -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from main import load_config

    return load_config(PROJECT_ROOT / "config.yaml")


def _configured_pretrained_model_path() -> Path:
    from mission.model_weights import resolve_pretrained_model_path

    pretrained_path = resolve_pretrained_model_path(_load_project_config(), "obstacle")
    if pretrained_path is None:
        raise ValueError("Missing models.obstacle.pretrained in config.yaml")
    return Path(pretrained_path)


def _configured_model_path() -> Path:
    from mission.model_weights import resolve_model_weight_path

    model_path = resolve_model_weight_path(_load_project_config(), "obstacle")
    if model_path is None:
        raise ValueError("Missing models.obstacle.directory/weights in config.yaml")
    return model_path


PRETRAINED_MODEL_PATH: Path = _configured_pretrained_model_path()
MODEL_PATH: Path = _configured_model_path()

INPUT_DIR: Path = BASE_DIR / "input"
INPUT_VIDEO_PATH: Path = INPUT_DIR / "mission_video.mp4"

OUTPUT_DIR: Path = BASE_DIR / "output"
CRATER_JSON_PATH: Path = OUTPUT_DIR / "crater_detect.json"
UXO_JSON_PATH: Path = OUTPUT_DIR / "uxo_detect.json"
ANNOTATED_VIDEO_PATH: Path = OUTPUT_DIR / "annotated_video.mp4"
LOG_PATH: Path = OUTPUT_DIR / "log.txt"

TRACKER_CONFIG_PATH: Path = BASE_DIR / "models" / "bytetrack.yaml"

CLASS_NAMES: dict[int, str] = {
    0: "crater",
    1: "cluster",
    2: "dumb",
    3: "missile",
}

CRATER_CLASS_NAME: str = "crater"
UXO_CLASS_NAMES: set[str] = {"cluster", "dumb", "missile"}

CONFIDENCE: float = 0.25
IOU_THRESHOLD: float = 0.7
IMAGE_SIZE: int = 640
DEVICE: str = "cpu"

TRAIN_EPOCHS: int = 100
TRAIN_BATCH_SIZE: int = 16
TRAIN_PROJECT: Path = BASE_DIR / "runs" / "train"
TRAIN_NAME: str = "yolo11_runway_obstacles"
TRAIN_WORKERS: int = 4

VIDEO_FRAME_STRIDE: int = 1
SAVE_ANNOTATED_VIDEO: bool = True
ANNOTATION_THICKNESS: int = 2
ANNOTATION_FONT_SCALE: float = 0.6
DEFAULT_VIDEO_FPS: float = 30.0
VIDEO_CODEC: str = "mp4v"

MIN_TRACK_OUTPUT_LENGTH: int = 5

UNKNOWN_ZONE: str = "UNKNOWN"
UNKNOWN_SIZE: str = "UNKNOWN"

# Zone cell sizes in centimeters. The first value is the flight/runway
# direction length, and the second value is lateral width.
FACILITY_ZONE_CELL_CM: tuple[float, float] = (160.0, 80.0)
TAXIWAY_ZONE_CELL_CM: tuple[float, float] = (100.0, 80.0)
RUNWAY_ZONE_CELL_CM: tuple[float, float] = (50.0, 80.0)

# Default zone layout. Origins are measured in centimeters in the future
# homography/world coordinate system. Adjust origins and counts to match the
# official competition map when it is finalized.
ZONE_LAYOUTS: tuple[dict, ...] = (
    {
        "name": "facility",
        "prefix": "FZ",
        "origin_cm": (0.0, 240.0),
        "cell_cm": FACILITY_ZONE_CELL_CM,
        "rows": 1,
        "cols": 10,
    },
    {
        "name": "taxiway",
        "prefix": "TW",
        "origin_cm": (0.0, 80.0),
        "cell_cm": TAXIWAY_ZONE_CELL_CM,
        "rows": 2,
        "cols": 10,
    },
    {
        "name": "runway",
        "prefix": "RW",
        "origin_cm": (0.0, 0.0),
        "cell_cm": RUNWAY_ZONE_CELL_CM,
        "rows": 1,
        "cols": 10,
    },
)
