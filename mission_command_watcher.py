"""Continuously watch a folder and run an external mission command per video."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
FILE_STABLE_CHECK_INTERVAL_SEC = 1.0
FILE_STABLE_REQUIRED_COUNT = 3
FILE_READY_TIMEOUT_SEC = 120.0
POLL_INTERVAL_SEC = 1.0

MISSION_SEQUENCE: dict[int, dict[str, str]] = {
    1: {"label": "FA-01", "mission": "facility"},
    2: {"label": "FA-02", "mission": "facility"},
    3: {"label": "FA-03", "mission": "facility"},
    4: {"label": "FA-04", "mission": "facility"},
    5: {"label": "FA-05", "mission": "facility"},
    6: {"label": "FA-06", "mission": "facility"},
    7: {"label": "MAPPING", "mission": "mapping"},
    8: {"label": "TW-A", "mission": "obstacle", "route": "TWA"},
    9: {"label": "RW", "mission": "obstacle", "route": "RW"},
    10: {"label": "TW-B", "mission": "obstacle", "route": "TWB"},
}


@dataclass(frozen=True)
class WatcherConfig:
    """Runtime configuration for command-based mission watching."""

    watch_dir: Path
    output_dir: Path
    runtime_dir: Path
    main_py: Path
    python_executable: str
    state_file: Path
    log_file: Path
    poll_interval_sec: float
    stable_check_interval_sec: float
    stable_required_count: int
    ready_timeout_sec: float
    skip_opencv_check: bool
    archive_dir: Path = Path("archive")


class MissionState:
    """Persist sequence and processed source files for restart-safe command routing."""

    def __init__(self, state_file: Path, reset: bool = False) -> None:
        """Load state or initialize the default state."""
        self.state_file = state_file
        if reset:
            self.reset()
        else:
            self.data = self._load()

    def next_sequence(self) -> int:
        """Return the logical sequence number for the next successful mission."""
        return int(self.data["next_sequence"])

    def reset(self) -> None:
        """Reset and persist state for a new watcher session."""
        self.data = {
            "next_sequence": 1,
            "processed_source_files": [],
            "completed_missions": [],
            "failed_missions": [],
        }
        self.save()

    def is_processed(self, source_path: Path) -> bool:
        """Return True if this source path was already completed or skipped."""
        return resolve_path(source_path) in self.data["processed_source_files"]

    def mark_completed(self, sequence: int, source_path: Path, output_path: Path, command: list[str]) -> None:
        """Record successful command completion and advance to the next sequence."""
        resolved = resolve_path(source_path)
        if resolved not in self.data["processed_source_files"]:
            self.data["processed_source_files"].append(resolved)
        self.data["completed_missions"].append(
            {
                "sequence": sequence,
                "source_path": resolved,
                "output_path": str(output_path),
                "command": command,
                "completed_at": now_iso(),
            }
        )
        self.data["next_sequence"] = sequence + 1
        self.save()

    def mark_failed(self, sequence: int, source_path: Path, command: list[str], error: str) -> None:
        """Record command failure without advancing the sequence."""
        self.data["failed_missions"].append(
            {
                "sequence": sequence,
                "source_path": resolve_path(source_path),
                "command": command,
                "error": error,
                "failed_at": now_iso(),
            }
        )
        self.save()

    def mark_unexpected_skipped(self, source_path: Path) -> None:
        """Record an 11th-or-later video as skipped so it is not retried forever."""
        resolved = resolve_path(source_path)
        if resolved not in self.data["processed_source_files"]:
            self.data["processed_source_files"].append(resolved)
        self.data["failed_missions"].append(
            {
                "sequence": self.next_sequence(),
                "source_path": resolved,
                "error": "All 10 expected videos are already processed. Unexpected video skipped.",
                "failed_at": now_iso(),
            }
        )
        self.save()

    def save(self) -> None:
        """Atomically save state JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_file.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(self.state_file)

    def _load(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {
                "next_sequence": 1,
                "processed_source_files": [],
                "completed_missions": [],
                "failed_missions": [],
            }
        with self.state_file.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        loaded.setdefault("next_sequence", 1)
        loaded.setdefault("processed_source_files", [])
        loaded.setdefault("completed_missions", [])
        loaded.setdefault("failed_missions", [])
        return loaded


class VideoReadyChecker:
    """Check whether an incoming video file is stable and readable."""

    @staticmethod
    def wait_until_ready(video_path: Path, config: WatcherConfig) -> bool:
        """Wait until the file size is stable and optionally OpenCV can read frame 1."""
        deadline = time.monotonic() + config.ready_timeout_sec
        previous_size = -1
        stable_count = 0
        logging.info("File readiness check started: %s", video_path)
        while time.monotonic() < deadline:
            if video_path.exists() and video_path.stat().st_size > 0:
                current_size = video_path.stat().st_size
                if current_size == previous_size:
                    stable_count += 1
                else:
                    stable_count = 0
                previous_size = current_size
                if stable_count >= config.stable_required_count:
                    if config.skip_opencv_check or VideoReadyChecker._can_open_video(video_path):
                        logging.info("File readiness check completed: %s", video_path)
                        return True
                    logging.warning("File is stable but OpenCV cannot read it yet: %s", video_path)
            time.sleep(config.stable_check_interval_sec)
        logging.error("File readiness check timed out: %s", video_path)
        return False

    @staticmethod
    def _can_open_video(video_path: Path) -> bool:
        """Return whether OpenCV can open the video and read the first frame."""
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return False
            ok, _ = capture.read()
            return bool(ok)
        finally:
            capture.release()


class CommandMissionWatcher:
    """Poll a folder and run `python main.py --mission ... --input ... --output ...`."""

    def __init__(self, config: WatcherConfig) -> None:
        """Initialize watcher state."""
        self.config = config
        self.state = MissionState(config.state_file)
        self.pending_paths: set[str] = set()

    def run_forever(self) -> None:
        """Continuously poll the watch folder and process videos in arrival order."""
        logging.info("Command mission watcher started: %s", self.config.watch_dir)
        while True:
            self.scan_once()
            time.sleep(self.config.poll_interval_sec)

    def scan_once(self) -> None:
        """Scan the watch folder once and process currently available videos."""
        self.config.watch_dir.mkdir(parents=True, exist_ok=True)
        videos = [path for path in self.config.watch_dir.iterdir() if is_supported_video(path)]
        videos.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
        for video_path in videos:
            self.process_video(video_path)

    def process_video(self, video_path: Path) -> None:
        """Process one video if it has not already been processed or queued."""
        resolved = resolve_path(video_path)
        if self.state.is_processed(video_path):
            logging.info("Already processed video skipped: %s", video_path)
            return
        if resolved in self.pending_paths:
            logging.debug("Video already pending; duplicate scan ignored: %s", video_path)
            return

        self.pending_paths.add(resolved)
        try:
            self._process_video_locked(video_path)
        finally:
            self.pending_paths.discard(resolved)

    def _process_video_locked(self, video_path: Path) -> None:
        sequence = self.state.next_sequence()
        if sequence > len(MISSION_SEQUENCE):
            logging.warning("All 10 expected videos are already processed. Unexpected video skipped: %s", video_path)
            self.state.mark_unexpected_skipped(video_path)
            return

        mission_info = MISSION_SEQUENCE[sequence]
        output_path = self.config.output_dir / f"mission_{sequence:02d}_result.json"
        if mission_info["mission"] == "mapping":
            output_path = load_map_output_dir(Path("config.yaml"))
        command = build_command(
            self.config,
            str(mission_info["mission"]),
            video_path,
            output_path,
            mission_info.get("route"),
        )

        logging.info(
            "Sequence %s (%s) will run mission=%s input=%s output=%s",
            sequence,
            mission_info["label"],
            mission_info["mission"],
            video_path,
            output_path,
        )

        if not VideoReadyChecker.wait_until_ready(video_path, self.config):
            self.state.mark_failed(sequence, video_path, command, "Video file was not ready before timeout.")
            return

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.stdout:
            logging.info("Mission stdout for sequence %s:\n%s", sequence, result.stdout.strip())
        if result.stderr:
            logging.warning("Mission stderr for sequence %s:\n%s", sequence, result.stderr.strip())

        if result.returncode != 0:
            error = f"Command exited with code {result.returncode}"
            logging.error("Mission failed: sequence=%s %s", sequence, error)
            self.state.mark_failed(sequence, video_path, command, error)
            return

        logging.info("Mission succeeded: sequence=%s", sequence)
        self.state.mark_completed(sequence, video_path, output_path, command)
        archive_path = self.config.archive_dir / video_path.name
        try:
            self.config.archive_dir.mkdir(parents=True, exist_ok=True)
            video_path.rename(archive_path)
        except OSError as exc:
            logging.warning("Video archiving failed; mission remains completed: %s", exc)
        else:
            logging.info("Video archived: %s", archive_path)


def build_command(
    config: WatcherConfig,
    mission: str,
    input_path: Path,
    output_path: Path,
    route: str | None = None,
) -> list[str]:
    """Build the external mission command exactly as requested."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.python_executable,
        str(config.main_py),
        "--mission",
        mission,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if mission == "obstacle" and route:
        command.extend(["--route", route])
    return command


def is_supported_video(path: Path) -> bool:
    """Return True for supported video files only."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def load_map_output_dir(config_path: Path) -> Path:
    """Read paths.map_output_dir from the project configuration."""
    in_paths_section = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            in_paths_section = line == "paths:"
            continue
        if in_paths_section and line.strip().startswith("map_output_dir:"):
            value = line.strip().split(":", 1)[1].strip().strip('"\'')
            return Path(value)
    raise KeyError(f"paths.map_output_dir not found in {config_path}")


def resolve_path(path: Path) -> str:
    """Resolve a path for stable state comparisons."""
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def now_iso() -> str:
    """Return current local timestamp."""
    return datetime.now().astimezone().isoformat()


def configure_logging(log_file: Path) -> None:
    """Configure console and file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Watch a folder and route videos to external mission main.py commands.")
    parser.add_argument("--watch-dir", type=Path, default=Path("input_data"))
    parser.add_argument("--archive-dir", type=Path, default=Path("archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_json"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--main-py", type=Path, default=Path("main.py"))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_SEC)
    parser.add_argument("--ready-timeout", type=float, default=FILE_READY_TIMEOUT_SEC)
    parser.add_argument("--skip-opencv-check", action="store_true")
    parser.add_argument("--once", action="store_true", help="scan once and exit; useful for tests")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> WatcherConfig:
    """Build a WatcherConfig from parsed CLI arguments."""
    state_file = args.runtime_dir / "mission_command_state.json"
    log_file = args.output_dir / "logs" / "mission_command_watcher.log"
    return WatcherConfig(
        watch_dir=args.watch_dir,
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        runtime_dir=args.runtime_dir,
        main_py=args.main_py,
        python_executable=args.python_executable,
        state_file=state_file,
        log_file=log_file,
        poll_interval_sec=args.poll_interval,
        stable_check_interval_sec=FILE_STABLE_CHECK_INTERVAL_SEC,
        stable_required_count=FILE_STABLE_REQUIRED_COUNT,
        ready_timeout_sec=args.ready_timeout,
        skip_opencv_check=args.skip_opencv_check,
    )


def main() -> None:
    """Run the command mission watcher."""
    args = parse_args()
    config = make_config(args)
    configure_logging(config.log_file)
    config.watch_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_dir.mkdir(parents=True, exist_ok=True)

    MissionState(config.state_file, reset=True)
    watcher = CommandMissionWatcher(config)
    if args.once:
        watcher.scan_once()
    else:
        watcher.run_forever()


if __name__ == "__main__":
    main()
