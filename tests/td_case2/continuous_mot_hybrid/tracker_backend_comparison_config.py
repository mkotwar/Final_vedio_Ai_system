from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from tests.td_case2.config import (
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_YOLO_MODEL_PATH,
    repo_root as td_repo_root,
    resolve_case_path,
)


@dataclass(frozen=True)
class TrackerBackendComparisonConfig:
    video_path: Path
    run_dir: Path
    camera_id: str
    camera_group: str
    camera_timezone: str
    processing_fps: float
    detector_fps: float
    device: str
    reid_model: str
    run_bytetrack: bool
    run_botsort_no_reid: bool
    run_botsort_reid: bool
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    match_thresh: float = 0.80
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare isolated tracker backends using one shared YOLO cache.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--camera-id", default="test_cam_01")
    parser.add_argument("--camera-group", default="single_camera_comparison")
    parser.add_argument("--camera-timezone", default="Asia/Kolkata")
    parser.add_argument("--processing-fps", type=float, default=10.0)
    parser.add_argument("--detector-fps", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-bytetrack", action="store_true")
    parser.add_argument("--run-botsort-no-reid", action="store_true")
    parser.add_argument("--run-botsort-reid", action="store_true")
    parser.add_argument("--reid-model", default="auto")
    return parser


def make_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = td_repo_root() / "debug_runs" / f"tracker_backend_comparison_{video_path.stem}_{timestamp}"
    for child in ("00_shared", "01_bytetrack", "02_botsort_no_reid", "03_botsort_reid", "04_comparison", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config(args: argparse.Namespace) -> TrackerBackendComparisonConfig:
    video_path = Path(args.video_path).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else make_run_dir(video_path)
    if args.run_dir:
        for child in ("00_shared", "01_bytetrack", "02_botsort_no_reid", "03_botsort_reid", "04_comparison", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
    run_bytetrack = bool(args.run_bytetrack)
    run_botsort_no_reid = bool(args.run_botsort_no_reid)
    run_botsort_reid = bool(args.run_botsort_reid)
    if not any((run_bytetrack, run_botsort_no_reid, run_botsort_reid)):
        run_bytetrack = True
        run_botsort_no_reid = True
        run_botsort_reid = True
    return TrackerBackendComparisonConfig(
        video_path=video_path,
        run_dir=run_dir,
        camera_id=str(args.camera_id),
        camera_group=str(args.camera_group),
        camera_timezone=str(args.camera_timezone),
        processing_fps=float(args.processing_fps),
        detector_fps=float(args.detector_fps),
        device=str(args.device),
        reid_model=str(args.reid_model),
        run_bytetrack=run_bytetrack,
        run_botsort_no_reid=run_botsort_no_reid,
        run_botsort_reid=run_botsort_reid,
    )


def resolve_models() -> tuple[Path | None, Path | None, Path | None]:
    def _path(env_name: str) -> Path | None:
        import os

        raw = str(os.environ.get(env_name, "")).strip()
        return resolve_case_path(raw) if raw else None

    return _path(ENV_PERSON_YOLO_MODEL_PATH), _path(ENV_OBJECT_YOLO_MODEL_PATH), _path(ENV_YOLO_MODEL_PATH)
