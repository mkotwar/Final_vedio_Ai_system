from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from tests.td_case2.config import (
    CASE_ENV_PATH,
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_YOLO_DEVICE,
    ENV_YOLO_MODEL_PATH,
    repo_root,
    resolve_case_path,
)


ENV_RUN_DIR = "TD_CASE2_CONTINUOUS_RUN_DIR"
ENV_VIDEO_PATH = "TD_CASE2_CONTINUOUS_VIDEO_PATH"
ENV_CAMERA_ID = "TD_CASE2_CONTINUOUS_CAMERA_ID"
ENV_CAMERA_GROUP = "TD_CASE2_CONTINUOUS_CAMERA_GROUP"
ENV_CAMERA_TIMEZONE = "TD_CASE2_CONTINUOUS_CAMERA_TIMEZONE"
ENV_PROCESSING_FPS = "TD_CASE2_CONTINUOUS_PROCESSING_FPS"
ENV_MOT_BACKEND = "TD_CASE2_CONTINUOUS_MOT_BACKEND"
ENV_DEVICE = "TD_CASE2_CONTINUOUS_DEVICE"
ENV_ROI = "TD_CASE2_CONTINUOUS_ROI"
ENV_ENABLE_SHORT_GAP_VISUAL_TRACKER = "TD_CASE2_CONTINUOUS_ENABLE_SHORT_GAP_VISUAL_TRACKER"
ENV_SHORT_GAP_VISUAL_TRACKER = "TD_CASE2_CONTINUOUS_VISUAL_TRACKER"
ENV_DETECTOR_NORMAL_INTERVAL = "TD_CASE2_CONTINUOUS_DETECTOR_NORMAL_INTERVAL"
ENV_DETECTOR_SPARSE_INTERVAL = "TD_CASE2_CONTINUOUS_DETECTOR_SPARSE_INTERVAL"
ENV_DETECTOR_IDLE_INTERVAL = "TD_CASE2_CONTINUOUS_DETECTOR_IDLE_INTERVAL"
ENV_DETECTOR_MAX_GAP = "TD_CASE2_CONTINUOUS_DETECTOR_MAX_GAP"
ENV_VISUAL_BRIDGE_MAX_SECONDS = "TD_CASE2_CONTINUOUS_VISUAL_BRIDGE_MAX_SECONDS"
ENV_LOST_RECOVERY_SECONDS = "TD_CASE2_CONTINUOUS_LOST_RECOVERY_SECONDS"
ENV_MIN_PERSON_CONFIRM_HITS = "TD_CASE2_CONTINUOUS_MIN_PERSON_CONFIRM_HITS"
ENV_MIN_VEHICLE_CONFIRM_HITS = "TD_CASE2_CONTINUOUS_MIN_VEHICLE_CONFIRM_HITS"
ENV_SAVE_DEBUG_FRAMES = "TD_CASE2_CONTINUOUS_SAVE_DEBUG_FRAMES"
ENV_YOLO_CONFIDENCE = "TD_CASE2_CONTINUOUS_YOLO_CONFIDENCE"
ENV_YOLO_IOU = "TD_CASE2_CONTINUOUS_YOLO_IOU"
ENV_TRACK_HIGH_THRESH = "TD_CASE2_CONTINUOUS_TRACK_HIGH_THRESH"
ENV_TRACK_LOW_THRESH = "TD_CASE2_CONTINUOUS_TRACK_LOW_THRESH"
ENV_MATCH_THRESH = "TD_CASE2_CONTINUOUS_MATCH_THRESH"
ENV_TRACK_BUFFER_SECONDS = "TD_CASE2_CONTINUOUS_TRACK_BUFFER_SECONDS"
ENV_DUPLICATE_OVERLAP_SECONDS = "TD_CASE2_CONTINUOUS_DUPLICATE_OVERLAP_SECONDS"


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _read_float(value: Any, default: float, *, allow_zero: bool = False) -> float:
    if value is None or str(value).strip() == "":
        return default
    parsed = float(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"Expected positive float, received {parsed}")
    return parsed


def _read_int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer, received {parsed}")
    return parsed


def _read_path_env(env_name: str) -> str:
    raw = os.environ.get(env_name, "").strip()
    if raw:
        return raw
    local_value = str(dotenv_values(CASE_ENV_PATH).get(env_name) or "").strip()
    return local_value


def _resolve_optional_case_path(raw_value: str | None) -> Path | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    return resolve_case_path(str(raw_value))


@dataclass(frozen=True)
class ContinuousMotHybridConfig:
    video_path: Path
    run_dir: Path
    camera_id: str
    camera_group: str
    camera_timezone: str
    processing_fps: float
    mot_backend: str
    device: str
    roi: str | None
    enable_short_gap_visual_tracker: bool
    short_gap_visual_tracker: str
    detector_normal_interval_seconds: float
    detector_sparse_interval_seconds: float
    detector_idle_interval_seconds: float
    detector_max_gap_seconds: float
    visual_bridge_max_seconds: float
    lost_recovery_seconds: float
    min_person_confirm_hits: int
    min_vehicle_confirm_hits: int
    save_debug_frames: bool
    yolo_confidence: float
    yolo_iou: float
    track_high_thresh: float
    track_low_thresh: float
    match_thresh: float
    track_buffer_seconds: float
    duplicate_overlap_seconds: float
    person_model_path: Path | None
    object_model_path: Path | None
    combined_model_path: Path | None

    @property
    def output_dirs(self) -> dict[str, Path]:
        return {
            "video": self.run_dir / "01_video",
            "frames": self.run_dir / "02_frames",
            "detections": self.run_dir / "03_detections",
            "tracking": self.run_dir / "04_tracking",
            "integrity": self.run_dir / "05_integrity",
            "reconciliation": self.run_dir / "06_reconciliation",
            "representative_frames": self.run_dir / "07_representative_frames",
            "identity_packages": self.run_dir / "08_identity_packages",
            "reports": self.run_dir / "09_reports",
            "logs": self.run_dir / "logs",
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["output_dirs"] = {key: str(value) for key, value in self.output_dirs.items()}
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated continuous MOT hybrid experiment.")
    parser.add_argument("--video-path")
    parser.add_argument("--run-dir")
    parser.add_argument("--camera-id")
    parser.add_argument("--camera-group")
    parser.add_argument("--camera-timezone")
    parser.add_argument("--processing-fps", type=float)
    parser.add_argument("--mot-backend", choices=("bytetrack", "botsort"))
    parser.add_argument("--device")
    parser.add_argument("--roi")
    parser.add_argument("--enable-short-gap-visual-tracker", action="store_true")
    parser.add_argument("--disable-short-gap-visual-tracker", action="store_true")
    return parser


def create_default_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root() / "debug_runs" / f"continuous_mot_hybrid_{video_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_output_dirs(config: ContinuousMotHybridConfig) -> None:
    for directory in config.output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)


def resolve_config(args: argparse.Namespace) -> ContinuousMotHybridConfig:
    raw_video_path = args.video_path or os.environ.get(ENV_VIDEO_PATH)
    if not raw_video_path:
        raise ValueError(f"{ENV_VIDEO_PATH} or --video-path is required.")
    video_path = Path(str(raw_video_path)).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    raw_run_dir = args.run_dir or os.environ.get(ENV_RUN_DIR)
    run_dir = Path(str(raw_run_dir)).expanduser().resolve() if raw_run_dir else create_default_run_dir(video_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    enable_short_gap_visual_tracker = _read_bool(os.environ.get(ENV_ENABLE_SHORT_GAP_VISUAL_TRACKER), True)
    if args.enable_short_gap_visual_tracker:
        enable_short_gap_visual_tracker = True
    if args.disable_short_gap_visual_tracker:
        enable_short_gap_visual_tracker = False

    config = ContinuousMotHybridConfig(
        video_path=video_path,
        run_dir=run_dir,
        camera_id=str(args.camera_id or os.environ.get(ENV_CAMERA_ID) or "test_cam_01"),
        camera_group=str(args.camera_group or os.environ.get(ENV_CAMERA_GROUP) or "single_camera_comparison"),
        camera_timezone=str(args.camera_timezone or os.environ.get(ENV_CAMERA_TIMEZONE) or "Asia/Kolkata"),
        processing_fps=_read_float(args.processing_fps or os.environ.get(ENV_PROCESSING_FPS), 10.0),
        mot_backend=str(args.mot_backend or os.environ.get(ENV_MOT_BACKEND) or "bytetrack").lower(),
        device=str(args.device or os.environ.get(ENV_DEVICE) or "auto"),
        roi=str(args.roi or os.environ.get(ENV_ROI)).strip() if (args.roi or os.environ.get(ENV_ROI)) else None,
        enable_short_gap_visual_tracker=enable_short_gap_visual_tracker,
        short_gap_visual_tracker=str(os.environ.get(ENV_SHORT_GAP_VISUAL_TRACKER) or "csrt").lower(),
        detector_normal_interval_seconds=_read_float(os.environ.get(ENV_DETECTOR_NORMAL_INTERVAL), 0.2),
        detector_sparse_interval_seconds=_read_float(os.environ.get(ENV_DETECTOR_SPARSE_INTERVAL), 0.3),
        detector_idle_interval_seconds=_read_float(os.environ.get(ENV_DETECTOR_IDLE_INTERVAL), 0.5),
        detector_max_gap_seconds=_read_float(os.environ.get(ENV_DETECTOR_MAX_GAP), 0.5),
        visual_bridge_max_seconds=_read_float(os.environ.get(ENV_VISUAL_BRIDGE_MAX_SECONDS), 0.3),
        lost_recovery_seconds=_read_float(os.environ.get(ENV_LOST_RECOVERY_SECONDS), 1.0),
        min_person_confirm_hits=_read_int(os.environ.get(ENV_MIN_PERSON_CONFIRM_HITS), 3),
        min_vehicle_confirm_hits=_read_int(os.environ.get(ENV_MIN_VEHICLE_CONFIRM_HITS), 2),
        save_debug_frames=_read_bool(os.environ.get(ENV_SAVE_DEBUG_FRAMES), False),
        yolo_confidence=_read_float(os.environ.get(ENV_YOLO_CONFIDENCE), 0.25),
        yolo_iou=_read_float(os.environ.get(ENV_YOLO_IOU), 0.45),
        track_high_thresh=_read_float(os.environ.get(ENV_TRACK_HIGH_THRESH), 0.30),
        track_low_thresh=_read_float(os.environ.get(ENV_TRACK_LOW_THRESH), 0.10),
        match_thresh=_read_float(os.environ.get(ENV_MATCH_THRESH), 0.80),
        track_buffer_seconds=_read_float(os.environ.get(ENV_TRACK_BUFFER_SECONDS), 1.0),
        duplicate_overlap_seconds=_read_float(os.environ.get(ENV_DUPLICATE_OVERLAP_SECONDS), 0.5),
        person_model_path=_resolve_optional_case_path(_read_path_env(ENV_PERSON_YOLO_MODEL_PATH)),
        object_model_path=_resolve_optional_case_path(_read_path_env(ENV_OBJECT_YOLO_MODEL_PATH)),
        combined_model_path=_resolve_optional_case_path(_read_path_env(ENV_YOLO_MODEL_PATH)),
    )
    ensure_output_dirs(config)
    return config
