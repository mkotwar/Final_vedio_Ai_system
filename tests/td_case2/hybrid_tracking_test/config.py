from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"
ENV_VIDEO_PATH = "TD_CASE2_VIDEO_PATH"
ENV_DEVICE = "TD_CASE2_HYBRID_DEVICE"
ENV_PROCESSING_FPS = "TD_CASE2_HYBRID_PROCESSING_FPS"
ENV_YOLO_INTERVAL_FRAMES = "TD_CASE2_HYBRID_YOLO_INTERVAL_FRAMES"
ENV_MAX_YOLO_GAP_SECONDS = "TD_CASE2_HYBRID_MAX_YOLO_GAP_SECONDS"
ENV_DETECTION_CONFIDENCE = "TD_CASE2_HYBRID_DETECTION_CONFIDENCE"
ENV_MIN_IOU_MATCH = "TD_CASE2_HYBRID_MIN_IOU_MATCH"
ENV_MAX_MISSED_REFRESHES = "TD_CASE2_HYBRID_MAX_MISSED_REFRESHES"
ENV_MAX_TRACK_IDLE_SECONDS = "TD_CASE2_HYBRID_MAX_TRACK_IDLE_SECONDS"
ENV_MIN_TRACK_HITS = "TD_CASE2_HYBRID_MIN_TRACK_HITS"
ENV_ENABLE_MOTION_TRIGGER = "TD_CASE2_HYBRID_ENABLE_MOTION_TRIGGER"
ENV_ENABLE_ENTRY_TRIGGER = "TD_CASE2_HYBRID_ENABLE_ENTRY_TRIGGER"
ENV_ENABLE_OVERLAP_TRIGGER = "TD_CASE2_HYBRID_ENABLE_OVERLAP_TRIGGER"
ENV_SAVE_VIDEO = "TD_CASE2_HYBRID_SAVE_VIDEO"
ENV_ENTRY_ZONES_JSON = "TD_CASE2_HYBRID_ENTRY_ZONES_JSON"
ENV_ENTRY_ZONES_FILE = "TD_CASE2_HYBRID_ENTRY_ZONES_FILE"
ENV_MOTION_MIN_AREA_RATIO = "TD_CASE2_HYBRID_MOTION_MIN_AREA_RATIO"
ENV_MOTION_PERSISTENCE_FRAMES = "TD_CASE2_HYBRID_MOTION_PERSISTENCE_FRAMES"
ENV_MOTION_TRACK_REGION_EXPANSION = "TD_CASE2_HYBRID_MOTION_TRACK_REGION_EXPANSION"
ENV_LOST_RECOVERY_SECONDS = "TD_CASE2_HYBRID_LOST_RECOVERY_SECONDS"
ENV_LOST_MAX_CENTER_DISTANCE_RATIO = "TD_CASE2_HYBRID_LOST_MAX_CENTER_DISTANCE_RATIO"
ENV_LOST_MIN_AREA_RATIO = "TD_CASE2_HYBRID_LOST_MIN_AREA_RATIO"
ENV_LOST_MAX_AREA_RATIO = "TD_CASE2_HYBRID_LOST_MAX_AREA_RATIO"
ENV_EMPTY_SCENE_YOLO_INTERVAL_SECONDS = "TD_CASE2_HYBRID_EMPTY_SCENE_YOLO_INTERVAL_SECONDS"
ENV_CLASS_CONFIDENCE_THRESHOLDS_JSON = "TD_CASE2_HYBRID_CLASS_CONFIDENCE_THRESHOLDS_JSON"


@dataclass(frozen=True)
class HybridTrackingConfig:
    run_dir: Path
    video_path: Path
    output_dir_name: str = "hybrid_tracking_test"
    processing_fps: float = 10.0
    yolo_interval_frames: int = 3
    max_yolo_gap_seconds: float = 0.5
    minimum_detection_confidence: float = 0.35
    minimum_iou_match: float = 0.20
    maximum_missed_yolo_refreshes: int = 8
    visual_tracker: str = "KCF"
    enable_scheduled_refresh: bool = True
    enable_motion_trigger: bool = True
    enable_entry_zone_trigger: bool = True
    enable_overlap_trigger: bool = True
    enable_box_validation_trigger: bool = True
    motion_persistence_frames: int = 3
    motion_min_area_ratio: float = 0.006
    motion_track_region_expansion: float = 0.30
    minimum_track_hits: int = 3
    maximum_track_idle_seconds: float = 2.0
    maximum_center_jump_diagonals: float = 1.5
    minimum_area_ratio_change: float = 0.50
    maximum_area_ratio_change: float = 2.00
    minimum_aspect_ratio_change: float = 0.50
    maximum_aspect_ratio_change: float = 2.00
    maximum_pairwise_overlap_iou: float = 0.70
    lost_track_recovery_seconds: float = 2.0
    lost_track_max_center_distance_ratio: float = 1.5
    lost_track_min_area_ratio: float = 0.40
    lost_track_max_area_ratio: float = 2.50
    empty_scene_yolo_interval_seconds: float = 0.5
    vehicle_class_compatibility_enabled: bool = True
    class_vote_history_enabled: bool = True
    motion_region_cooldown_seconds: float = 0.75
    appearance_alpha: float = 0.35
    save_annotated_video: bool = True
    save_frame_level_json: bool = True
    device: str | None = None
    entry_zones: list[dict[str, float]] = field(default_factory=list)
    class_confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "person": 0.35,
            "car": 0.35,
            "motorcycle": 0.30,
            "bus": 0.40,
            "truck": 0.40,
        }
    )
    trajectory_history_limit: int = 120
    overlap_trigger_cooldown_frames: int = 3
    minimum_visible_area_ratio: float = 0.0002

    @property
    def output_dir(self) -> Path:
        return self.run_dir / self.output_dir_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["video_path"] = str(self.video_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _read_bool(raw_value: Any, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip()
    if text == "":
        return default_value
    normalized = text.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw_value!r}")


def _read_float(raw_value: Any, default_value: float) -> float:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    else:
        text = str(raw_value).strip()
        if text == "":
            return default_value
        value = float(text)
    if value <= 0:
        raise ValueError(f"Expected a positive float, received: {value}")
    return value


def _read_int(raw_value: Any, default_value: int) -> int:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        raise ValueError(f"Expected a positive integer, received boolean: {raw_value}")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        value = int(raw_value)
    else:
        text = str(raw_value).strip()
        if text == "":
            return default_value
        value = int(text)
    if value <= 0:
        raise ValueError(f"Expected a positive integer, received: {value}")
    return value


def _load_entry_zones_from_sources() -> list[dict[str, float]]:
    raw_json = os.environ.get(ENV_ENTRY_ZONES_JSON, "").strip()
    if raw_json:
        payload = json.loads(raw_json)
        if isinstance(payload, list):
            return payload
    raw_file = os.environ.get(ENV_ENTRY_ZONES_FILE, "").strip()
    if raw_file:
        path = Path(raw_file).expanduser()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
    return []


def _load_class_thresholds() -> dict[str, float]:
    raw_json = os.environ.get(ENV_CLASS_CONFIDENCE_THRESHOLDS_JSON, "").strip()
    if not raw_json:
        return {}
    payload = json.loads(raw_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{ENV_CLASS_CONFIDENCE_THRESHOLDS_JSON} must decode to an object.")
    thresholds: dict[str, float] = {}
    for key, value in payload.items():
        thresholds[str(key).lower()] = _read_float(value, 0.0)
    return thresholds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated hybrid td_case2 tracking experiment.")
    parser.add_argument("--video-path")
    parser.add_argument("--run-dir")
    parser.add_argument("--processing-fps", type=float)
    parser.add_argument("--yolo-interval-frames", type=int)
    parser.add_argument("--max-yolo-gap-seconds", type=float)
    parser.add_argument("--minimum-detection-confidence", type=float)
    parser.add_argument("--minimum-iou-match", type=float)
    parser.add_argument("--maximum-missed-yolo-refreshes", type=int)
    parser.add_argument("--maximum-track-idle-seconds", type=float)
    parser.add_argument("--minimum-track-hits", type=int)
    parser.add_argument("--motion-min-area-ratio", type=float)
    parser.add_argument("--motion-persistence-frames", type=int)
    parser.add_argument("--motion-track-region-expansion", type=float)
    parser.add_argument("--lost-track-recovery-seconds", type=float)
    parser.add_argument("--lost-track-max-center-distance-ratio", type=float)
    parser.add_argument("--lost-track-min-area-ratio", type=float)
    parser.add_argument("--lost-track-max-area-ratio", type=float)
    parser.add_argument("--empty-scene-yolo-interval-seconds", type=float)
    parser.add_argument("--device")
    parser.add_argument("--save-annotated-video", action="store_true")
    parser.add_argument("--no-save-annotated-video", action="store_true")
    parser.add_argument("--enable-motion-trigger", action="store_true")
    parser.add_argument("--disable-motion-trigger", action="store_true")
    parser.add_argument("--enable-entry-zone-trigger", action="store_true")
    parser.add_argument("--disable-entry-zone-trigger", action="store_true")
    parser.add_argument("--enable-overlap-trigger", action="store_true")
    parser.add_argument("--disable-overlap-trigger", action="store_true")
    parser.add_argument("--entry-zones-file")
    return parser


def _resolve_path(raw_value: str, *, label: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if label == "run_dir":
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory is not a directory: {path}")
    else:
        if not path.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")
    return path


def _cli_env_value(cli_value: Any, env_name: str) -> Any:
    return cli_value if cli_value is not None else os.environ.get(env_name)


def _arg(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, name, None)


def resolve_config(args: argparse.Namespace) -> HybridTrackingConfig:
    raw_run_dir = _cli_env_value(_arg(args, "run_dir"), ENV_RUN_DIR)
    raw_video_path = _cli_env_value(_arg(args, "video_path"), ENV_VIDEO_PATH)
    if not raw_run_dir:
        raise ValueError(f"{ENV_RUN_DIR} or --run-dir is required.")
    if not raw_video_path:
        raise ValueError(f"{ENV_VIDEO_PATH} or --video-path is required.")

    entry_zones = _load_entry_zones_from_sources()
    if _arg(args, "entry_zones_file"):
        entry_zones_path = Path(str(_arg(args, "entry_zones_file"))).expanduser()
        if not entry_zones_path.exists():
            raise FileNotFoundError(f"Entry-zone file does not exist: {entry_zones_path}")
        payload = json.loads(entry_zones_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            entry_zones = payload

    save_annotated_video = _read_bool(os.environ.get(ENV_SAVE_VIDEO), True)
    if _arg(args, "save_annotated_video"):
        save_annotated_video = True
    if _arg(args, "no_save_annotated_video"):
        save_annotated_video = False

    enable_motion_trigger = _read_bool(os.environ.get(ENV_ENABLE_MOTION_TRIGGER), True)
    if _arg(args, "enable_motion_trigger"):
        enable_motion_trigger = True
    if _arg(args, "disable_motion_trigger"):
        enable_motion_trigger = False

    enable_entry_trigger = _read_bool(os.environ.get(ENV_ENABLE_ENTRY_TRIGGER), True)
    if _arg(args, "enable_entry_zone_trigger"):
        enable_entry_trigger = True
    if _arg(args, "disable_entry_zone_trigger"):
        enable_entry_trigger = False

    enable_overlap_trigger = _read_bool(os.environ.get(ENV_ENABLE_OVERLAP_TRIGGER), True)
    if _arg(args, "enable_overlap_trigger"):
        enable_overlap_trigger = True
    if _arg(args, "disable_overlap_trigger"):
        enable_overlap_trigger = False

    class_confidence_thresholds = HybridTrackingConfig(run_dir=Path("."), video_path=Path(".")).class_confidence_thresholds
    env_thresholds = _load_class_thresholds()
    if env_thresholds:
        class_confidence_thresholds = {**class_confidence_thresholds, **env_thresholds}

    return HybridTrackingConfig(
        run_dir=_resolve_path(str(raw_run_dir), label="run_dir"),
        video_path=_resolve_path(str(raw_video_path), label="video_path"),
        processing_fps=_read_float(_cli_env_value(_arg(args, "processing_fps"), ENV_PROCESSING_FPS), 10.0),
        yolo_interval_frames=_read_int(_cli_env_value(_arg(args, "yolo_interval_frames"), ENV_YOLO_INTERVAL_FRAMES), 3),
        max_yolo_gap_seconds=_read_float(_cli_env_value(_arg(args, "max_yolo_gap_seconds"), ENV_MAX_YOLO_GAP_SECONDS), 0.5),
        minimum_detection_confidence=_read_float(_cli_env_value(_arg(args, "minimum_detection_confidence"), ENV_DETECTION_CONFIDENCE), 0.35),
        minimum_iou_match=_read_float(_cli_env_value(_arg(args, "minimum_iou_match"), ENV_MIN_IOU_MATCH), 0.20),
        maximum_missed_yolo_refreshes=_read_int(_cli_env_value(_arg(args, "maximum_missed_yolo_refreshes"), ENV_MAX_MISSED_REFRESHES), 8),
        maximum_track_idle_seconds=_read_float(_cli_env_value(_arg(args, "maximum_track_idle_seconds"), ENV_MAX_TRACK_IDLE_SECONDS), 2.0),
        minimum_track_hits=_read_int(_cli_env_value(_arg(args, "minimum_track_hits"), ENV_MIN_TRACK_HITS), 3),
        motion_min_area_ratio=_read_float(_cli_env_value(_arg(args, "motion_min_area_ratio"), ENV_MOTION_MIN_AREA_RATIO), 0.006),
        motion_persistence_frames=_read_int(_cli_env_value(_arg(args, "motion_persistence_frames"), ENV_MOTION_PERSISTENCE_FRAMES), 3),
        motion_track_region_expansion=_read_float(_cli_env_value(_arg(args, "motion_track_region_expansion"), ENV_MOTION_TRACK_REGION_EXPANSION), 0.30),
        lost_track_recovery_seconds=_read_float(_cli_env_value(_arg(args, "lost_track_recovery_seconds"), ENV_LOST_RECOVERY_SECONDS), 2.0),
        lost_track_max_center_distance_ratio=_read_float(_cli_env_value(_arg(args, "lost_track_max_center_distance_ratio"), ENV_LOST_MAX_CENTER_DISTANCE_RATIO), 1.5),
        lost_track_min_area_ratio=_read_float(_cli_env_value(_arg(args, "lost_track_min_area_ratio"), ENV_LOST_MIN_AREA_RATIO), 0.40),
        lost_track_max_area_ratio=_read_float(_cli_env_value(_arg(args, "lost_track_max_area_ratio"), ENV_LOST_MAX_AREA_RATIO), 2.50),
        empty_scene_yolo_interval_seconds=_read_float(_cli_env_value(_arg(args, "empty_scene_yolo_interval_seconds"), ENV_EMPTY_SCENE_YOLO_INTERVAL_SECONDS), 0.5),
        save_annotated_video=save_annotated_video,
        enable_motion_trigger=enable_motion_trigger,
        enable_entry_zone_trigger=enable_entry_trigger,
        enable_overlap_trigger=enable_overlap_trigger,
        device=str(_cli_env_value(_arg(args, "device"), ENV_DEVICE)).strip() if _cli_env_value(_arg(args, "device"), ENV_DEVICE) else None,
        entry_zones=entry_zones,
        class_confidence_thresholds=class_confidence_thresholds,
    )
