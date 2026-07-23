from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class WorkerConfigError(ValueError):
    """Raised when worker configuration is invalid."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    enabled: bool = False
    frame_queue_size: int = 20
    detection_queue_size: int = 20
    completed_track_queue_size: int = 20
    error_queue_size: int = 20
    camera_reader_daemon: bool = False
    detection_worker_daemon: bool = False
    tracking_worker_daemon: bool = False
    persistence_worker_daemon: bool = False
    vehicle_colour_worker_daemon: bool = False
    queue_put_timeout_seconds: float = 2.0
    queue_get_timeout_seconds: float = 1.0
    shutdown_timeout_seconds: float = 30.0
    stop_on_camera_error: bool = False
    stop_on_detector_error: bool = True
    stop_on_tracking_error: bool = True
    stop_on_persistence_error: bool = False
    enable_persistence_worker: bool = False
    enable_vehicle_colour_worker: bool = False
    vehicle_colour_queue_size: int = 20
    vehicle_colour_worker_count: int = 1
    enable_anpr_worker: bool = False
    anpr_queue_size: int = 20
    anpr_worker_count: int = 1
    anpr_worker_daemon: bool = False

    def __post_init__(self) -> None:
        for field_name in ("frame_queue_size", "detection_queue_size", "completed_track_queue_size", "error_queue_size", "vehicle_colour_queue_size", "vehicle_colour_worker_count", "anpr_queue_size", "anpr_worker_count"):
            if int(getattr(self, field_name)) <= 0:
                raise WorkerConfigError(f"{field_name} must be positive.")
        for field_name in ("queue_put_timeout_seconds", "queue_get_timeout_seconds", "shutdown_timeout_seconds"):
            if float(getattr(self, field_name)) <= 0.0:
                raise WorkerConfigError(f"{field_name} must be positive.")


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped.strip('"')


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise WorkerConfigError("Worker config root must be a mapping.")
        return payload
    payload: dict[str, Any] = {"workers": {}}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "workers:":
            continue
        key, _, value = stripped.partition(":")
        if not _:
            raise WorkerConfigError("Invalid workers.yaml structure.")
        payload["workers"][key.strip()] = _parse_scalar(value)
    return payload


def load_worker_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> WorkerConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise WorkerConfigError(f"Worker config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw = payload.get("workers")
    if not isinstance(raw, dict):
        raise WorkerConfigError("Worker config must contain a 'workers' mapping.")
    config = WorkerConfig(
        enabled=bool(raw.get("enabled", False)),
        frame_queue_size=int(raw.get("frame_queue_size", 20)),
        detection_queue_size=int(raw.get("detection_queue_size", 20)),
        completed_track_queue_size=int(raw.get("completed_track_queue_size", 50)),
        error_queue_size=int(raw.get("error_queue_size", 50)),
        camera_reader_daemon=bool(raw.get("camera_reader_daemon", False)),
        detection_worker_daemon=bool(raw.get("detection_worker_daemon", False)),
        tracking_worker_daemon=bool(raw.get("tracking_worker_daemon", False)),
        persistence_worker_daemon=bool(raw.get("persistence_worker_daemon", False)),
        vehicle_colour_worker_daemon=bool(raw.get("vehicle_colour_worker_daemon", False)),
        queue_put_timeout_seconds=float(raw.get("queue_put_timeout_seconds", 5.0)),
        queue_get_timeout_seconds=float(raw.get("queue_get_timeout_seconds", 1.0)),
        shutdown_timeout_seconds=float(raw.get("shutdown_timeout_seconds", 30.0)),
        stop_on_camera_error=bool(raw.get("stop_on_camera_error", False)),
        stop_on_detector_error=bool(raw.get("stop_on_detector_error", True)),
        stop_on_tracking_error=bool(raw.get("stop_on_tracking_error", True)),
        stop_on_persistence_error=bool(raw.get("stop_on_persistence_error", False)),
        enable_persistence_worker=bool(raw.get("enable_persistence_worker", raw.get("persist_completed_tracks", False))),
        enable_vehicle_colour_worker=bool(raw.get("enable_vehicle_colour_worker", False)),
        vehicle_colour_queue_size=int(raw.get("vehicle_colour_queue_size", 20)),
        vehicle_colour_worker_count=int(raw.get("vehicle_colour_worker_count", 1)),
        enable_anpr_worker=bool(raw.get("enable_anpr_worker", False)),
        anpr_queue_size=int(raw.get("anpr_queue_size", 20)),
        anpr_worker_count=int(raw.get("anpr_worker_count", 1)),
        anpr_worker_daemon=bool(raw.get("anpr_worker_daemon", False)),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            frame_queue_size=int(overrides.get("frame_queue_size", config.frame_queue_size)),
            detection_queue_size=int(overrides.get("detection_queue_size", config.detection_queue_size)),
            completed_track_queue_size=int(overrides.get("completed_track_queue_size", config.completed_track_queue_size)),
            error_queue_size=int(overrides.get("error_queue_size", config.error_queue_size)),
            camera_reader_daemon=bool(overrides.get("camera_reader_daemon", config.camera_reader_daemon)),
            detection_worker_daemon=bool(overrides.get("detection_worker_daemon", config.detection_worker_daemon)),
            tracking_worker_daemon=bool(overrides.get("tracking_worker_daemon", config.tracking_worker_daemon)),
            persistence_worker_daemon=bool(overrides.get("persistence_worker_daemon", config.persistence_worker_daemon)),
            vehicle_colour_worker_daemon=bool(overrides.get("vehicle_colour_worker_daemon", config.vehicle_colour_worker_daemon)),
            queue_put_timeout_seconds=float(overrides.get("queue_put_timeout_seconds", config.queue_put_timeout_seconds)),
            queue_get_timeout_seconds=float(overrides.get("queue_get_timeout_seconds", config.queue_get_timeout_seconds)),
            shutdown_timeout_seconds=float(overrides.get("shutdown_timeout_seconds", config.shutdown_timeout_seconds)),
            stop_on_camera_error=bool(overrides.get("stop_on_camera_error", config.stop_on_camera_error)),
            stop_on_detector_error=bool(overrides.get("stop_on_detector_error", config.stop_on_detector_error)),
            stop_on_tracking_error=bool(overrides.get("stop_on_tracking_error", config.stop_on_tracking_error)),
            stop_on_persistence_error=bool(overrides.get("stop_on_persistence_error", config.stop_on_persistence_error)),
            enable_persistence_worker=bool(overrides.get("enable_persistence_worker", overrides.get("persist_completed_tracks", config.enable_persistence_worker))),
            enable_vehicle_colour_worker=bool(overrides.get("enable_vehicle_colour_worker", config.enable_vehicle_colour_worker)),
            vehicle_colour_queue_size=int(overrides.get("vehicle_colour_queue_size", config.vehicle_colour_queue_size)),
            vehicle_colour_worker_count=int(overrides.get("vehicle_colour_worker_count", config.vehicle_colour_worker_count)),
            enable_anpr_worker=bool(overrides.get("enable_anpr_worker", config.enable_anpr_worker)),
            anpr_queue_size=int(overrides.get("anpr_queue_size", config.anpr_queue_size)),
            anpr_worker_count=int(overrides.get("anpr_worker_count", config.anpr_worker_count)),
            anpr_worker_daemon=bool(overrides.get("anpr_worker_daemon", config.anpr_worker_daemon)),
        )
    return config
