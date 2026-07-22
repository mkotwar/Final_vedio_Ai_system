from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SUPPORTED_TRACKING_BACKENDS = ("supervision_bytetrack", "ultralytics_bytetrack")


class TrackingConfigError(ValueError):
    """Raised when the tracking configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    backend: str = "ultralytics_bytetrack"
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.30
    match_thresh: float = 0.80
    track_buffer: int = 30
    min_confirmed_observations: int = 3
    max_lost_frames: int = 30
    preserve_state_per_camera: bool = True

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_TRACKING_BACKENDS:
            raise TrackingConfigError(f"Unsupported tracking backend: {self.backend}")
        for field_name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise TrackingConfigError(f"{field_name} must be between 0 and 1.")
        if int(self.track_buffer) <= 0:
            raise TrackingConfigError("track_buffer must be positive.")
        if int(self.min_confirmed_observations) <= 0:
            raise TrackingConfigError("min_confirmed_observations must be positive.")
        if int(self.max_lost_frames) < 0:
            raise TrackingConfigError("max_lost_frames must be non-negative.")


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
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
        return stripped


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise TrackingConfigError("Tracking config root must be a mapping.")
        return payload

    payload: dict[str, Any] = {"tracking": {}}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "tracking:":
            continue
        key, _, value = stripped.partition(":")
        if not _:
            raise TrackingConfigError("Invalid tracking.yaml structure.")
        payload["tracking"][key.strip()] = _parse_scalar(value)
    return payload


def load_tracking_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> TrackingConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise TrackingConfigError(f"Tracking config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw_config = payload.get("tracking")
    if not isinstance(raw_config, dict):
        raise TrackingConfigError("Tracking config must contain a 'tracking' mapping.")
    config = TrackingConfig(
        backend=str(raw_config.get("backend", "ultralytics_bytetrack")),
        track_high_thresh=float(raw_config.get("track_high_thresh", 0.30)),
        track_low_thresh=float(raw_config.get("track_low_thresh", 0.10)),
        new_track_thresh=float(raw_config.get("new_track_thresh", 0.30)),
        match_thresh=float(raw_config.get("match_thresh", 0.80)),
        track_buffer=int(raw_config.get("track_buffer", 30)),
        min_confirmed_observations=int(raw_config.get("min_confirmed_observations", 3)),
        max_lost_frames=int(raw_config.get("max_lost_frames", 30)),
        preserve_state_per_camera=bool(raw_config.get("preserve_state_per_camera", True)),
    )
    if overrides:
        config = replace(
            config,
            backend=str(overrides.get("backend", config.backend)),
            track_high_thresh=float(overrides.get("track_high_thresh", config.track_high_thresh)),
            track_low_thresh=float(overrides.get("track_low_thresh", config.track_low_thresh)),
            new_track_thresh=float(overrides.get("new_track_thresh", config.new_track_thresh)),
            match_thresh=float(overrides.get("match_thresh", config.match_thresh)),
            track_buffer=int(overrides.get("track_buffer", config.track_buffer)),
            min_confirmed_observations=int(overrides.get("min_confirmed_observations", config.min_confirmed_observations)),
            max_lost_frames=int(overrides.get("max_lost_frames", config.max_lost_frames)),
            preserve_state_per_camera=bool(overrides.get("preserve_state_per_camera", config.preserve_state_per_camera)),
        )
    return config


def tracking_overrides_from_env() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    env_map = {
        "TD_CASE2_MULTICAM_TRACKING_BACKEND": ("backend", str),
        "TD_CASE2_MULTICAM_TRACK_HIGH_THRESH": ("track_high_thresh", float),
        "TD_CASE2_MULTICAM_TRACK_LOW_THRESH": ("track_low_thresh", float),
        "TD_CASE2_MULTICAM_NEW_TRACK_THRESH": ("new_track_thresh", float),
        "TD_CASE2_MULTICAM_MATCH_THRESH": ("match_thresh", float),
        "TD_CASE2_MULTICAM_TRACK_BUFFER": ("track_buffer", int),
        "TD_CASE2_MULTICAM_MIN_CONFIRMED_OBSERVATIONS": ("min_confirmed_observations", int),
        "TD_CASE2_MULTICAM_MAX_LOST_FRAMES": ("max_lost_frames", int),
    }
    for env_name, (target_name, caster) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[target_name] = caster(raw_value)
    return overrides
