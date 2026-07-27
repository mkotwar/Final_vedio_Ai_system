from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


SUPPORTED_TRACKING_BACKENDS = ("supervision_bytetrack", "ultralytics_bytetrack")


class TrackingConfigError(ValueError):
    """Raised when the tracking configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ClassStabilizationConfig:
    enabled: bool = True
    strategy: str = "confidence_weighted_vote"
    observation_count_weight: float = 0.10
    max_confidence_weight: float = 0.25
    minimum_observations: int = 2
    minimum_winner_margin: float = 0.20
    lock_after_observations: int = 5
    allow_unlock_on_strong_conflict: bool = True
    strong_conflict_min_observations: int = 3
    strong_conflict_margin: float = 0.75

    def __post_init__(self) -> None:
        if int(self.minimum_observations) <= 0:
            raise TrackingConfigError("class_stabilization.minimum_observations must be positive.")
        if int(self.lock_after_observations) <= 0:
            raise TrackingConfigError("class_stabilization.lock_after_observations must be positive.")
        if int(self.strong_conflict_min_observations) <= 0:
            raise TrackingConfigError("class_stabilization.strong_conflict_min_observations must be positive.")
        for field_name in ("observation_count_weight", "max_confidence_weight", "minimum_winner_margin", "strong_conflict_margin"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise TrackingConfigError(f"class_stabilization.{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class FragmentLinkingConfig:
    enabled: bool = False
    maximum_gap_seconds: float = 1.5
    minimum_spatial_score: float = 0.55
    minimum_class_compatibility: float = 0.50
    require_no_time_overlap: bool = True
    reject_verified_plate_conflict: bool = True

    def __post_init__(self) -> None:
        if float(self.maximum_gap_seconds) < 0.0:
            raise TrackingConfigError("fragment_linking.maximum_gap_seconds must be non-negative.")
        for field_name in ("minimum_spatial_score", "minimum_class_compatibility"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise TrackingConfigError(f"fragment_linking.{field_name} must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class IdentityContinuityConfig:
    enabled: bool = True
    minimum_spatial_score: float = 0.20
    minimum_class_compatibility: float = 0.50
    maximum_area_ratio: float = 3.50
    hard_split_spatial_score: float = 0.08

    def __post_init__(self) -> None:
        for field_name in ("minimum_spatial_score", "minimum_class_compatibility", "hard_split_spatial_score"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise TrackingConfigError(f"identity_continuity.{field_name} must be between 0 and 1.")
        if float(self.maximum_area_ratio) < 1.0:
            raise TrackingConfigError("identity_continuity.maximum_area_ratio must be at least 1.")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    backend: str = "supervision_bytetrack"
    track_high_thresh: float = 0.15
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.30
    match_thresh: float = 0.80
    track_buffer: int = 30
    track_activation_threshold: float = 0.15
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.80
    frame_rate: float | None = None
    minimum_consecutive_frames: int = 1
    min_confirmed_observations: int = 3
    max_lost_frames: int = 30
    preserve_state_per_camera: bool = True
    class_stabilization: ClassStabilizationConfig = field(default_factory=ClassStabilizationConfig)
    class_aliases: dict[str, str] = field(default_factory=lambda: {
        "3 wheeler": "3wheeler",
        "3-wheeler": "3wheeler",
        "three_wheeler": "3wheeler",
        "three wheeler": "3wheeler",
        "auto": "3wheeler",
        "auto rickshaw": "3wheeler",
        "auto_rickshaw": "3wheeler",
        "rickshaw": "3wheeler",
    })
    class_families: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "light_passenger": ("car", "3wheeler"),
        "heavy_vehicle": ("bus", "truck"),
    })
    fragment_linking: FragmentLinkingConfig = field(default_factory=FragmentLinkingConfig)
    identity_continuity: IdentityContinuityConfig = field(default_factory=IdentityContinuityConfig)

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_TRACKING_BACKENDS:
            raise TrackingConfigError(f"Unsupported tracking backend: {self.backend}")
        for field_name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise TrackingConfigError(f"{field_name} must be between 0 and 1.")
        if int(self.track_buffer) <= 0:
            raise TrackingConfigError("track_buffer must be positive.")
        if int(self.lost_track_buffer) <= 0:
            raise TrackingConfigError("lost_track_buffer must be positive.")
        if self.frame_rate is not None and float(self.frame_rate) <= 0.0:
            raise TrackingConfigError("frame_rate must be positive when provided.")
        if int(self.minimum_consecutive_frames) <= 0:
            raise TrackingConfigError("minimum_consecutive_frames must be positive.")
        if int(self.min_confirmed_observations) <= 0:
            raise TrackingConfigError("min_confirmed_observations must be positive.")
        if int(self.max_lost_frames) < 0:
            raise TrackingConfigError("max_lost_frames must be non-negative.")
        normalized_aliases = {_normalize_alias_key(key): str(value).strip().lower() for key, value in self.class_aliases.items() if str(key).strip()}
        normalized_families = {
            str(name).strip(): tuple(str(member).strip().lower() for member in members if str(member).strip())
            for name, members in self.class_families.items()
            if str(name).strip()
        }
        object.__setattr__(self, "class_aliases", normalized_aliases)
        object.__setattr__(self, "class_families", normalized_families)


def _normalize_alias_key(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


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


def _normalize_tracking_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    if "track_activation_threshold" not in normalized and "track_high_thresh" in normalized:
        normalized["track_activation_threshold"] = normalized["track_high_thresh"]
    if "lost_track_buffer" not in normalized and "track_buffer" in normalized:
        normalized["lost_track_buffer"] = normalized["track_buffer"]
    if "minimum_matching_threshold" not in normalized and "match_thresh" in normalized:
        normalized["minimum_matching_threshold"] = normalized["match_thresh"]
    return normalized


def load_tracking_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> TrackingConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise TrackingConfigError(f"Tracking config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw_config = payload.get("tracking")
    if not isinstance(raw_config, dict):
        raise TrackingConfigError("Tracking config must contain a 'tracking' mapping.")
    config = TrackingConfig(
        backend=str(raw_config.get("backend", "supervision_bytetrack")),
        track_high_thresh=float(raw_config.get("track_high_thresh", raw_config.get("track_activation_threshold", 0.15))),
        track_low_thresh=float(raw_config.get("track_low_thresh", 0.10)),
        new_track_thresh=float(raw_config.get("new_track_thresh", 0.30)),
        match_thresh=float(raw_config.get("match_thresh", 0.80)),
        track_buffer=int(raw_config.get("track_buffer", 30)),
        track_activation_threshold=float(raw_config.get("track_activation_threshold", raw_config.get("track_high_thresh", 0.15))),
        lost_track_buffer=int(raw_config.get("lost_track_buffer", raw_config.get("track_buffer", 30))),
        minimum_matching_threshold=float(raw_config.get("minimum_matching_threshold", raw_config.get("match_thresh", 0.80))),
        frame_rate=float(raw_config["frame_rate"]) if raw_config.get("frame_rate") is not None else None,
        minimum_consecutive_frames=int(raw_config.get("minimum_consecutive_frames", 1)),
        min_confirmed_observations=int(raw_config.get("min_confirmed_observations", 3)),
        max_lost_frames=int(raw_config.get("max_lost_frames", 30)),
        preserve_state_per_camera=bool(raw_config.get("preserve_state_per_camera", True)),
        class_stabilization=ClassStabilizationConfig(**dict(raw_config.get("class_stabilization", {}) or {})),
        class_aliases={str(key): str(value) for key, value in dict(raw_config.get("class_aliases", {}) or {}).items()},
        class_families={
            str(key): tuple(str(item) for item in value)
            for key, value in dict(raw_config.get("class_families", {}) or {}).items()
            if isinstance(value, (list, tuple))
        },
        fragment_linking=FragmentLinkingConfig(**dict(raw_config.get("fragment_linking", {}) or {})),
        identity_continuity=IdentityContinuityConfig(**dict(raw_config.get("identity_continuity", {}) or {})),
    )
    if overrides:
        overrides = _normalize_tracking_overrides(overrides)
        config = replace(
            config,
            backend=str(overrides.get("backend", config.backend)),
            track_high_thresh=float(overrides.get("track_high_thresh", config.track_high_thresh)),
            track_low_thresh=float(overrides.get("track_low_thresh", config.track_low_thresh)),
            new_track_thresh=float(overrides.get("new_track_thresh", config.new_track_thresh)),
            match_thresh=float(overrides.get("match_thresh", config.match_thresh)),
            track_buffer=int(overrides.get("track_buffer", config.track_buffer)),
            track_activation_threshold=float(overrides.get("track_activation_threshold", config.track_activation_threshold)),
            lost_track_buffer=int(overrides.get("lost_track_buffer", config.lost_track_buffer)),
            minimum_matching_threshold=float(overrides.get("minimum_matching_threshold", config.minimum_matching_threshold)),
            frame_rate=float(overrides["frame_rate"]) if overrides.get("frame_rate") is not None else config.frame_rate,
            minimum_consecutive_frames=int(overrides.get("minimum_consecutive_frames", config.minimum_consecutive_frames)),
            min_confirmed_observations=int(overrides.get("min_confirmed_observations", config.min_confirmed_observations)),
            max_lost_frames=int(overrides.get("max_lost_frames", config.max_lost_frames)),
            preserve_state_per_camera=bool(overrides.get("preserve_state_per_camera", config.preserve_state_per_camera)),
            class_stabilization=ClassStabilizationConfig(**dict(overrides.get("class_stabilization", asdict(config.class_stabilization)) or {})),
            class_aliases={str(key): str(value) for key, value in dict(overrides.get("class_aliases", config.class_aliases) or {}).items()},
            class_families={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(overrides.get("class_families", config.class_families) or {}).items()
                if isinstance(value, (list, tuple))
            },
            fragment_linking=FragmentLinkingConfig(**dict(overrides.get("fragment_linking", asdict(config.fragment_linking)) or {})),
            identity_continuity=IdentityContinuityConfig(**dict(overrides.get("identity_continuity", asdict(config.identity_continuity)) or {})),
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
        "TD_CASE2_MULTICAM_TRACK_ACTIVATION_THRESHOLD": ("track_activation_threshold", float),
        "TD_CASE2_MULTICAM_LOST_TRACK_BUFFER": ("lost_track_buffer", int),
        "TD_CASE2_MULTICAM_MINIMUM_MATCHING_THRESHOLD": ("minimum_matching_threshold", float),
        "TD_CASE2_MULTICAM_FRAME_RATE": ("frame_rate", float),
        "TD_CASE2_MULTICAM_MINIMUM_CONSECUTIVE_FRAMES": ("minimum_consecutive_frames", int),
        "TD_CASE2_MULTICAM_MIN_CONFIRMED_OBSERVATIONS": ("min_confirmed_observations", int),
        "TD_CASE2_MULTICAM_MAX_LOST_FRAMES": ("max_lost_frames", int),
    }
    for env_name, (target_name, caster) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[target_name] = caster(raw_value)
    return overrides
