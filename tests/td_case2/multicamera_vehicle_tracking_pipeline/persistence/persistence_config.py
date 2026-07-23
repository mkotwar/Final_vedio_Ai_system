from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SUPPORTED_OBSERVATION_MODES = ("all", "sampled", "none")
SUPPORTED_PERSISTENCE_BACKENDS = ("disabled", "dry_run", "old_public", "analytics_supabase")
SUPPORTED_TRACK_MEDIA_ROLES = (
    "BEST_OVERALL",
    "FIRST",
    "MIDDLE",
    "LAST",
    "HIGHEST_CONFIDENCE",
    "LARGEST",
    "SHARPEST",
)


class PersistenceConfigError(ValueError):
    """Raised when persistence configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    backend: str = "disabled"
    enabled: bool = False
    sync_cameras: bool = True
    write_completed_tracks_only: bool = True
    include_discarded_tracks: bool = False
    observation_mode: str = "all"
    observation_batch_size: int = 100
    observation_sample_every_n: int = 5
    dry_run: bool = False
    fail_on_database_error: bool = True
    persist_track_media: bool = True
    track_media_roles: tuple[str, ...] = ("BEST_OVERALL",)
    fail_pipeline_on_track_media_persistence_error: bool = False

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in SUPPORTED_PERSISTENCE_BACKENDS:
            raise PersistenceConfigError(f"Unsupported persistence backend: {self.backend}")
        if backend == "disabled" and self.enabled:
            backend = "dry_run" if self.dry_run else "old_public"
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "enabled", backend != "disabled")
        object.__setattr__(self, "dry_run", backend == "dry_run")
        if self.observation_mode not in SUPPORTED_OBSERVATION_MODES:
            raise PersistenceConfigError(f"Unsupported observation_mode: {self.observation_mode}")
        if int(self.observation_batch_size) <= 0:
            raise PersistenceConfigError("observation_batch_size must be positive.")
        if int(self.observation_sample_every_n) <= 0:
            raise PersistenceConfigError("observation_sample_every_n must be positive.")
        normalized_roles: list[str] = []
        seen_roles: set[str] = set()
        for role in self.track_media_roles:
            normalized_role = str(role).strip().upper()
            if normalized_role not in SUPPORTED_TRACK_MEDIA_ROLES:
                raise PersistenceConfigError(f"Unsupported track media role: {role}")
            if normalized_role in seen_roles:
                continue
            seen_roles.add(normalized_role)
            normalized_roles.append(normalized_role)
        object.__setattr__(self, "track_media_roles", tuple(normalized_roles))
        if self.persist_track_media and not self.track_media_roles:
            raise PersistenceConfigError("track_media_roles must not be empty when persist_track_media is enabled.")


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
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
            raise PersistenceConfigError("Persistence config root must be a mapping.")
        return payload
    payload: dict[str, Any] = {"persistence": {}}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "persistence:":
            continue
        key, _, value = stripped.partition(":")
        if not _:
            raise PersistenceConfigError("Invalid persistence.yaml structure.")
        payload["persistence"][key.strip()] = _parse_scalar(value)
    return payload


def load_persistence_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> PersistenceConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise PersistenceConfigError(f"Persistence config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw = payload.get("persistence")
    if not isinstance(raw, dict):
        raise PersistenceConfigError("Persistence config must contain a 'persistence' mapping.")
    backend = str(raw.get("backend", "")).strip().lower()
    if not backend:
        enabled = bool(raw.get("enabled", False))
        dry_run = bool(raw.get("dry_run", False))
        backend = "disabled"
        if enabled:
            backend = "dry_run" if dry_run else "old_public"
    config = PersistenceConfig(
        backend=backend,
        enabled=bool(raw.get("enabled", False)),
        sync_cameras=bool(raw.get("sync_cameras", True)),
        write_completed_tracks_only=bool(raw.get("write_completed_tracks_only", True)),
        include_discarded_tracks=bool(raw.get("include_discarded_tracks", False)),
        observation_mode=str(raw.get("observation_mode", "all")),
        observation_batch_size=int(raw.get("observation_batch_size", 100)),
        observation_sample_every_n=int(raw.get("observation_sample_every_n", 5)),
        dry_run=bool(raw.get("dry_run", False)),
        fail_on_database_error=bool(raw.get("fail_on_database_error", True)),
        persist_track_media=bool(raw.get("persist_track_media", True)),
        track_media_roles=tuple(raw.get("track_media_roles", ("BEST_OVERALL",))),
        fail_pipeline_on_track_media_persistence_error=bool(raw.get("fail_pipeline_on_track_media_persistence_error", False)),
    )
    if overrides:
        override_backend = overrides.get("backend", config.backend)
        override_enabled = bool(overrides.get("enabled", config.enabled))
        override_dry_run = bool(overrides.get("dry_run", config.dry_run))
        if "backend" not in overrides:
            override_backend = "disabled"
            if override_enabled:
                override_backend = "dry_run" if override_dry_run else "old_public"
        config = replace(
            config,
            backend=str(override_backend),
            enabled=override_enabled,
            sync_cameras=bool(overrides.get("sync_cameras", config.sync_cameras)),
            write_completed_tracks_only=bool(overrides.get("write_completed_tracks_only", config.write_completed_tracks_only)),
            include_discarded_tracks=bool(overrides.get("include_discarded_tracks", config.include_discarded_tracks)),
            observation_mode=str(overrides.get("observation_mode", config.observation_mode)),
            observation_batch_size=int(overrides.get("observation_batch_size", config.observation_batch_size)),
            observation_sample_every_n=int(overrides.get("observation_sample_every_n", config.observation_sample_every_n)),
            dry_run=override_dry_run,
            fail_on_database_error=bool(overrides.get("fail_on_database_error", config.fail_on_database_error)),
            persist_track_media=bool(overrides.get("persist_track_media", config.persist_track_media)),
            track_media_roles=tuple(overrides.get("track_media_roles", config.track_media_roles)),
            fail_pipeline_on_track_media_persistence_error=bool(
                overrides.get("fail_pipeline_on_track_media_persistence_error", config.fail_pipeline_on_track_media_persistence_error)
            ),
        )
    return config


def persistence_overrides_from_env() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    env_map = {
        "TD_CASE2_MULTICAM_PERSISTENCE_BACKEND": ("backend", str),
        "TD_CASE2_MULTICAM_PERSISTENCE_ENABLED": ("enabled", lambda value: value.lower() == "true"),
        "TD_CASE2_MULTICAM_PERSISTENCE_DRY_RUN": ("dry_run", lambda value: value.lower() == "true"),
        "TD_CASE2_MULTICAM_INCLUDE_DISCARDED": ("include_discarded_tracks", lambda value: value.lower() == "true"),
        "TD_CASE2_MULTICAM_OBSERVATION_MODE": ("observation_mode", str),
        "TD_CASE2_MULTICAM_OBSERVATION_BATCH_SIZE": ("observation_batch_size", int),
        "TD_CASE2_MULTICAM_OBSERVATION_SAMPLE_EVERY_N": ("observation_sample_every_n", int),
        "TD_CASE2_MULTICAM_FAIL_ON_DATABASE_ERROR": ("fail_on_database_error", lambda value: value.lower() == "true"),
        "TD_CASE2_MULTICAM_PERSIST_TRACK_MEDIA": ("persist_track_media", lambda value: value.lower() == "true"),
        "TD_CASE2_MULTICAM_FAIL_ON_TRACK_MEDIA_ERROR": ("fail_pipeline_on_track_media_persistence_error", lambda value: value.lower() == "true"),
    }
    for env_name, (target_name, caster) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[target_name] = caster(raw_value)
    return overrides
