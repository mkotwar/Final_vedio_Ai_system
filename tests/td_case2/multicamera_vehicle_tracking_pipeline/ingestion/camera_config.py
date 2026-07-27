from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


class CameraConfigError(ValueError):
    """Raised when camera configuration is invalid."""


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_code: str
    camera_name: str
    source_path: Path
    enabled: bool
    start_time: datetime | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_source_path(raw_path: str, *, config_path: Path) -> Path:
    candidate = Path(str(raw_path).strip()).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_path.parent.parent / candidate).resolve()


def _parse_start_time(value: Any, *, camera_code: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise CameraConfigError(f"Camera '{camera_code}' has invalid start_time: {value!r}") from exc
    return parsed


def _extract_source_path(raw_item: dict[str, Any], *, index: int) -> str:
    raw_source_path = raw_item.get("source_path")
    if raw_source_path not in (None, ""):
        return str(raw_source_path)
    raw_source = raw_item.get("source")
    if isinstance(raw_source, dict):
        nested_path = raw_source.get("path")
        if nested_path not in (None, ""):
            return str(nested_path)
    raise CameraConfigError(
        "Camera entry at index "
        f"{index} is missing required field 'source_path' or nested field 'source.path'."
    )


def _extract_camera_name(raw_item: dict[str, Any], *, camera_code: str) -> str:
    raw_name = raw_item.get("camera_name")
    if raw_name is None:
        return camera_code
    camera_name = str(raw_name).strip()
    return camera_name or camera_code


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise CameraConfigError("Camera configuration root must be a mapping.")
        return payload

    # Minimal fallback parser for the simple cameras.yaml structure used in tests.
    cameras: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "cameras:":
            continue
        if stripped.startswith("- "):
            if current is not None:
                cameras.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if stripped:
                key, _, value = stripped.partition(":")
                current[key.strip()] = _parse_scalar(value.strip())
            continue
        if current is None:
            raise CameraConfigError("Invalid cameras.yaml structure.")
        key, _, value = stripped.partition(":")
        current[key.strip()] = _parse_scalar(value.strip())
    if current is not None:
        cameras.append(current)
    return {"cameras": cameras}


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def load_camera_configs(config_path: str | Path, *, include_disabled: bool = False, validate_paths: bool = True) -> list[CameraConfig]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise CameraConfigError(f"Camera config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise CameraConfigError("Camera config must contain a non-empty 'cameras' list.")

    configs: list[CameraConfig] = []
    seen_codes: set[str] = set()
    for index, raw_item in enumerate(raw_cameras):
        if not isinstance(raw_item, dict):
            raise CameraConfigError(f"Camera entry at index {index} must be a mapping.")
        for required in ("camera_code", "enabled"):
            if required not in raw_item:
                raise CameraConfigError(f"Camera entry at index {index} is missing required field '{required}'.")
        camera_code = str(raw_item["camera_code"]).strip()
        if not camera_code:
            raise CameraConfigError(f"Camera entry at index {index} has an empty camera_code.")
        if camera_code in seen_codes:
            raise CameraConfigError(f"Duplicate camera_code found: {camera_code}")
        seen_codes.add(camera_code)

        resolved_path = _resolve_source_path(_extract_source_path(raw_item, index=index), config_path=path)
        if validate_paths and not resolved_path.exists():
            raise CameraConfigError(f"Camera '{camera_code}' source_path does not exist: {resolved_path}")
        config = CameraConfig(
            camera_code=camera_code,
            camera_name=_extract_camera_name(raw_item, camera_code=camera_code),
            source_path=resolved_path,
            enabled=bool(raw_item["enabled"]),
            start_time=_parse_start_time(raw_item.get("start_time"), camera_code=camera_code),
        )
        if config.enabled or include_disabled:
            configs.append(config)
    return configs


def apply_file_source_timestamp_policy(
    camera_configs: list[CameraConfig],
    *,
    run_started_at: datetime | None,
) -> list[CameraConfig]:
    """Apply the run/video-source timestamp policy for local-file cameras.

    The multicamera validation pipeline currently reads from file-backed sources.
    When a camera config omits an explicit recording start time, we anchor the
    source to the run-level start timestamp so frame-relative times can be
    translated into stable timestamptz values for persistence.
    """

    if run_started_at is None:
        return list(camera_configs)
    return [
        config if config.start_time is not None else replace(config, start_time=run_started_at)
        for config in camera_configs
    ]
