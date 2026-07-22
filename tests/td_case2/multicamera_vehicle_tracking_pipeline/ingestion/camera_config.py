from __future__ import annotations

from dataclasses import dataclass
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
        for required in ("camera_code", "camera_name", "source_path", "enabled"):
            if required not in raw_item:
                raise CameraConfigError(f"Camera entry at index {index} is missing required field '{required}'.")
        camera_code = str(raw_item["camera_code"]).strip()
        if not camera_code:
            raise CameraConfigError(f"Camera entry at index {index} has an empty camera_code.")
        if camera_code in seen_codes:
            raise CameraConfigError(f"Duplicate camera_code found: {camera_code}")
        seen_codes.add(camera_code)

        resolved_path = _resolve_source_path(str(raw_item["source_path"]), config_path=path)
        if validate_paths and not resolved_path.exists():
            raise CameraConfigError(f"Camera '{camera_code}' source_path does not exist: {resolved_path}")
        config = CameraConfig(
            camera_code=camera_code,
            camera_name=str(raw_item["camera_name"]).strip(),
            source_path=resolved_path,
            enabled=bool(raw_item["enabled"]),
            start_time=_parse_start_time(raw_item.get("start_time"), camera_code=camera_code),
        )
        if config.enabled or include_disabled:
            configs.append(config)
    return configs
