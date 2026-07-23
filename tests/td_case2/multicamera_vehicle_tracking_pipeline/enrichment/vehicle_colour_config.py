from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .vehicle_colour_mapping import SUPPORTED_VEHICLE_COLOURS


class VehicleColourConfigError(ValueError):
    """Raised when vehicle-colour configuration is invalid."""


@dataclass(frozen=True, slots=True)
class VehicleColourConfig:
    enabled: bool = False
    backend: str = "florence"
    source_media_type: str = "BEST_VEHICLE_CROP"
    persist_result: bool = True
    fail_pipeline_on_error: bool = False
    minimum_crop_width: int = 100
    minimum_crop_height: int = 80
    minimum_confidence: float = 0.5
    allowed_colours: tuple[str, ...] = SUPPORTED_VEHICLE_COLOURS
    default_confidence_when_missing: float = 0.5
    max_retries: int = 1
    task_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.backend != "florence":
            raise VehicleColourConfigError("vehicle_colour backend must be 'florence'.")
        if self.source_media_type != "BEST_VEHICLE_CROP":
            raise VehicleColourConfigError("vehicle_colour source_media_type must be BEST_VEHICLE_CROP.")
        if self.minimum_crop_width <= 0 or self.minimum_crop_height <= 0:
            raise VehicleColourConfigError("minimum crop dimensions must be positive.")
        if not 0.0 <= float(self.minimum_confidence) <= 1.0:
            raise VehicleColourConfigError("minimum_confidence must be between 0 and 1.")
        if not 0.0 <= float(self.default_confidence_when_missing) <= 1.0:
            raise VehicleColourConfigError("default_confidence_when_missing must be between 0 and 1.")
        if int(self.max_retries) < 0:
            raise VehicleColourConfigError("max_retries must be non-negative.")
        if int(self.task_timeout_seconds) <= 0:
            raise VehicleColourConfigError("task_timeout_seconds must be positive.")
        if not self.allowed_colours or "UNKNOWN" not in self.allowed_colours:
            raise VehicleColourConfigError("allowed_colours must contain UNKNOWN.")


def load_vehicle_colour_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> VehicleColourConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise VehicleColourConfigError(f"Vehicle colour config file does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        raise VehicleColourConfigError("PyYAML is required to load vehicle colour config.")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise VehicleColourConfigError("Vehicle colour config root must be a mapping.")
    raw = payload.get("vehicle_colour", {})
    if not isinstance(raw, dict):
        raise VehicleColourConfigError("Vehicle colour config must contain a 'vehicle_colour' mapping.")
    config = VehicleColourConfig(
        enabled=bool(raw.get("enabled", False)),
        backend=str(raw.get("backend", "florence")),
        source_media_type=str(raw.get("source_media_type", "BEST_VEHICLE_CROP")),
        persist_result=bool(raw.get("persist_result", True)),
        fail_pipeline_on_error=bool(raw.get("fail_pipeline_on_error", False)),
        minimum_crop_width=int(raw.get("minimum_crop_width", 100)),
        minimum_crop_height=int(raw.get("minimum_crop_height", 80)),
        minimum_confidence=float(raw.get("minimum_confidence", 0.5)),
        allowed_colours=tuple(str(item) for item in raw.get("allowed_colours", SUPPORTED_VEHICLE_COLOURS)),
        default_confidence_when_missing=float(raw.get("default_confidence_when_missing", 0.5)),
        max_retries=int(raw.get("max_retries", 1)),
        task_timeout_seconds=int(raw.get("task_timeout_seconds", 60)),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            persist_result=bool(overrides.get("persist_result", config.persist_result)),
            fail_pipeline_on_error=bool(overrides.get("fail_pipeline_on_error", config.fail_pipeline_on_error)),
            minimum_confidence=float(overrides.get("minimum_confidence", config.minimum_confidence)),
            default_confidence_when_missing=float(overrides.get("default_confidence_when_missing", config.default_confidence_when_missing)),
        )
    return config

