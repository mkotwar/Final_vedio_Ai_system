from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .vehicle_body_type_mapping import SUPPORTED_VEHICLE_BODY_TYPES, normalize_vehicle_body_type


class VehicleBodyTypeConfigError(ValueError):
    """Raised when vehicle body-type configuration is invalid."""


@dataclass(frozen=True, slots=True)
class VehicleBodyTypeConfig:
    enabled: bool = False
    backend: str = "florence"
    minimum_confidence: float = 0.5
    default_confidence_when_missing: float = 0.5
    allowed_body_types: tuple[str, ...] = SUPPORTED_VEHICLE_BODY_TYPES
    prompt: str = (
        "Classify the body type of the vehicle in this image. "
        "Return exactly one configured body-type label. Return only the label."
    )
    fail_pipeline_on_error: bool = False

    def __post_init__(self) -> None:
        if self.backend != "florence":
            raise VehicleBodyTypeConfigError("vehicle_body_type backend must be 'florence'.")
        if not 0.0 <= float(self.minimum_confidence) <= 1.0:
            raise VehicleBodyTypeConfigError("minimum_confidence must be between 0 and 1.")
        if not 0.0 <= float(self.default_confidence_when_missing) <= 1.0:
            raise VehicleBodyTypeConfigError("default_confidence_when_missing must be between 0 and 1.")
        if not self.allowed_body_types:
            raise VehicleBodyTypeConfigError("allowed_body_types must not be empty.")
        normalized = tuple(normalize_vehicle_body_type(value) for value in self.allowed_body_types)
        if "UNKNOWN" not in normalized:
            raise VehicleBodyTypeConfigError("allowed_body_types must contain UNKNOWN.")
        if len(set(normalized)) != len(normalized):
            raise VehicleBodyTypeConfigError("allowed_body_types must not contain duplicates after normalization.")
        object.__setattr__(self, "allowed_body_types", normalized)
        if not str(self.prompt).strip():
            raise VehicleBodyTypeConfigError("prompt must not be empty.")


def load_vehicle_body_type_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> VehicleBodyTypeConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise VehicleBodyTypeConfigError(f"Vehicle body type config file does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        raise VehicleBodyTypeConfigError("PyYAML is required to load vehicle body type config.")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise VehicleBodyTypeConfigError("Vehicle body type config root must be a mapping.")
    raw = payload.get("vehicle_body_type", {})
    if not isinstance(raw, dict):
        raise VehicleBodyTypeConfigError("Vehicle body type config must contain a 'vehicle_body_type' mapping.")
    config = VehicleBodyTypeConfig(
        enabled=bool(raw.get("enabled", False)),
        backend=str(raw.get("backend", "florence")),
        minimum_confidence=float(raw.get("minimum_confidence", 0.5)),
        default_confidence_when_missing=float(raw.get("default_confidence_when_missing", 0.5)),
        allowed_body_types=tuple(str(item) for item in raw.get("allowed_body_types", SUPPORTED_VEHICLE_BODY_TYPES)),
        prompt=str(raw.get("prompt", "Classify the body type of the vehicle in this image. Return exactly one configured body-type label. Return only the label.")),
        fail_pipeline_on_error=bool(raw.get("fail_pipeline_on_error", False)),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            minimum_confidence=float(overrides.get("minimum_confidence", config.minimum_confidence)),
            default_confidence_when_missing=float(overrides.get("default_confidence_when_missing", config.default_confidence_when_missing)),
            fail_pipeline_on_error=bool(overrides.get("fail_pipeline_on_error", config.fail_pipeline_on_error)),
        )
    return config
