from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SUPPORTED_ALLOWED_CLASSES = ("3wheeler", "bus", "car", "motorcycle", "truck")
SUPPORTED_DEVICE_PREFIXES = ("auto", "cpu", "cuda", "cuda:0")
SUPPORTED_DOWNLOADABLE_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"}


class DetectionConfigError(ValueError):
    """Raised when detection configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    model_path: str
    fallback_model_path: str | None = None
    allow_fallback: bool = True
    device: str = "auto"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    allowed_classes: tuple[str, ...] = SUPPORTED_ALLOWED_CLASSES

    def __post_init__(self) -> None:
        model_path = str(self.model_path).strip()
        if not model_path:
            raise DetectionConfigError("vehicle_detector.model_path must not be empty.")
        if not _is_supported_model_reference(model_path):
            raise DetectionConfigError(f"Primary model path is invalid or unsupported: {model_path}")
        fallback_model_path = None if self.fallback_model_path in (None, "") else str(self.fallback_model_path).strip()
        if fallback_model_path and not _is_supported_model_reference(fallback_model_path):
            raise DetectionConfigError(f"Fallback model path is invalid or unsupported: {fallback_model_path}")
        if self.device not in SUPPORTED_DEVICE_PREFIXES:
            raise DetectionConfigError(f"Unsupported device value: {self.device}")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise DetectionConfigError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= float(self.iou_threshold) <= 1.0:
            raise DetectionConfigError("iou_threshold must be between 0 and 1.")
        if int(self.image_size) <= 0:
            raise DetectionConfigError("image_size must be positive.")
        normalized_classes = tuple(str(item).strip().lower() for item in self.allowed_classes)
        if not normalized_classes:
            raise DetectionConfigError("allowed_classes must not be empty.")
        for item in normalized_classes:
            if item not in SUPPORTED_ALLOWED_CLASSES:
                raise DetectionConfigError(f"Unsupported allowed class: {item}")
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "fallback_model_path", fallback_model_path)
        object.__setattr__(self, "allowed_classes", normalized_classes)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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
            raise DetectionConfigError("Detection config root must be a mapping.")
        return payload

    payload: dict[str, Any] = {"vehicle_detector": {}}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "vehicle_detector:":
            current_list_key = None
            continue
        if stripped.startswith("- "):
            if current_list_key is None:
                raise DetectionConfigError("Invalid detection.yaml list structure.")
            payload["vehicle_detector"].setdefault(current_list_key, []).append(_parse_scalar(stripped[2:]))
            continue
        key, _, value = stripped.partition(":")
        if not _:
            raise DetectionConfigError("Invalid detection.yaml structure.")
        key = key.strip()
        value = value.strip()
        if value == "":
            current_list_key = key
            payload["vehicle_detector"][key] = []
        else:
            current_list_key = None
            payload["vehicle_detector"][key] = _parse_scalar(value)
    return payload


def _is_supported_model_reference(value: str) -> bool:
    candidate = Path(value)
    if candidate.exists():
        return True
    return candidate.name == value and value in SUPPORTED_DOWNLOADABLE_MODELS


def load_detection_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> DetectionConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise DetectionConfigError(f"Detection config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw_config = payload.get("vehicle_detector")
    if not isinstance(raw_config, dict):
        raise DetectionConfigError("Detection config must contain a 'vehicle_detector' mapping.")
    config = DetectionConfig(
        model_path=str(raw_config.get("model_path", "")),
        fallback_model_path=raw_config.get("fallback_model_path"),
        allow_fallback=bool(raw_config.get("allow_fallback", True)),
        device=str(raw_config.get("device", "auto")),
        confidence_threshold=float(raw_config.get("confidence_threshold", 0.25)),
        iou_threshold=float(raw_config.get("iou_threshold", 0.45)),
        image_size=int(raw_config.get("image_size", 640)),
        allowed_classes=tuple(raw_config.get("allowed_classes", SUPPORTED_ALLOWED_CLASSES)),
    )
    if overrides:
        config = replace(
            config,
            model_path=str(overrides.get("model_path", config.model_path)),
            fallback_model_path=overrides.get("fallback_model_path", config.fallback_model_path),
            allow_fallback=bool(overrides.get("allow_fallback", config.allow_fallback)),
            device=str(overrides.get("device", config.device)),
            confidence_threshold=float(overrides.get("confidence_threshold", config.confidence_threshold)),
            iou_threshold=float(overrides.get("iou_threshold", config.iou_threshold)),
            image_size=int(overrides.get("image_size", config.image_size)),
            allowed_classes=tuple(overrides.get("allowed_classes", config.allowed_classes)),
        )
    return config


def detection_overrides_from_env() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    env_map = {
        "TD_CASE2_MULTICAM_MODEL_PATH": ("model_path", str),
        "TD_CASE2_MULTICAM_FALLBACK_MODEL_PATH": ("fallback_model_path", str),
        "TD_CASE2_MULTICAM_DEVICE": ("device", str),
        "TD_CASE2_MULTICAM_CONFIDENCE": ("confidence_threshold", float),
        "TD_CASE2_MULTICAM_IOU": ("iou_threshold", float),
        "TD_CASE2_MULTICAM_IMAGE_SIZE": ("image_size", int),
    }
    for env_name, (target_name, caster) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[target_name] = caster(raw_value)
    return overrides
