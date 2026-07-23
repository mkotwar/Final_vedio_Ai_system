from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..persistence.track_media_mapping import SUPPORTED_TRACK_MEDIA_ROLES


class AnprConfigError(ValueError):
    """Raised when ANPR configuration is invalid."""


SUPPORTED_COUNTRY_PROFILES = ("INDIA",)
SUPPORTED_OCR_BACKENDS = ("florence",)
SUPPORTED_DETECTOR_BACKENDS = ("yolo",)


@dataclass(frozen=True, slots=True)
class PlateDetectorConfig:
    backend: str = "yolo"
    model_path: str | None = None
    model_path_env: str = "PLATE_DETECTOR_MODEL_PATH"
    device: str = "auto"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    maximum_detections_per_vehicle_crop: int = 3

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_DETECTOR_BACKENDS:
            raise AnprConfigError(f"Unsupported plate detector backend: {self.backend}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise AnprConfigError(f"Unsupported plate detector device: {self.device}")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise AnprConfigError("plate_detector.confidence_threshold must be between 0 and 1.")
        if not 0.0 <= float(self.iou_threshold) <= 1.0:
            raise AnprConfigError("plate_detector.iou_threshold must be between 0 and 1.")
        if int(self.maximum_detections_per_vehicle_crop) <= 0:
            raise AnprConfigError("plate_detector.maximum_detections_per_vehicle_crop must be positive.")


@dataclass(frozen=True, slots=True)
class PlateSelectionConfig:
    minimum_width: int = 30
    minimum_height: int = 10
    minimum_area: int = 300
    minimum_aspect_ratio: float = 1.5
    maximum_aspect_ratio: float = 8.0
    confidence_weight: float = 0.35
    sharpness_weight: float = 0.30
    size_weight: float = 0.20
    aspect_ratio_weight: float = 0.10
    edge_penalty_weight: float = 0.05

    def __post_init__(self) -> None:
        for field_name in ("minimum_width", "minimum_height", "minimum_area"):
            if int(getattr(self, field_name)) <= 0:
                raise AnprConfigError(f"plate_selection.{field_name} must be positive.")
        if float(self.minimum_aspect_ratio) <= 0 or float(self.maximum_aspect_ratio) <= 0:
            raise AnprConfigError("plate_selection aspect ratios must be positive.")
        if float(self.minimum_aspect_ratio) > float(self.maximum_aspect_ratio):
            raise AnprConfigError("plate_selection.minimum_aspect_ratio must not exceed maximum_aspect_ratio.")
        for field_name in (
            "confidence_weight",
            "sharpness_weight",
            "size_weight",
            "aspect_ratio_weight",
            "edge_penalty_weight",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise AnprConfigError(f"plate_selection.{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class AnprOcrConfig:
    backend: str = "florence"
    minimum_confidence: float = 0.5
    maximum_retries: int = 1
    retry_with_preprocessing: bool = True
    fallback_to_other_plate_candidates: bool = True
    maximum_plate_candidates_for_ocr: int = 3
    task_prompt: str = "<OCR>"

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_OCR_BACKENDS:
            raise AnprConfigError(f"Unsupported OCR backend: {self.backend}")
        if not 0.0 <= float(self.minimum_confidence) <= 1.0:
            raise AnprConfigError("ocr.minimum_confidence must be between 0 and 1.")
        if int(self.maximum_retries) < 0:
            raise AnprConfigError("ocr.maximum_retries must be non-negative.")
        if int(self.maximum_plate_candidates_for_ocr) <= 0:
            raise AnprConfigError("ocr.maximum_plate_candidates_for_ocr must be positive.")
        if not str(self.task_prompt).strip():
            raise AnprConfigError("ocr.task_prompt must not be empty.")


@dataclass(frozen=True, slots=True)
class AnprValidationConfig:
    country_profile: str = "INDIA"
    preserve_raw_text: bool = True
    minimum_normalized_length: int = 6
    maximum_normalized_length: int = 12
    allow_unverified_result: bool = True
    persist_only_verified_as_primary: bool = True

    def __post_init__(self) -> None:
        if self.country_profile not in SUPPORTED_COUNTRY_PROFILES:
            raise AnprConfigError(f"Unsupported country profile: {self.country_profile}")
        if int(self.minimum_normalized_length) <= 0 or int(self.maximum_normalized_length) <= 0:
            raise AnprConfigError("validation normalized lengths must be positive.")
        if int(self.minimum_normalized_length) > int(self.maximum_normalized_length):
            raise AnprConfigError("validation.minimum_normalized_length must not exceed maximum_normalized_length.")


@dataclass(frozen=True, slots=True)
class AnprMediaConfig:
    save_plate_crops: bool = True
    save_preprocessed_plate_crops: bool = False
    artifact_subdirectory: str = "plate_evidence"

    def __post_init__(self) -> None:
        if not str(self.artifact_subdirectory).strip():
            raise AnprConfigError("media.artifact_subdirectory must not be empty.")


@dataclass(frozen=True, slots=True)
class AnprConfig:
    enabled: bool = False
    plate_detector: PlateDetectorConfig = field(default_factory=PlateDetectorConfig)
    vehicle_evidence_roles: tuple[str, ...] = (
        "HIGHEST_CONFIDENCE",
        "SHARPEST",
        "BEST_OVERALL",
        "LARGEST",
        "MIDDLE",
        "FIRST",
        "LAST",
    )
    maximum_vehicle_crops_per_track: int = 5
    plate_selection: PlateSelectionConfig = field(default_factory=PlateSelectionConfig)
    ocr: AnprOcrConfig = field(default_factory=AnprOcrConfig)
    validation: AnprValidationConfig = field(default_factory=AnprValidationConfig)
    media: AnprMediaConfig = field(default_factory=AnprMediaConfig)
    persist_result: bool = True
    fail_pipeline_on_error: bool = False

    def __post_init__(self) -> None:
        if int(self.maximum_vehicle_crops_per_track) <= 0:
            raise AnprConfigError("maximum_vehicle_crops_per_track must be positive.")
        if not self.vehicle_evidence_roles:
            raise AnprConfigError("vehicle_evidence_roles must not be empty.")
        normalized_roles = tuple(str(role).strip().upper() for role in self.vehicle_evidence_roles)
        unsupported = [role for role in normalized_roles if role not in SUPPORTED_TRACK_MEDIA_ROLES]
        if unsupported:
            raise AnprConfigError(f"Unsupported vehicle_evidence_roles: {', '.join(unsupported)}")
        object.__setattr__(self, "vehicle_evidence_roles", normalized_roles)


def load_anpr_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> AnprConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise AnprConfigError(f"ANPR config file does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is None:
        raise AnprConfigError("PyYAML is required to load ANPR config.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise AnprConfigError("ANPR config root must be a mapping.")
    raw = payload.get("anpr", {})
    if not isinstance(raw, dict):
        raise AnprConfigError("ANPR config must contain an 'anpr' mapping.")
    detector_raw = _mapping(raw, "plate_detector")
    selection_raw = _mapping(raw, "plate_selection")
    ocr_raw = _mapping(raw, "ocr")
    validation_raw = _mapping(raw, "validation")
    media_raw = _mapping(raw, "media")
    config = AnprConfig(
        enabled=bool(raw.get("enabled", False)),
        plate_detector=PlateDetectorConfig(
            backend=str(detector_raw.get("backend", "yolo")),
            model_path=_optional_str(detector_raw.get("model_path")),
            model_path_env=str(detector_raw.get("model_path_env", "PLATE_DETECTOR_MODEL_PATH")),
            device=str(detector_raw.get("device", "auto")),
            confidence_threshold=float(detector_raw.get("confidence_threshold", 0.25)),
            iou_threshold=float(detector_raw.get("iou_threshold", 0.45)),
            maximum_detections_per_vehicle_crop=int(detector_raw.get("maximum_detections_per_vehicle_crop", 3)),
        ),
        vehicle_evidence_roles=tuple(str(item) for item in raw.get("vehicle_evidence_roles", (
            "HIGHEST_CONFIDENCE",
            "SHARPEST",
            "BEST_OVERALL",
            "LARGEST",
            "MIDDLE",
            "FIRST",
            "LAST",
        ))),
        maximum_vehicle_crops_per_track=int(raw.get("maximum_vehicle_crops_per_track", 5)),
        plate_selection=PlateSelectionConfig(
            minimum_width=int(selection_raw.get("minimum_width", 30)),
            minimum_height=int(selection_raw.get("minimum_height", 10)),
            minimum_area=int(selection_raw.get("minimum_area", 300)),
            minimum_aspect_ratio=float(selection_raw.get("minimum_aspect_ratio", 1.5)),
            maximum_aspect_ratio=float(selection_raw.get("maximum_aspect_ratio", 8.0)),
            confidence_weight=float(selection_raw.get("confidence_weight", 0.35)),
            sharpness_weight=float(selection_raw.get("sharpness_weight", 0.30)),
            size_weight=float(selection_raw.get("size_weight", 0.20)),
            aspect_ratio_weight=float(selection_raw.get("aspect_ratio_weight", 0.10)),
            edge_penalty_weight=float(selection_raw.get("edge_penalty_weight", 0.05)),
        ),
        ocr=AnprOcrConfig(
            backend=str(ocr_raw.get("backend", "florence")),
            minimum_confidence=float(ocr_raw.get("minimum_confidence", 0.5)),
            maximum_retries=int(ocr_raw.get("maximum_retries", 1)),
            retry_with_preprocessing=bool(ocr_raw.get("retry_with_preprocessing", True)),
            fallback_to_other_plate_candidates=bool(ocr_raw.get("fallback_to_other_plate_candidates", True)),
            maximum_plate_candidates_for_ocr=int(ocr_raw.get("maximum_plate_candidates_for_ocr", 3)),
            task_prompt=str(ocr_raw.get("task_prompt", "<OCR>")),
        ),
        validation=AnprValidationConfig(
            country_profile=str(validation_raw.get("country_profile", "INDIA")),
            preserve_raw_text=bool(validation_raw.get("preserve_raw_text", True)),
            minimum_normalized_length=int(validation_raw.get("minimum_normalized_length", 6)),
            maximum_normalized_length=int(validation_raw.get("maximum_normalized_length", 12)),
            allow_unverified_result=bool(validation_raw.get("allow_unverified_result", True)),
            persist_only_verified_as_primary=bool(validation_raw.get("persist_only_verified_as_primary", True)),
        ),
        media=AnprMediaConfig(
            save_plate_crops=bool(media_raw.get("save_plate_crops", True)),
            save_preprocessed_plate_crops=bool(media_raw.get("save_preprocessed_plate_crops", False)),
            artifact_subdirectory=str(media_raw.get("artifact_subdirectory", "plate_evidence")),
        ),
        persist_result=bool(raw.get("persist_result", True)),
        fail_pipeline_on_error=bool(raw.get("fail_pipeline_on_error", False)),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            persist_result=bool(overrides.get("persist_result", config.persist_result)),
            fail_pipeline_on_error=bool(overrides.get("fail_pipeline_on_error", config.fail_pipeline_on_error)),
        )
    return config


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise AnprConfigError(f"ANPR config '{key}' must be a mapping.")
    return value


def _optional_str(value: object) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)
