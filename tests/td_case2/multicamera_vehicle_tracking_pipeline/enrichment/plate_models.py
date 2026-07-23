from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class PlateModelValidationError(ValueError):
    """Raised when a plate-related model is invalid."""


def _ensure_relative_uri(value: str, *, field_name: str) -> str:
    candidate = str(value).strip()
    if not candidate:
        raise PlateModelValidationError(f"{field_name} must not be empty.")
    windows_path = PureWindowsPath(candidate)
    if windows_path.drive:
        raise PlateModelValidationError(f"{field_name} must be relative, not a Windows drive path.")
    if candidate.startswith("\\\\"):
        raise PlateModelValidationError(f"{field_name} must be relative, not a UNC path.")
    normalized = candidate.replace("\\", "/")
    if normalized.startswith("/"):
        raise PlateModelValidationError(f"{field_name} must be relative, not absolute.")
    if any(part == ".." for part in PurePosixPath(normalized).parts):
        raise PlateModelValidationError(f"{field_name} must not contain path traversal.")
    return normalized


def _ensure_confidence(value: float | None, *, field_name: str) -> None:
    if value is not None and not 0.0 <= float(value) <= 1.0:
        raise PlateModelValidationError(f"{field_name} must be between 0 and 1.")


def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload = {} if metadata is None else dict(metadata)
    for key, value in payload.items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise PlateModelValidationError(f"metadata field '{key}' must not contain binary data.")
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            raise PlateModelValidationError(f"metadata field '{key}' must not contain image arrays.")
    return payload


@dataclass(frozen=True, slots=True)
class PlateDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None

    def __post_init__(self) -> None:
        _ensure_confidence(self.confidence, field_name="confidence")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise PlateModelValidationError("PlateDetection bbox must satisfy x2 > x1 and y2 > y1.")


@dataclass(frozen=True, slots=True)
class VehicleEvidenceInput:
    track_uuid: str
    camera_code: str
    source_vehicle_role: str
    source_vehicle_storage_uri: str
    local_file_path: Path
    frame_number: int
    video_time_seconds: float
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int
    sharpness_score: float
    edge_penalty: float
    overall_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_vehicle_storage_uri", _ensure_relative_uri(self.source_vehicle_storage_uri, field_name="source_vehicle_storage_uri"))
        if not self.local_file_path.exists():
            raise PlateModelValidationError(f"Vehicle evidence file does not exist: {self.local_file_path}")
        _ensure_confidence(self.confidence, field_name="confidence")
        if self.crop_width <= 0 or self.crop_height <= 0:
            raise PlateModelValidationError("VehicleEvidenceInput crop dimensions must be positive.")


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    track_uuid: str
    camera_code: str
    source_vehicle_role: str
    source_vehicle_storage_uri: str
    plate_bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    crop_width: int
    crop_height: int
    area: int
    aspect_ratio: float
    sharpness_score: float
    edge_penalty: float
    overall_score: float
    local_file_path: Path
    relative_storage_uri: str
    frame_number: int
    video_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_confidence(self.detector_confidence, field_name="detector_confidence")
        x1, y1, x2, y2 = self.plate_bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise PlateModelValidationError("PlateCandidate plate_bbox_xyxy must satisfy x2 > x1 and y2 > y1.")
        if self.crop_width <= 0 or self.crop_height <= 0 or self.area <= 0:
            raise PlateModelValidationError("PlateCandidate crop dimensions and area must be positive.")
        if float(self.aspect_ratio) <= 0:
            raise PlateModelValidationError("PlateCandidate aspect_ratio must be positive.")
        object.__setattr__(self, "source_vehicle_storage_uri", _ensure_relative_uri(self.source_vehicle_storage_uri, field_name="source_vehicle_storage_uri"))
        object.__setattr__(self, "relative_storage_uri", _ensure_relative_uri(self.relative_storage_uri, field_name="relative_storage_uri"))
        object.__setattr__(self, "metadata", _sanitize_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class NormalizedRegistrationText:
    raw_text: str
    cleaned_text: str
    candidate_values: tuple[str, ...]
    transformations: tuple[str, ...] = ()
    ambiguity_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlateValidationResult:
    normalized_text: str | None
    is_verified: bool
    status: str
    confidence_adjustment: float
    matched_pattern: str | None
    reasons: tuple[str, ...]
    candidate_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlateOcrResult:
    raw_text: str
    normalized_text: str | None
    confidence: float
    status: str
    verification_status: str
    country_profile: str
    backend: str
    model_name: str | None
    adapter_name: str | None
    source_vehicle_track_id: str
    source_plate_storage_uri: str
    source_vehicle_storage_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_confidence(self.confidence, field_name="confidence")
        object.__setattr__(self, "source_plate_storage_uri", _ensure_relative_uri(self.source_plate_storage_uri, field_name="source_plate_storage_uri"))
        object.__setattr__(self, "source_vehicle_storage_uri", _ensure_relative_uri(self.source_vehicle_storage_uri, field_name="source_vehicle_storage_uri"))
        if len(self.raw_text) > 512:
            raise PlateModelValidationError("PlateOcrResult raw_text must be bounded.")
        object.__setattr__(self, "metadata", _sanitize_metadata(self.metadata))
