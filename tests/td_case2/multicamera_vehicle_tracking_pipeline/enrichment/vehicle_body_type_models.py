from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .vehicle_body_type_mapping import SUPPORTED_VEHICLE_BODY_TYPES


VEHICLE_BODY_TYPE_STATUSES = (
    "SUCCESS",
    "LOW_CONFIDENCE",
    "MODEL_ERROR",
    "IMAGE_MISSING",
    "IMAGE_INVALID",
    "PARSE_ERROR",
    "UNKNOWN_RESULT",
    "DISABLED",
)


class VehicleBodyTypeValidationError(ValueError):
    """Raised when a vehicle body-type result is invalid."""


def _normalize_relative_uri(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    candidate = str(value).strip().replace("\\", "/")
    if PureWindowsPath(candidate).drive or candidate.startswith("/") or candidate.startswith("//"):
        raise VehicleBodyTypeValidationError("source_storage_uri must remain relative.")
    if any(part == ".." for part in PurePosixPath(candidate).parts):
        raise VehicleBodyTypeValidationError("source_storage_uri must not contain path traversal.")
    return candidate


@dataclass(frozen=True, slots=True)
class VehicleBodyTypeResult:
    canonical_body_type: str
    raw_output: str
    confidence: float
    status: str
    backend: str = "florence"
    model_name: str | None = None
    adapter_name: str | None = None
    source_storage_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.canonical_body_type not in SUPPORTED_VEHICLE_BODY_TYPES:
            raise VehicleBodyTypeValidationError(f"Unsupported canonical body type: {self.canonical_body_type}")
        confidence = float(self.confidence)
        if confidence != confidence or confidence in (float("inf"), float("-inf")):
            raise VehicleBodyTypeValidationError("confidence must be finite.")
        if not 0.0 <= confidence <= 1.0:
            raise VehicleBodyTypeValidationError("confidence must be between 0 and 1.")
        if self.status not in VEHICLE_BODY_TYPE_STATUSES:
            raise VehicleBodyTypeValidationError(f"Unsupported vehicle body-type status: {self.status}")
        if len(self.raw_output) > 4000:
            raise VehicleBodyTypeValidationError("raw_output must be bounded in length.")
        object.__setattr__(self, "source_storage_uri", _normalize_relative_uri(self.source_storage_uri))
        metadata = deepcopy(self.metadata)
        if not isinstance(metadata, dict):
            raise VehicleBodyTypeValidationError("metadata must be a dictionary.")
        for forbidden_key in ("image_bytes", "image_array", "model_path", "adapter_path", "processor_path", "credentials", "base64"):
            metadata.pop(forbidden_key, None)
        for key, value in list(metadata.items()):
            if isinstance(value, (bytes, bytearray, memoryview)):
                metadata.pop(key, None)
            elif hasattr(value, "shape") and hasattr(value, "dtype"):
                metadata.pop(key, None)
        object.__setattr__(self, "metadata", metadata)

    def to_report_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_body_type": self.canonical_body_type,
            "confidence": self.confidence,
            "backend": self.backend,
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "source_storage_uri": self.source_storage_uri,
            "raw_output": self.raw_output,
            "metadata": deepcopy(self.metadata),
        }
