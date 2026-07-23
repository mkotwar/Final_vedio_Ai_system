from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .vehicle_colour_mapping import SUPPORTED_VEHICLE_COLOURS


VEHICLE_COLOUR_STATUSES = (
    "SUCCESS",
    "LOW_CONFIDENCE",
    "MODEL_ERROR",
    "IMAGE_MISSING",
    "IMAGE_INVALID",
    "PARSE_ERROR",
    "UNKNOWN_RESULT",
    "DISABLED",
)


class VehicleColourValidationError(ValueError):
    """Raised when a vehicle colour result is invalid."""


def _normalize_relative_uri(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    candidate = str(value).strip().replace("\\", "/")
    if PureWindowsPath(candidate).drive or candidate.startswith("/") or candidate.startswith("//"):
        raise VehicleColourValidationError("source_storage_uri must remain relative.")
    if any(part == ".." for part in PurePosixPath(candidate).parts):
        raise VehicleColourValidationError("source_storage_uri must not contain path traversal.")
    return candidate


@dataclass(frozen=True, slots=True)
class VehicleColourResult:
    canonical_colour: str
    raw_output: str
    confidence: float
    status: str
    secondary_colour: str | None = None
    backend: str = "florence"
    model_name: str | None = None
    adapter_name: str | None = None
    source_media_type: str = "BEST_VEHICLE_CROP"
    source_storage_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.canonical_colour not in SUPPORTED_VEHICLE_COLOURS:
            raise VehicleColourValidationError(f"Unsupported canonical colour: {self.canonical_colour}")
        if self.secondary_colour is not None and self.secondary_colour not in SUPPORTED_VEHICLE_COLOURS:
            raise VehicleColourValidationError(f"Unsupported secondary colour: {self.secondary_colour}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise VehicleColourValidationError("confidence must be between 0 and 1.")
        if self.status not in VEHICLE_COLOUR_STATUSES:
            raise VehicleColourValidationError(f"Unsupported vehicle colour status: {self.status}")
        if len(self.raw_output) > 4000:
            raise VehicleColourValidationError("raw_output must be bounded in length.")
        object.__setattr__(self, "source_storage_uri", _normalize_relative_uri(self.source_storage_uri))
        metadata = deepcopy(self.metadata)
        if not isinstance(metadata, dict):
            raise VehicleColourValidationError("metadata must be a dictionary.")
        for forbidden_key in ("image_bytes", "image_array", "model_path", "adapter_path", "processor_path"):
            metadata.pop(forbidden_key, None)
        object.__setattr__(self, "metadata", metadata)

    def to_report_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_colour": self.canonical_colour,
            "secondary_colour": self.secondary_colour,
            "confidence": self.confidence,
            "backend": self.backend,
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "source_media_type": self.source_media_type,
            "source_storage_uri": self.source_storage_uri,
            "raw_output": self.raw_output,
            "metadata": deepcopy(self.metadata),
        }

