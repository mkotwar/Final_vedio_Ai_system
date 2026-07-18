from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .crop_selection import SelectedCropJob
from .serialization import dataclass_to_dict
from .validation import validate_allowed_value, validate_non_empty_string, validate_non_negative_int, validate_probability


ANPR_RESULT_STATUSES = {
    "success",
    "empty_output",
    "model_disabled",
    "input_missing",
    "load_error",
    "inference_error",
    "parse_error",
}
TRACK_PROCESSING_STATUSES = {
    "success",
    "partial",
    "no_jobs",
    "no_plate_candidates",
    "model_disabled",
    "input_missing",
    "load_error",
    "inference_error",
}

COLOUR_VOCABULARY = {
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
    "gold",
    "beige",
    "purple",
    "pink",
    "maroon",
    "unknown",
}
COLOUR_SYNONYMS = {
    "grey": "gray",
    "dark grey": "gray",
    "light grey": "gray",
    "dark gray": "gray",
    "light gray": "gray",
    "cream": "beige",
    "tan": "beige",
    "navy": "blue",
}


def normalize_raw_plate_text(raw_text: str) -> str:
    """Normalize raw OCR text without applying final plate-format validation."""

    normalized = str(raw_text or "").upper()
    normalized = re.sub(r"<\s*OCR\s*>", " ", normalized)
    normalized = re.sub(r"[^A-Z0-9]+", "", normalized)
    return normalized.strip("._-:;,'\" ")


def normalize_vehicle_colour(raw_text: str) -> str:
    raw = str(raw_text or "").strip().lower()
    if not raw:
        return "unknown"
    compact = re.sub(r"[^a-z ]+", " ", raw)
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact in COLOUR_SYNONYMS:
        return COLOUR_SYNONYMS[compact]
    if compact in COLOUR_VOCABULARY:
        return compact
    for phrase, canonical in COLOUR_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", compact):
            return canonical
    for colour in COLOUR_VOCABULARY - {"unknown"}:
        if re.search(rf"\b{re.escape(colour)}\b", compact):
            return colour
    return "unknown"


@dataclass(frozen=True)
class PlateDetectionCandidate:
    source_id: str
    track_id: int
    track_generation: int
    crop_role: str
    crop_rank: int
    frame_index: int
    vehicle_crop_path: str
    plate_rank: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    padded_bbox_xyxy: tuple[int, int, int, int]
    plate_crop_path: str | None
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        validate_non_empty_string(self.crop_role, "crop_role")
        validate_non_negative_int(self.crop_rank, "crop_rank")
        validate_non_negative_int(self.frame_index, "frame_index")
        validate_non_empty_string(self.vehicle_crop_path, "vehicle_crop_path")
        validate_positive = validate_non_negative_int
        validate_positive(self.plate_rank, "plate_rank")
        validate_probability(self.confidence, "confidence")
        validate_allowed_value(self.status, ANPR_RESULT_STATUSES, "status")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class FlorenceOcrResult:
    source_id: str
    track_id: int
    track_generation: int
    crop_role: str
    crop_rank: int
    frame_index: int
    plate_rank: int
    plate_crop_path: str | None
    raw_text: str
    normalized_text: str
    status: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_allowed_value(self.status, ANPR_RESULT_STATUSES, "status")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class FlorenceColourResult:
    source_id: str
    track_id: int
    track_generation: int
    crop_role: str
    crop_rank: int
    frame_index: int
    vehicle_crop_path: str
    raw_text: str
    normalized_colour: str
    status: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_allowed_value(self.status, ANPR_RESULT_STATUSES, "status")
        validate_allowed_value(self.normalized_colour, COLOUR_VOCABULARY, "normalized_colour")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class TrackAnprColourResult:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    object_class: str | None
    lifecycle_completion_reason: str | None
    processing_status: str
    selected_crop_jobs: list[SelectedCropJob] = field(default_factory=list)
    plate_candidates: list[PlateDetectionCandidate] = field(default_factory=list)
    ocr_results: list[FlorenceOcrResult] = field(default_factory=list)
    colour_result: FlorenceColourResult | None = None
    raw_plate_texts: list[str] = field(default_factory=list)
    normalized_plate_texts: list[str] = field(default_factory=list)
    normalized_colour: str = "unknown"
    failure_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_allowed_value(self.processing_status, TRACK_PROCESSING_STATUSES, "processing_status")
        validate_allowed_value(self.normalized_colour, COLOUR_VOCABULARY, "normalized_colour")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
