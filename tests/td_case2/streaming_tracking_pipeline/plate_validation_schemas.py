from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import dataclass_to_dict


FORMAT_STATUSES = {
    "strict_format_match",
    "relaxed_format_match",
    "partial_plate",
    "not_plate_like",
}
FINAL_PLATE_STATUSES = {
    "verified",
    "weak",
    "invalid",
    "no_plate_detected",
    "ocr_empty",
    "ocr_failed",
    "insufficient_evidence",
}


@dataclass(frozen=True)
class PlateValidationConfig:
    minimum_verified_score: float = 0.72
    minimum_weak_score: float = 0.35
    maximum_substitutions_per_candidate: int = 2
    maximum_generated_variants: int = 20
    minimum_agreement_support: int = 2
    allow_single_strong_candidate: bool = True
    strong_detector_confidence: float = 0.60
    strong_format_score: float = 0.90
    format_weight: float = 0.40
    detector_weight: float = 0.20
    ocr_quality_weight: float = 0.15
    crop_quality_weight: float = 0.10
    agreement_weight: float = 0.15


@dataclass(frozen=True)
class PlateTextCandidate:
    source_id: str
    track_id: int
    track_generation: int
    raw_ocr_text: str
    normalized_text: str
    extracted_text: str
    corrected_text: str | None
    substitutions: list[dict[str, Any]]
    format_status: str
    format_score: float
    ocr_confidence: float | None
    plate_detection_confidence: float | None
    crop_role: str | None
    crop_rank: int | None
    frame_index: int | None
    timestamp_sec: float | None
    plate_crop_path: str | None
    source_vehicle_crop_path: str | None
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_for_selection(self) -> str:
        return self.corrected_text or self.extracted_text

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class PlateAgreementResult:
    source_id: str
    track_id: int
    track_generation: int
    candidate_count: int
    unique_candidate_count: int
    exact_groups: dict[str, int]
    best_candidate: str | None
    best_support_count: int
    best_similarity_score: float
    disagreement_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class FinalTrackAnprResult:
    source_id: str
    track_id: int
    track_generation: int
    object_class: str | None
    final_plate_text: str | None
    plate_status: str
    confidence: float
    support_count: int
    selected_candidate: PlateTextCandidate | None
    all_candidates: list[PlateTextCandidate]
    agreement: PlateAgreementResult | None
    normalized_colour: str | None
    raw_colour: str | None
    representative_frame_index: int | None
    representative_timestamp_sec: float | None
    representative_vehicle_crop_path: str | None
    representative_plate_crop_path: str | None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
