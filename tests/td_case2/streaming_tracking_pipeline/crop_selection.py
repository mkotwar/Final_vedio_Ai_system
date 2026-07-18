from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import BestCropScoreConfig, BestCropSelectionConfig
from .crop_artifacts import CompletedTrackCropBundle
from .schemas import BoundingBox, CropCandidate, TrackCompletionReason, TrackRecord, TrackStatus
from .serialization import dataclass_to_dict
from .validation import (
    validate_allowed_value,
    validate_finite_float,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_probability,
)


SELECTED_CROP_ROLES = ("primary", "fallback")
SELECTION_STATUSES = ("primary_selected", "fallback_only", "no_valid_crop", "disabled", "error")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quality_value(candidate: CropCandidate, name: str) -> Any:
    return getattr(candidate.quality, name, None)


def _normalize_optional_probability(value: float | None, name: str, missing: list[str]) -> float:
    if value is None:
        missing.append(name)
        return 0.0
    return _clip01(float(value))


def _normalize_optional_capped(value: float | None, name: str, cap: float, missing: list[str]) -> float:
    if value is None:
        missing.append(name)
        return 0.0
    return _clip01(float(value) / max(float(cap), 1e-9))


def _brightness_component(value: float | None, target: float, missing: list[str]) -> float:
    if value is None:
        missing.append("brightness")
        return 0.0
    denominator = max(target, 1.0 - target, 1e-9)
    return _clip01(1.0 - (abs(float(value) - target) / denominator))


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    union = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return _clip01(intersection / union)


@dataclass(frozen=True)
class CropSelectionScore:
    """Bounded final score with normalized components.

    Normalization:
    - probabilities stay in 0..1;
    - bbox area is divided by `bbox_area_normalization_cap`;
    - sharpness and contrast are divided by configured caps;
    - brightness is best at `target_brightness` and falls toward 0 at extremes;
    - temporal position prefers the middle of a completed bundle.
    """

    confidence_component: float
    area_component: float
    sharpness_component: float
    brightness_component: float
    contrast_component: float
    completeness_component: float
    plate_visibility_component: float
    temporal_component: float
    edge_penalty: float
    clipping_penalty: float
    low_resolution_penalty: float
    final_score: float
    missing_metric_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for field_name in (
            "confidence_component",
            "area_component",
            "sharpness_component",
            "brightness_component",
            "contrast_component",
            "completeness_component",
            "plate_visibility_component",
            "temporal_component",
            "edge_penalty",
            "clipping_penalty",
            "low_resolution_penalty",
            "final_score",
        ):
            validate_probability(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class ScoredCropCandidate:
    candidate: CropCandidate
    score: CropSelectionScore
    original_index: int


@dataclass(frozen=True)
class SelectedCrop:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    frame_index: int
    timestamp_sec: float
    vehicle_crop_path: str | None
    full_frame_path: str | None
    crop_bbox: BoundingBox
    class_name: str | None
    role: str
    rank: int
    final_score: float
    score_breakdown: CropSelectionScore
    selection_reasons: list[str] = field(default_factory=list)
    rejection_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if self.vehicle_crop_path is not None:
            validate_non_empty_string(self.vehicle_crop_path, "vehicle_crop_path")
        if self.full_frame_path is not None:
            validate_non_empty_string(self.full_frame_path, "full_frame_path")
        if self.class_name is not None:
            validate_non_empty_string(self.class_name, "class_name")
        object.__setattr__(self, "role", validate_allowed_value(self.role, SELECTED_CROP_ROLES, "role"))
        validate_non_negative_int(self.rank, "rank")
        validate_probability(self.final_score, "final_score")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class SelectedCropJob:
    """OCR-ready selected-crop job description. This schema does not execute OCR."""

    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    object_class: str | None
    lifecycle_completion_reason: str | None
    crop_role: str
    crop_rank: int
    frame_index: int
    timestamp_sec: float
    vehicle_crop_path: str
    full_frame_path: str | None
    selection_score: float
    quality_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class SelectedTrackCropSet:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    lifecycle_record: TrackRecord
    candidate_count: int
    eligible_primary_count: int
    eligible_fallback_count: int
    primary_crops: list[SelectedCrop] = field(default_factory=list)
    fallback_crop: SelectedCrop | None = None
    selection_status: str = "no_valid_crop"
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        validate_non_negative_int(self.candidate_count, "candidate_count")
        validate_non_negative_int(self.eligible_primary_count, "eligible_primary_count")
        validate_non_negative_int(self.eligible_fallback_count, "eligible_fallback_count")
        object.__setattr__(
            self,
            "selection_status",
            validate_allowed_value(self.selection_status, SELECTION_STATUSES, "selection_status"),
        )
        for crop in self.primary_crops:
            self._validate_crop_identity(crop)
            if crop.role != "primary":
                raise ValueError("primary_crops may only contain role=primary.")
        if self.fallback_crop is not None:
            self._validate_crop_identity(self.fallback_crop)
            if self.fallback_crop.role != "fallback":
                raise ValueError("fallback_crop must have role=fallback.")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    def to_crop_jobs(self) -> list[SelectedCropJob]:
        selected = list(self.primary_crops)
        if self.fallback_crop is not None:
            selected.append(self.fallback_crop)
        jobs: list[SelectedCropJob] = []
        for crop in selected:
            if not crop.vehicle_crop_path:
                continue
            jobs.append(
                SelectedCropJob(
                    source_id=crop.source_id,
                    track_id=crop.track_id,
                    track_generation=crop.track_generation,
                    source_track_id=crop.source_track_id,
                    object_class=crop.class_name or self.lifecycle_record.dominant_class,
                    lifecycle_completion_reason=self.lifecycle_record.completion_reason.value
                    if self.lifecycle_record.completion_reason is not None
                    else None,
                    crop_role=crop.role,
                    crop_rank=crop.rank,
                    frame_index=crop.frame_index,
                    timestamp_sec=crop.timestamp_sec,
                    vehicle_crop_path=crop.vehicle_crop_path,
                    full_frame_path=crop.full_frame_path,
                    selection_score=crop.final_score,
                    quality_warnings=list(crop.rejection_warnings),
                    metadata=dict(crop.metadata),
                )
            )
        return jobs

    def _validate_crop_identity(self, crop: SelectedCrop) -> None:
        if (crop.source_id, crop.track_id, crop.track_generation) != (self.source_id, self.track_id, self.track_generation):
            raise ValueError("Selected crop identity does not match its selected track crop set.")


class FinalBestCropSelector:
    """Deterministic final best-crop selector for completed crop bundles."""

    def __init__(self, config: BestCropSelectionConfig | None = None, score_config: BestCropScoreConfig | None = None) -> None:
        self.config = config or BestCropSelectionConfig()
        self.score_config = score_config or BestCropScoreConfig()

    def select(self, bundle: CompletedTrackCropBundle) -> SelectedTrackCropSet:
        lifecycle_record = _bundle_to_lifecycle_record(bundle)
        if not self.config.enabled:
            return SelectedTrackCropSet(
                source_id=bundle.source_id,
                track_id=bundle.track_id,
                track_generation=bundle.track_generation,
                source_track_id=bundle.source_track_id,
                lifecycle_record=lifecycle_record,
                candidate_count=len(bundle.candidates),
                eligible_primary_count=0,
                eligible_fallback_count=0,
                selection_status="disabled",
                rejection_reason_counts={"selector_disabled": len(bundle.candidates)},
                metadata={"ocr_ready_jobs": []},
            )

        candidates, duplicate_count = self._deduplicate_candidates(bundle)
        scored = self._score_candidates(candidates)
        rejections: Counter[str] = Counter()
        if duplicate_count:
            rejections["duplicate_candidate_removed"] = duplicate_count

        primary_gate_reasons = self._track_primary_gate_reasons(bundle, candidates)
        primary_eligible: list[ScoredCropCandidate] = []
        fallback_eligible: list[ScoredCropCandidate] = []
        candidate_warnings: dict[tuple[int, str | None], list[str]] = {}

        for scored_candidate in scored:
            primary_reasons = list(primary_gate_reasons)
            primary_reasons.extend(self._primary_rejection_reasons(scored_candidate))
            if primary_reasons:
                for reason in primary_reasons:
                    rejections[reason] += 1
            else:
                primary_eligible.append(scored_candidate)

            fallback_reasons = self._fallback_rejection_reasons(scored_candidate)
            if fallback_reasons:
                for reason in fallback_reasons:
                    rejections[f"fallback_{reason}"] += 1
            else:
                fallback_eligible.append(scored_candidate)
            candidate_warnings[self._candidate_key(scored_candidate.candidate)] = primary_reasons

        selected_primary, diversity_counts, diversity_rejections = self._select_primary(primary_eligible)
        rejections.update(diversity_rejections)
        selected_primary_keys = {self._candidate_key(item.candidate) for item in selected_primary}
        selected_primary_crops = [
            self._to_selected_crop(
                item,
                role="primary",
                rank=index + 1,
                reasons=["passed_primary_eligibility", self.config.primary_selection_policy],
                warnings=candidate_warnings.get(self._candidate_key(item.candidate), []),
                diversity_mode="strict",
            )
            for index, item in enumerate(selected_primary)
        ]

        selected_fallback: SelectedCrop | None = None
        if self.config.keep_fallback_crop:
            fallback_pool = list(fallback_eligible)
            if selected_primary_crops:
                if self.config.keep_distinct_fallback_when_primary_short and len(selected_primary_crops) < self.config.primary_crop_count:
                    fallback_pool = [item for item in fallback_pool if self._candidate_key(item.candidate) not in selected_primary_keys]
                else:
                    fallback_pool = []
            fallback_candidate = self._select_fallback_candidate(fallback_pool)
            if fallback_candidate is not None:
                selected_fallback = self._to_selected_crop(
                    fallback_candidate,
                    role="fallback",
                    rank=1,
                    reasons=["passed_fallback_eligibility", self.config.fallback_selection_policy],
                    warnings=candidate_warnings.get(self._candidate_key(fallback_candidate.candidate), []),
                    diversity_mode="not_required",
                )
        elif not selected_primary_crops:
            rejections["fallback_disabled"] += 1

        if selected_primary_crops:
            status = "primary_selected"
        elif selected_fallback is not None:
            status = "fallback_only"
        else:
            status = "no_valid_crop"
            if not candidates:
                rejections["no_candidates"] += 1

        result = SelectedTrackCropSet(
            source_id=bundle.source_id,
            track_id=bundle.track_id,
            track_generation=bundle.track_generation,
            source_track_id=bundle.source_track_id,
            lifecycle_record=lifecycle_record,
            candidate_count=len(candidates),
            eligible_primary_count=len(primary_eligible),
            eligible_fallback_count=len(fallback_eligible),
            primary_crops=selected_primary_crops,
            fallback_crop=selected_fallback,
            selection_status=status,
            rejection_reason_counts=dict(sorted(rejections.items())),
            metadata={
                "completion_reason": bundle.completion_reason,
                "dominant_class": bundle.metadata.get("dominant_class"),
                "strict_diversity_selection_count": diversity_counts["strict"],
                "relaxed_diversity_selection_count": diversity_counts["relaxed"],
                "duplicate_candidates_removed": duplicate_count,
                "ocr_ready_jobs": [job.to_dict() for job in SelectedTrackCropSet(
                    source_id=bundle.source_id,
                    track_id=bundle.track_id,
                    track_generation=bundle.track_generation,
                    source_track_id=bundle.source_track_id,
                    lifecycle_record=lifecycle_record,
                    candidate_count=len(candidates),
                    eligible_primary_count=len(primary_eligible),
                    eligible_fallback_count=len(fallback_eligible),
                    primary_crops=selected_primary_crops,
                    fallback_crop=selected_fallback,
                    selection_status=status,
                    rejection_reason_counts=dict(sorted(rejections.items())),
                ).to_crop_jobs()],
            },
        )
        return result

    def select_many(self, bundles: Sequence[CompletedTrackCropBundle]) -> list[SelectedTrackCropSet]:
        return [self.select(bundle) for bundle in sorted(bundles, key=lambda item: (item.source_id, item.track_id, item.track_generation))]

    def score_candidate(self, candidate: CropCandidate, *, temporal_index: int = 0, temporal_total: int = 1) -> CropSelectionScore:
        missing: list[str] = []
        q = candidate.quality
        confidence = _normalize_optional_probability(candidate.detection_confidence if candidate.detection_confidence is not None else q.detection_confidence, "confidence", missing)
        area = _normalize_optional_capped(q.bbox_area_ratio, "bbox_area_ratio", self.score_config.bbox_area_normalization_cap, missing)
        sharpness = _normalize_optional_capped(q.sharpness, "sharpness", self.score_config.sharpness_normalization_cap, missing)
        brightness = _brightness_component(q.brightness, self.score_config.target_brightness, missing)
        contrast = _normalize_optional_capped(q.contrast, "contrast", self.score_config.contrast_normalization_cap, missing)
        completeness = _normalize_optional_probability(q.crop_completeness, "crop_completeness", missing)
        plate_visibility = _normalize_optional_probability(q.plate_visibility_score, "plate_visibility_score", missing)
        temporal = _temporal_position_score(temporal_index, temporal_total)
        edge_penalty = 1.0 if bool(q.edge_touching) else 0.0
        clipping_penalty = 1.0 if bool(q.padding_clipped) or (q.crop_completeness is not None and q.crop_completeness < 1.0) else 0.0
        width = q.crop_width or 0
        height = q.crop_height or 0
        low_resolution_penalty = 1.0 if width < self.score_config.low_resolution_width or height < self.score_config.low_resolution_height else 0.0

        positive_weight = (
            self.score_config.confidence_weight
            + self.score_config.bbox_area_weight
            + self.score_config.sharpness_weight
            + self.score_config.brightness_weight
            + self.score_config.contrast_weight
            + self.score_config.completeness_weight
            + self.score_config.plate_visibility_weight
            + self.score_config.temporal_position_weight
        )
        positive = (
            self.score_config.confidence_weight * confidence
            + self.score_config.bbox_area_weight * area
            + self.score_config.sharpness_weight * sharpness
            + self.score_config.brightness_weight * brightness
            + self.score_config.contrast_weight * contrast
            + self.score_config.completeness_weight * completeness
            + self.score_config.plate_visibility_weight * plate_visibility
            + self.score_config.temporal_position_weight * temporal
        ) / max(positive_weight, 1e-9)
        penalty_weight = self.score_config.edge_penalty_weight + self.score_config.clipping_penalty_weight + self.score_config.low_resolution_penalty_weight
        penalty = (
            self.score_config.edge_penalty_weight * edge_penalty
            + self.score_config.clipping_penalty_weight * clipping_penalty
            + self.score_config.low_resolution_penalty_weight * low_resolution_penalty
        ) / max(penalty_weight, 1e-9) if penalty_weight > 0.0 else 0.0
        final = _clip01(positive - (penalty * min(0.5, penalty_weight / max(positive_weight + penalty_weight, 1e-9))))
        return CropSelectionScore(
            confidence_component=round(confidence, 6),
            area_component=round(area, 6),
            sharpness_component=round(sharpness, 6),
            brightness_component=round(brightness, 6),
            contrast_component=round(contrast, 6),
            completeness_component=round(completeness, 6),
            plate_visibility_component=round(plate_visibility, 6),
            temporal_component=round(temporal, 6),
            edge_penalty=round(edge_penalty, 6),
            clipping_penalty=round(clipping_penalty, 6),
            low_resolution_penalty=round(low_resolution_penalty, 6),
            final_score=round(final, 6),
            missing_metric_names=sorted(set(missing)),
        )

    def _score_candidates(self, candidates: list[CropCandidate]) -> list[ScoredCropCandidate]:
        return sorted(
            [
                ScoredCropCandidate(
                    candidate=candidate,
                    score=self.score_candidate(candidate, temporal_index=index, temporal_total=len(candidates)),
                    original_index=index,
                )
                for index, candidate in enumerate(sorted(candidates, key=self._stable_candidate_order))
            ],
            key=self._tie_break_key,
        )

    def _deduplicate_candidates(self, bundle: CompletedTrackCropBundle) -> tuple[list[CropCandidate], int]:
        seen: set[tuple[int, float, str | None]] = set()
        retained: list[CropCandidate] = []
        duplicate_count = 0
        for candidate in sorted(bundle.candidates, key=self._stable_candidate_order):
            self._validate_candidate_identity(bundle, candidate)
            key = self._candidate_key(candidate)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            retained.append(candidate)
        return retained, duplicate_count

    def _validate_candidate_identity(self, bundle: CompletedTrackCropBundle, candidate: CropCandidate) -> None:
        if candidate.source_id is not None and candidate.source_id != bundle.source_id:
            raise ValueError("Candidate source_id does not match completed bundle.")
        if candidate.track_id != bundle.track_id or candidate.track_generation != bundle.track_generation:
            raise ValueError("Candidate track_id/track_generation does not match completed bundle.")

    def _track_primary_gate_reasons(self, bundle: CompletedTrackCropBundle, candidates: list[CropCandidate]) -> list[str]:
        reasons: list[str] = []
        if bundle.observation_count < self.config.minimum_track_observations_for_primary:
            reasons.append("track_observation_count_below_primary_min")
        if len(candidates) < self.config.minimum_candidates_for_primary:
            reasons.append("candidate_count_below_primary_min")
        return reasons

    def _primary_rejection_reasons(self, scored: ScoredCropCandidate) -> list[str]:
        candidate = scored.candidate
        q = candidate.quality
        reasons: list[str] = []
        if scored.score.final_score < self.config.minimum_primary_score:
            reasons.append("score_below_primary_threshold")
        if self.config.require_crop_path and not candidate.vehicle_crop_path:
            reasons.append("missing_crop_path")
        if (q.crop_width or 0) < self.config.minimum_crop_width or (q.crop_height or 0) < self.config.minimum_crop_height:
            reasons.append("crop_too_small")
        if self.config.require_non_edge_touching_for_primary and q.edge_touching:
            reasons.append("edge_touching")
        if self.config.require_complete_crop_for_primary and (q.crop_completeness is None or q.crop_completeness < 1.0):
            reasons.append("crop_incomplete")
        if self.config.minimum_sharpness_for_primary is not None and (q.sharpness is None or q.sharpness < self.config.minimum_sharpness_for_primary):
            reasons.append("sharpness_below_threshold")
        if self.config.minimum_brightness_for_primary is not None and (q.brightness is None or q.brightness < self.config.minimum_brightness_for_primary):
            reasons.append("brightness_out_of_range")
        if self.config.maximum_brightness_for_primary is not None and (q.brightness is None or q.brightness > self.config.maximum_brightness_for_primary):
            reasons.append("brightness_out_of_range")
        if self.config.minimum_contrast_for_primary is not None and (q.contrast is None or q.contrast < self.config.minimum_contrast_for_primary):
            reasons.append("contrast_below_threshold")
        return reasons

    def _fallback_rejection_reasons(self, scored: ScoredCropCandidate) -> list[str]:
        candidate = scored.candidate
        q = candidate.quality
        reasons: list[str] = []
        if self.config.require_crop_path and not candidate.vehicle_crop_path:
            reasons.append("missing_crop_path")
        if (q.crop_width or 0) <= 0 or (q.crop_height or 0) <= 0:
            reasons.append("zero_size_crop")
        if self.config.minimum_fallback_score is not None and scored.score.final_score < self.config.minimum_fallback_score:
            reasons.append("score_below_fallback_threshold")
        if not self.config.allow_edge_touching_fallback and q.edge_touching:
            reasons.append("edge_touching")
        if not self.config.allow_incomplete_fallback and (q.crop_completeness is None or q.crop_completeness < 1.0):
            reasons.append("crop_incomplete")
        return reasons

    def _select_primary(self, candidates: list[ScoredCropCandidate]) -> tuple[list[ScoredCropCandidate], dict[str, int], Counter[str]]:
        selected: list[ScoredCropCandidate] = []
        counts = {"strict": 0, "relaxed": 0}
        rejections: Counter[str] = Counter()
        for item in candidates:
            if len(selected) >= self.config.primary_crop_count:
                break
            reason = self._diversity_rejection(item, selected)
            if reason is not None and self.config.primary_selection_policy != "quality_only":
                rejections[reason] += 1
                continue
            selected.append(item)
            counts["strict"] += 1
        if self.config.allow_relaxed_diversity_backfill and len(selected) < self.config.primary_crop_count:
            selected_keys = {self._candidate_key(item.candidate) for item in selected}
            for item in candidates:
                if len(selected) >= self.config.primary_crop_count:
                    break
                if self._candidate_key(item.candidate) in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(self._candidate_key(item.candidate))
                counts["relaxed"] += 1
        return selected, counts, rejections

    def _diversity_rejection(self, candidate: ScoredCropCandidate, selected: list[ScoredCropCandidate]) -> str | None:
        for existing in selected:
            if self.config.primary_selection_policy in {"quality_with_temporal_diversity", "hybrid"}:
                if abs(candidate.candidate.frame_index - existing.candidate.frame_index) < self.config.minimum_frame_separation:
                    return "insufficient_frame_separation"
                if abs(candidate.candidate.timestamp_sec - existing.candidate.timestamp_sec) < self.config.minimum_temporal_separation_sec:
                    return "insufficient_time_separation"
            if self.config.primary_selection_policy in {"quality_with_visual_diversity", "hybrid"} and self.config.maximum_bbox_overlap_similarity is not None:
                if _bbox_iou(candidate.candidate.crop_bbox or candidate.candidate.bbox, existing.candidate.crop_bbox or existing.candidate.bbox) > self.config.maximum_bbox_overlap_similarity:
                    return "duplicate_visual_candidate"
        return None

    def _select_fallback_candidate(self, candidates: list[ScoredCropCandidate]) -> ScoredCropCandidate | None:
        if not candidates:
            return None
        if self.config.fallback_selection_policy == "earliest_valid":
            return sorted(candidates, key=lambda item: (item.candidate.frame_index, item.candidate.timestamp_sec, -item.score.final_score))[0]
        if self.config.fallback_selection_policy == "largest_valid":
            return sorted(candidates, key=lambda item: (-item.candidate.bbox.area, -item.score.final_score, item.candidate.frame_index))[0]
        if self.config.fallback_selection_policy == "sharpest_valid":
            return sorted(candidates, key=lambda item: (-(item.candidate.quality.sharpness or 0.0), -item.score.final_score, item.candidate.frame_index))[0]
        return sorted(candidates, key=self._tie_break_key)[0]

    def _to_selected_crop(
        self,
        scored: ScoredCropCandidate,
        *,
        role: str,
        rank: int,
        reasons: list[str],
        warnings: list[str],
        diversity_mode: str,
    ) -> SelectedCrop:
        candidate = scored.candidate
        return SelectedCrop(
            source_id=candidate.source_id or "",
            track_id=candidate.track_id,
            track_generation=candidate.track_generation,
            source_track_id=candidate.source_track_id,
            frame_index=candidate.frame_index,
            timestamp_sec=candidate.timestamp_sec,
            vehicle_crop_path=candidate.vehicle_crop_path,
            full_frame_path=candidate.full_frame_path,
            crop_bbox=candidate.crop_bbox or candidate.bbox,
            class_name=candidate.class_name,
            role=role,
            rank=rank,
            final_score=scored.score.final_score,
            score_breakdown=scored.score,
            selection_reasons=list(reasons),
            rejection_warnings=list(warnings),
            metadata={
                "preliminary_rank_score": candidate.preliminary_rank_score,
                "detection_confidence": candidate.detection_confidence,
                "source_bbox": candidate.bbox.to_xyxy(),
                "diversity_mode": diversity_mode,
            },
        )

    def _candidate_key(self, candidate: CropCandidate) -> tuple[int, float, str | None]:
        return (candidate.frame_index, candidate.timestamp_sec, candidate.vehicle_crop_path)

    def _stable_candidate_order(self, candidate: CropCandidate) -> tuple[int, float, str, int]:
        return (candidate.frame_index, candidate.timestamp_sec, candidate.vehicle_crop_path or "", candidate.track_id)

    def _tie_break_key(self, scored: ScoredCropCandidate) -> tuple[float, float, float, float, float, int, str]:
        candidate = scored.candidate
        preliminary = candidate.preliminary_rank_score if candidate.preliminary_rank_score is not None else 0.0
        confidence = candidate.detection_confidence if candidate.detection_confidence is not None else candidate.quality.detection_confidence
        sharpness = candidate.quality.sharpness or 0.0
        return (
            -scored.score.final_score,
            -preliminary,
            -confidence,
            -candidate.bbox.area,
            -sharpness,
            candidate.frame_index,
            candidate.vehicle_crop_path or "",
        )


def _temporal_position_score(index: int, total_count: int) -> float:
    if total_count <= 1:
        return 1.0
    midpoint = (total_count - 1) / 2.0
    max_distance = max(midpoint, 1.0)
    return _clip01(1.0 - (abs(index - midpoint) / max_distance))


def _bundle_to_lifecycle_record(bundle: CompletedTrackCropBundle) -> TrackRecord:
    candidates = list(bundle.candidates)
    frames = [item.frame_index for item in candidates]
    times = [item.timestamp_sec for item in candidates]
    dominant_class = bundle.metadata.get("dominant_class")
    if not dominant_class and candidates:
        dominant_class = candidates[0].class_name
    completion_reason = None
    if bundle.completion_reason:
        completion_reason = TrackCompletionReason(bundle.completion_reason)
    return TrackRecord(
        source_id=bundle.source_id,
        track_id=bundle.track_id,
        source_track_id=bundle.source_track_id,
        track_generation=bundle.track_generation,
        status=TrackStatus.COMPLETED,
        first_seen_frame=min(frames) if frames else 0,
        last_seen_frame=max(frames) if frames else 0,
        first_seen_sec=min(times) if times else 0.0,
        last_seen_sec=max(times) if times else 0.0,
        observation_count=bundle.observation_count,
        missed_frame_count=0,
        class_votes={str(dominant_class): bundle.observation_count} if dominant_class else {},
        crop_candidates=candidates,
        completion_reason=completion_reason,
        last_bbox=candidates[-1].bbox if candidates else None,
        last_confidence=candidates[-1].detection_confidence if candidates else None,
        last_class_name=candidates[-1].class_name if candidates else None,
    )
