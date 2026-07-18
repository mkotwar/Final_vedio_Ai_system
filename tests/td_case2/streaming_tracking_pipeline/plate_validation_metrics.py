from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .plate_agreement import agreement_score
from .plate_validation_schemas import FinalTrackAnprResult, PlateAgreementResult, PlateTextCandidate, PlateValidationConfig


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_candidate(
    candidate: PlateTextCandidate,
    agreement: PlateAgreementResult | None,
    config: PlateValidationConfig | None = None,
) -> dict[str, float]:
    cfg = config or PlateValidationConfig()
    detector = clip01(candidate.plate_detection_confidence or 0.0)
    text = candidate.text_for_selection()
    length_score = 1.0 if 8 <= len(text) <= 10 else 0.65 if 6 <= len(text) <= 11 else 0.25
    noisy_penalty = 0.25 if len(candidate.substitutions) > 1 else 0.0
    ocr_quality = clip01(length_score - noisy_penalty)
    role_score = 1.0 if candidate.crop_role == "primary" else 0.75 if candidate.crop_role == "fallback" else 0.5
    rank_score = 1.0 if candidate.crop_rank in (None, 1) else max(0.25, 1.0 - (0.15 * (candidate.crop_rank - 1)))
    crop_quality = clip01((role_score + rank_score) / 2.0)
    agree = agreement_score(candidate, agreement)
    final = (
        candidate.format_score * cfg.format_weight
        + detector * cfg.detector_weight
        + ocr_quality * cfg.ocr_quality_weight
        + crop_quality * cfg.crop_quality_weight
        + agree * cfg.agreement_weight
    )
    return {
        "format_score": round(candidate.format_score, 6),
        "detector_score": round(detector, 6),
        "ocr_quality_score": round(ocr_quality, 6),
        "crop_quality_score": round(crop_quality, 6),
        "agreement_score": round(agree, 6),
        "final_candidate_score": round(clip01(final), 6),
    }


def build_plate_validation_metrics(
    candidates: list[PlateTextCandidate],
    agreements: list[PlateAgreementResult],
    finals: list[FinalTrackAnprResult],
    *,
    track_generations_processed: int,
    tracks_with_plate_detection: int,
    tracks_without_plate_detection: int,
    raw_ocr_candidate_count: int | None = None,
) -> dict[str, Any]:
    status_counts = Counter(result.plate_status for result in finals)
    format_counts = Counter(candidate.format_status for candidate in candidates)
    correction_counts = Counter(str(len(candidate.substitutions)) for candidate in candidates)
    correction_types = Counter(f"{item.get('from')}->{item.get('to')}" for candidate in candidates for item in candidate.substitutions)
    by_crop_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_crop_rank: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_conf: dict[str, Counter[str]] = defaultdict(Counter)
    by_ocr_length: dict[str, Counter[str]] = defaultdict(Counter)
    by_generation: dict[str, Counter[str]] = defaultdict(Counter)
    for result in finals:
        cls = result.object_class or "unknown"
        by_class[cls][result.plate_status] += 1
        by_generation[str(result.track_generation)][result.plate_status] += 1
        candidate = result.selected_candidate
        if candidate is None:
            continue
        by_crop_role[str(candidate.crop_role or "unknown")][result.plate_status] += 1
        by_crop_rank[str(candidate.crop_rank if candidate.crop_rank is not None else "unknown")][result.plate_status] += 1
        by_conf[_confidence_bucket(candidate.plate_detection_confidence)][result.plate_status] += 1
        by_ocr_length[_length_bucket(len(candidate.normalized_text))][result.plate_status] += 1
    score_values = [round(result.confidence, 2) for result in finals]
    text_lengths = Counter(str(len(candidate.text_for_selection())) for candidate in candidates)
    return {
        "track_generations_processed": track_generations_processed,
        "tracks_with_plate_detection": tracks_with_plate_detection,
        "tracks_without_plate_detection": tracks_without_plate_detection,
        "raw_ocr_candidates": raw_ocr_candidate_count if raw_ocr_candidate_count is not None else len([candidate for candidate in candidates if candidate.raw_ocr_text]),
        "normalized_candidates": len(candidates),
        "corrected_candidates": len([candidate for candidate in candidates if candidate.corrected_text]),
        "strict_format_matches": format_counts["strict_format_match"],
        "relaxed_format_matches": format_counts["relaxed_format_match"],
        "partial_plate_candidates": format_counts["partial_plate"],
        "not_plate_like_candidates": format_counts["not_plate_like"],
        "verified_results": status_counts["verified"],
        "weak_results": status_counts["weak"],
        "invalid_results": status_counts["invalid"],
        "no_plate_results": status_counts["no_plate_detected"],
        "exact_agreement_groups": sum(1 for item in agreements if any(count >= 2 for count in item.exact_groups.values())),
        "similarity_based_agreements": sum(1 for item in agreements if "similarity_agreement" in item.disagreement_reasons or "one_character_disagreement" in item.disagreement_reasons),
        "single_candidate_results": sum(1 for item in agreements if item.candidate_count == 1),
        "multi_candidate_results": sum(1 for item in agreements if item.candidate_count > 1),
        "correction_counts": dict(correction_counts),
        "correction_types": dict(correction_types),
        "status_counts": dict(status_counts),
        "format_status_counts": dict(format_counts),
        "plate_text_length_distribution": dict(text_lengths),
        "score_distribution": _score_distribution(score_values),
        "by_crop_role": {key: dict(value) for key, value in sorted(by_crop_role.items())},
        "by_crop_rank": {key: dict(value) for key, value in sorted(by_crop_rank.items())},
        "by_vehicle_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "by_plate_detector_confidence_bucket": {key: dict(value) for key, value in sorted(by_conf.items())},
        "by_ocr_text_length": {key: dict(value) for key, value in sorted(by_ocr_length.items())},
        "by_track_generation": {key: dict(value) for key, value in sorted(by_generation.items())},
    }


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 0.80:
        return "0.80-1.00"
    if value >= 0.60:
        return "0.60-0.79"
    if value >= 0.40:
        return "0.40-0.59"
    if value >= 0.20:
        return "0.20-0.39"
    return "0.00-0.19"


def _length_bucket(length: int) -> str:
    if length <= 3:
        return "0-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def _score_distribution(scores: list[float]) -> dict[str, int]:
    buckets = Counter()
    for score in scores:
        if score >= 0.80:
            buckets["0.80-1.00"] += 1
        elif score >= 0.60:
            buckets["0.60-0.79"] += 1
        elif score >= 0.40:
            buckets["0.40-0.59"] += 1
        elif score >= 0.20:
            buckets["0.20-0.39"] += 1
        else:
            buckets["0.00-0.19"] += 1
    return dict(buckets)
