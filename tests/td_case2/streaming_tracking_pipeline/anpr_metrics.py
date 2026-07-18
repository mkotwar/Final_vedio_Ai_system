from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .anpr_schemas import TrackAnprColourResult


def confidence_bucket(value: float) -> str:
    if value >= 0.80:
        return "0.80-1.00"
    if value >= 0.60:
        return "0.60-0.79"
    if value >= 0.40:
        return "0.40-0.59"
    if value >= 0.20:
        return "0.20-0.39"
    return "0.00-0.19"


def build_step7_metrics(results: Iterable[TrackAnprColourResult], *, extra_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = list(results)
    status_counts = Counter(row.processing_status for row in rows)
    colour_counts = Counter(row.normalized_colour for row in rows)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_generation: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_buckets = Counter()
    ranks = Counter()
    failures = Counter()

    plate_candidate_count = 0
    ocr_result_count = 0
    raw_plate_text_count = 0
    tracks_with_raw_plate_text = 0
    tracks_with_colour = 0
    for row in rows:
        object_class = row.object_class or "unknown"
        by_class[object_class][row.processing_status] += 1
        by_generation[str(row.track_generation)][row.processing_status] += 1
        if row.raw_plate_texts:
            tracks_with_raw_plate_text += 1
        if row.normalized_colour != "unknown":
            tracks_with_colour += 1
        for reason in row.failure_reasons:
            failures[reason.split(":", 1)[0]] += 1
        plate_candidate_count += len(row.plate_candidates)
        ocr_result_count += len(row.ocr_results)
        raw_plate_text_count += len(row.raw_plate_texts)
        for candidate in row.plate_candidates:
            by_role[candidate.crop_role][row.processing_status] += 1
            ranks[f"{candidate.crop_role}:{candidate.crop_rank}"] += 1
            confidence_buckets[confidence_bucket(candidate.confidence)] += 1

    payload: dict[str, Any] = {
        "tracks_processed": len(rows),
        "status_counts": dict(status_counts),
        "plate_candidate_count": plate_candidate_count,
        "ocr_result_count": ocr_result_count,
        "raw_plate_text_count": raw_plate_text_count,
        "tracks_with_raw_plate_text": tracks_with_raw_plate_text,
        "tracks_with_vehicle_colour": tracks_with_colour,
        "colour_counts": dict(colour_counts),
        "by_crop_role": {key: dict(value) for key, value in sorted(by_role.items())},
        "by_object_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "by_track_generation": {key: dict(value) for key, value in sorted(by_generation.items())},
        "plate_confidence_buckets": dict(confidence_buckets),
        "plate_candidate_rank_counts": dict(ranks),
        "failure_reason_counts": dict(failures),
    }
    if extra_metrics:
        payload["stage_metrics"] = extra_metrics
    return payload
