from __future__ import annotations

import re
from typing import Any

from .search_query_schemas import VehicleSearchQuery, VehicleSearchResult


def rank_vehicle_record(
    record: dict[str, Any],
    query: VehicleSearchQuery,
    *,
    include_weak_plates: bool = True,
) -> VehicleSearchResult:
    components: dict[str, float] = {}
    matched_filters: list[str] = []
    matched_tokens = _matched_tokens(record, query.free_text_tokens)
    plate_status = str(record.get("plate_status") or "")
    plate_text = _normalize_plate(record.get("plate_text"))

    if query.plate_text and plate_text == query.plate_text:
        if plate_status == "verified":
            components["exact_verified_plate"] = 100.0
        elif plate_status == "weak" and include_weak_plates:
            components["exact_weak_plate"] = 80.0
        matched_filters.append("plate_text")
    if query.plate_prefix and plate_text.startswith(query.plate_prefix):
        components["plate_prefix"] = 60.0 if plate_status == "verified" else 45.0
        matched_filters.append("plate_prefix")
    display_class = record.get("normalized_class_name") or record.get("object_class")
    display_colour = record.get("dominant_clothing_color") or record.get("dominant_colour") or record.get("normalized_colour")
    if query.object_classes and _lower(display_class) in query.object_classes:
        components["class"] = 20.0
        matched_filters.append("object_class")
    if query.colours and _lower(display_colour) in query.colours:
        components["colour"] = 18.0
        matched_filters.append("colour")
    if query.plate_statuses and plate_status in query.plate_statuses:
        components["plate_status"] = 15.0
        matched_filters.append("plate_status")
    if query.start_time_sec is not None or query.end_time_sec is not None:
        components["time_overlap"] = 10.0
        matched_filters.append("time_overlap")
    if query.track_id is not None and int(record.get("track_id") or -1) == query.track_id:
        components["track_id"] = 12.0
        matched_filters.append("track_id")
    if query.track_generation is not None and int(record.get("track_generation") or -1) == query.track_generation:
        components["track_generation"] = 8.0
        matched_filters.append("track_generation")
    if matched_tokens:
        components["free_text_tokens"] = float(len(matched_tokens) * 2)

    evidence = 0.0
    if record.get("representative_vehicle_crop_path"):
        evidence += 2.0
    if record.get("representative_plate_crop_path"):
        evidence += 2.0
    if record.get("normalized_colour"):
        evidence += 1.0
    if evidence:
        components["evidence_completeness"] = evidence

    score = round(sum(components.values()), 6)
    return VehicleSearchResult(
        rank=0,
        score=score,
        record_id=str(record.get("record_id") or ""),
        source_id=str(record.get("source_id") or ""),
        track_id=int(record.get("track_id") or 0),
        track_generation=int(record.get("track_generation") or 0),
        object_class=display_class,
        colour=display_colour,
        plate_text=record.get("plate_text"),
        plate_status=plate_status,
        first_seen_sec=record.get("first_seen_sec"),
        last_seen_sec=record.get("last_seen_sec"),
        representative_vehicle_crop_path=record.get("representative_vehicle_crop_path"),
        representative_plate_crop_path=record.get("representative_plate_crop_path"),
        matched_filters=sorted(set(matched_filters)),
        matched_tokens=matched_tokens,
        warnings=list(record.get("warnings") or []),
        score_components=components,
    )


def assign_ranks(results: list[VehicleSearchResult]) -> list[VehicleSearchResult]:
    ranked = sorted(
        results,
        key=lambda result: (
            -result.score,
            _status_priority(result.plate_status),
            result.record_id,
        ),
    )
    return [
        VehicleSearchResult(
            rank=index,
            score=result.score,
            record_id=result.record_id,
            source_id=result.source_id,
            track_id=result.track_id,
            track_generation=result.track_generation,
            object_class=result.object_class,
            colour=result.colour,
            plate_text=result.plate_text,
            plate_status=result.plate_status,
            first_seen_sec=result.first_seen_sec,
            last_seen_sec=result.last_seen_sec,
            representative_vehicle_crop_path=result.representative_vehicle_crop_path,
            representative_plate_crop_path=result.representative_plate_crop_path,
            matched_filters=result.matched_filters,
            matched_tokens=result.matched_tokens,
            warnings=result.warnings,
            score_components=result.score_components,
        )
        for index, result in enumerate(ranked, start=1)
    ]


def _matched_tokens(record: dict[str, Any], query_tokens: list[str]) -> list[str]:
    searchable = {str(token).lower() for token in record.get("searchable_tokens") or []}
    search_text = str(record.get("search_text") or "").lower()
    matched: list[str] = []
    for token in query_tokens:
        normalized = str(token).lower()
        if normalized in searchable or re.search(rf"\b{re.escape(normalized)}\b", search_text):
            matched.append(normalized)
    return matched


def _normalize_plate(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _status_priority(status: str) -> int:
    return {"verified": 0, "weak": 1, "invalid": 2, "no_plate_detected": 3}.get(status, 4)
