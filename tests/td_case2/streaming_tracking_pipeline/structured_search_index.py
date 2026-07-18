from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .search_query_parser import parse_vehicle_search_query
from .search_query_schemas import VehicleSearchQuery, VehicleSearchResponse
from .search_result_ranker import assign_ranks, rank_vehicle_record
from .serialization import read_jsonl


STEP10_INPUT_RELATIVE_PATH = Path("09_searchable_objects") / "searchable_vehicle_records.jsonl"


class StructuredVehicleSearchIndex:
    def __init__(self, records: list[dict[str, Any]], *, include_weak_plates: bool = True) -> None:
        self.records = [dict(record) for record in records]
        self.include_weak_plates = include_weak_plates

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, *, include_weak_plates: bool = True) -> "StructuredVehicleSearchIndex":
        return cls(read_jsonl(Path(run_dir) / STEP10_INPUT_RELATIVE_PATH), include_weak_plates=include_weak_plates)

    def search(self, query: str | VehicleSearchQuery, *, top_k: int | None = None) -> VehicleSearchResponse:
        parsed = parse_vehicle_search_query(query) if isinstance(query, str) else query
        started = time.perf_counter()
        candidates = [record for record in self.records if self._record_matches(record, parsed)]
        ranked = assign_ranks(
            [rank_vehicle_record(record, parsed, include_weak_plates=self.include_weak_plates) for record in candidates]
        )
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]
        runtime = round(time.perf_counter() - started, 6)
        return VehicleSearchResponse(
            query=parsed,
            total_records_searched=len(self.records),
            total_matches=len(candidates),
            results=ranked,
            runtime_sec=runtime,
            warnings=list(parsed.warnings),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "records_indexed": len(self.records),
            "include_weak_plates": self.include_weak_plates,
            "records_by_class": _count_by_class(self.records),
            "records_by_colour": _count_by_colour(self.records),
            "records_by_plate_status": _count_by(self.records, "plate_status"),
        }

    def _record_matches(self, record: dict[str, Any], query: VehicleSearchQuery) -> bool:
        if query.object_classes and _record_class(record) not in query.object_classes:
            return False
        if query.colours and _record_search_colour(record) not in query.colours:
            return False
        if query.plate_statuses and str(record.get("plate_status") or "") not in query.plate_statuses:
            return False
        if query.track_id is not None and int(record.get("track_id") or -1) != query.track_id:
            return False
        if query.track_generation is not None and int(record.get("track_generation") or -1) != query.track_generation:
            return False
        if (query.start_time_sec is not None or query.end_time_sec is not None) and not _overlaps_time(record, query):
            return False
        if query.plate_text and not self._matches_plate_text(record, query.plate_text):
            return False
        if query.plate_prefix and not self._matches_plate_prefix(record, query.plate_prefix):
            return False
        if query.free_text_tokens and not _has_structured_filters(query) and not _matches_any_token(record, query.free_text_tokens):
            return False
        return True

    def _matches_plate_text(self, record: dict[str, Any], plate_text: str) -> bool:
        status = str(record.get("plate_status") or "")
        if status not in {"verified", "weak"}:
            return False
        if status == "weak" and not self.include_weak_plates:
            return False
        return _normalize_plate(record.get("plate_text")) == plate_text

    def _matches_plate_prefix(self, record: dict[str, Any], plate_prefix: str) -> bool:
        status = str(record.get("plate_status") or "")
        if status not in {"verified", "weak"}:
            return False
        if status == "weak" and not self.include_weak_plates:
            return False
        return _normalize_plate(record.get("plate_text")).startswith(plate_prefix)


def _overlaps_time(record: dict[str, Any], query: VehicleSearchQuery) -> bool:
    first_seen = record.get("first_seen_sec")
    last_seen = record.get("last_seen_sec")
    if first_seen is None and last_seen is None:
        return False
    start = float(first_seen if first_seen is not None else last_seen)
    end = float(last_seen if last_seen is not None else first_seen)
    query_start = query.start_time_sec if query.start_time_sec is not None else float("-inf")
    query_end = query.end_time_sec if query.end_time_sec is not None else float("inf")
    return start <= query_end and end >= query_start


def _matches_any_token(record: dict[str, Any], tokens: list[str]) -> bool:
    searchable = {str(token).lower() for token in record.get("searchable_tokens") or []}
    search_text = str(record.get("search_text") or "").lower()
    return any(token in searchable or token in search_text for token in tokens)


def _has_structured_filters(query: VehicleSearchQuery) -> bool:
    return bool(
        query.object_classes
        or query.colours
        or query.plate_text
        or query.plate_prefix
        or query.plate_statuses
        or query.start_time_sec is not None
        or query.end_time_sec is not None
        or query.track_id is not None
        or query.track_generation is not None
    )


def _normalize_plate(value: Any) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()


def _count_by(records: list[dict[str, Any]], key: str, *, missing: str | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key) or missing
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _count_by_class(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = _record_class(record) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_by_colour(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = _record_search_colour(record) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _record_class(record: dict[str, Any]) -> str:
    return str(record.get("normalized_class_name") or record.get("object_class") or "").lower()


def _record_search_colour(record: dict[str, Any]) -> str:
    colour = str(record.get("dominant_clothing_color") or record.get("dominant_colour") or record.get("normalized_colour") or "").lower()
    confidence = record.get("colour_confidence")
    if confidence is not None:
        try:
            if float(confidence) < 0.18:
                return ""
        except (TypeError, ValueError):
            return ""
    return colour
