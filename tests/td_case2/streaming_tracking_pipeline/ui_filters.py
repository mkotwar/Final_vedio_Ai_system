from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .search_query_parser import parse_vehicle_search_query
from .search_result_ranker import assign_ranks, rank_vehicle_record
from .structured_search_index import StructuredVehicleSearchIndex


PlatePresence = Literal["any", "with_plate", "without_plate"]
SortMode = Literal["relevance", "time", "confidence", "track_id"]


@dataclass(frozen=True)
class UIRecordFilters:
    text_query: str = ""
    object_class: str | None = None
    dominant_colour: str | None = None
    plate_status: str | None = None
    exact_plate: str | None = None
    plate_prefix: str | None = None
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    track_id: int | None = None
    track_generation: int | None = None
    minimum_confidence: float | None = None
    verified_only: bool = False
    include_weak_plates: bool = True
    plate_presence: PlatePresence = "any"
    sort_by: SortMode = "relevance"
    top_k: int | None = None


def filter_records(records: list[dict[str, Any]], filters: UIRecordFilters) -> list[dict[str, Any]]:
    ranked_records = _apply_text_query(records, filters)
    filtered = [record for record in ranked_records if _matches_widget_filters(record, filters)]
    sorted_records = sort_records(filtered, filters.sort_by)
    if filters.top_k and filters.top_k > 0:
        return sorted_records[: filters.top_k]
    return sorted_records


def paginate_records(records: list[dict[str, Any]], *, page: int = 1, page_size: int = 25) -> tuple[list[dict[str, Any]], dict[str, int]]:
    safe_page_size = max(1, int(page_size))
    total_pages = max(1, (len(records) + safe_page_size - 1) // safe_page_size)
    safe_page = min(max(1, int(page)), total_pages)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return records[start:end], {
        "page": safe_page,
        "page_size": safe_page_size,
        "total_pages": total_pages,
        "total_records": len(records),
    }


def sort_records(records: list[dict[str, Any]], sort_by: SortMode) -> list[dict[str, Any]]:
    if sort_by == "time":
        return sorted(records, key=lambda record: (_float_or_inf(record.get("first_seen_sec")), _identity_sort(record)))
    if sort_by == "confidence":
        return sorted(records, key=lambda record: (-_confidence(record), _identity_sort(record)))
    if sort_by == "track_id":
        return sorted(records, key=_identity_sort)
    return sorted(records, key=lambda record: (int(record.get("_ui_rank") or 999999), -float(record.get("_ui_score") or 0.0), _identity_sort(record)))


def available_filter_values(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "classes": _unique(record_class(record) for record in records if record_class(record)),
        "colours": _unique(record_colour(record) for record in records if record_colour(record)),
        "plate_statuses": _unique(str(record.get("plate_status") or "") for record in records if record.get("plate_status")),
    }


def record_class(record: dict[str, Any]) -> str:
    return str(record.get("normalized_class_name") or record.get("object_class") or "").lower()


def record_colour(record: dict[str, Any]) -> str:
    confidence = record.get("colour_confidence")
    if confidence is not None:
        try:
            if float(confidence) < 0.18:
                return ""
        except (TypeError, ValueError):
            return ""
    return str(record.get("dominant_clothing_color") or record.get("dominant_colour") or record.get("normalized_colour") or "").lower()


def record_has_plate(record: dict[str, Any], *, include_weak_plates: bool = True) -> bool:
    status = str(record.get("plate_status") or "")
    if status == "verified":
        return bool(record.get("plate_text"))
    if status == "weak" and include_weak_plates:
        return bool(record.get("plate_text"))
    return False


def time_overlaps(record: dict[str, Any], start_time_sec: float | None, end_time_sec: float | None) -> bool:
    if start_time_sec is None and end_time_sec is None:
        return True
    first_seen = record.get("first_seen_sec")
    last_seen = record.get("last_seen_sec")
    if first_seen is None and last_seen is None:
        return False
    record_start = float(first_seen if first_seen is not None else last_seen)
    record_end = float(last_seen if last_seen is not None else first_seen)
    query_start = start_time_sec if start_time_sec is not None else float("-inf")
    query_end = end_time_sec if end_time_sec is not None else float("inf")
    return record_start <= query_end and record_end >= query_start


def _apply_text_query(records: list[dict[str, Any]], filters: UIRecordFilters) -> list[dict[str, Any]]:
    query = filters.text_query.strip()
    if not query:
        ranked = assign_ranks([rank_vehicle_record(record, parse_vehicle_search_query(""), include_weak_plates=filters.include_weak_plates) for record in records])
    else:
        response = StructuredVehicleSearchIndex(records, include_weak_plates=filters.include_weak_plates).search(query)
        ranked_by_id = {result.record_id: result for result in response.results}
        return [_attach_search_fields(record, ranked_by_id[str(record.get("record_id"))]) for record in records if str(record.get("record_id")) in ranked_by_id]
    ranked_by_id = {result.record_id: result for result in ranked}
    return [_attach_search_fields(record, ranked_by_id.get(str(record.get("record_id")))) for record in records]


def _attach_search_fields(record: dict[str, Any], result: Any | None) -> dict[str, Any]:
    payload = dict(record)
    if result is not None:
        payload["_ui_rank"] = result.rank
        payload["_ui_score"] = result.score
        payload["_ui_matched_filters"] = list(result.matched_filters)
        payload["_ui_matched_tokens"] = list(result.matched_tokens)
    return payload


def _matches_widget_filters(record: dict[str, Any], filters: UIRecordFilters) -> bool:
    if filters.object_class and record_class(record) != filters.object_class:
        return False
    if filters.dominant_colour and record_colour(record) != filters.dominant_colour:
        return False
    if filters.plate_status and str(record.get("plate_status") or "") != filters.plate_status:
        return False
    if filters.verified_only and str(record.get("plate_status") or "") != "verified":
        return False
    if filters.exact_plate and not _matches_exact_plate(record, filters.exact_plate, filters.include_weak_plates):
        return False
    if filters.plate_prefix and not _matches_plate_prefix(record, filters.plate_prefix, filters.include_weak_plates):
        return False
    if not time_overlaps(record, filters.start_time_sec, filters.end_time_sec):
        return False
    if filters.track_id is not None and int(record.get("track_id") or -1) != filters.track_id:
        return False
    if filters.track_generation is not None and int(record.get("track_generation") or -1) != filters.track_generation:
        return False
    if filters.minimum_confidence is not None and _confidence(record) < float(filters.minimum_confidence):
        return False
    if filters.plate_presence == "with_plate" and not record_has_plate(record, include_weak_plates=filters.include_weak_plates):
        return False
    if filters.plate_presence == "without_plate" and record_has_plate(record, include_weak_plates=filters.include_weak_plates):
        return False
    return True


def _matches_exact_plate(record: dict[str, Any], plate_text: str, include_weak_plates: bool) -> bool:
    if str(record.get("plate_status") or "") == "invalid":
        return False
    if str(record.get("plate_status") or "") == "weak" and not include_weak_plates:
        return False
    if str(record.get("plate_status") or "") not in {"verified", "weak"}:
        return False
    return _normalize_plate(record.get("plate_text")) == _normalize_plate(plate_text)


def _matches_plate_prefix(record: dict[str, Any], plate_prefix: str, include_weak_plates: bool) -> bool:
    if str(record.get("plate_status") or "") == "invalid":
        return False
    if str(record.get("plate_status") or "") == "weak" and not include_weak_plates:
        return False
    if str(record.get("plate_status") or "") not in {"verified", "weak"}:
        return False
    return _normalize_plate(record.get("plate_text")).startswith(_normalize_plate(plate_prefix))


def _normalize_plate(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()


def _confidence(record: dict[str, Any]) -> float:
    for key in ("plate_confidence", "confidence", "colour_confidence"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _float_or_inf(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _identity_sort(record: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(record.get("source_id") or ""),
        int(record.get("track_id") or 0),
        int(record.get("track_generation") or 0),
        str(record.get("record_id") or ""),
    )


def _unique(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value is not None and str(value) != ""})
