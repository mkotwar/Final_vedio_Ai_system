from __future__ import annotations

from typing import Any

from .search_result_card_schemas import VehicleResultCard, VehicleResultCardPackage


def build_vehicle_result_card(search_result: dict[str, Any], source_record: dict[str, Any] | None = None) -> VehicleResultCard:
    record = source_record or {}
    warnings = sorted(set(list(search_result.get("warnings") or []) + list(record.get("warnings") or [])))
    vehicle_path = search_result.get("representative_vehicle_crop_path") or record.get("representative_vehicle_crop_path")
    plate_path = search_result.get("representative_plate_crop_path") or record.get("representative_plate_crop_path")
    if not vehicle_path:
        warnings.append("missing_vehicle_image")
    if not plate_path:
        warnings.append("missing_plate_image")

    object_class = search_result.get("object_class") or record.get("object_class")
    colour = search_result.get("colour") or record.get("normalized_colour")
    plate_status = str(search_result.get("plate_status") or record.get("plate_status") or "unknown")
    plate_text = _display_plate_text(search_result.get("plate_text") or record.get("plate_text"), plate_status)
    first_seen = _optional_float(search_result.get("first_seen_sec") if search_result.get("first_seen_sec") is not None else record.get("first_seen_sec"))
    last_seen = _optional_float(search_result.get("last_seen_sec") if search_result.get("last_seen_sec") is not None else record.get("last_seen_sec"))
    duration = _optional_float(record.get("duration_sec"))
    confidence = _optional_float(record.get("plate_confidence"))
    track_id = int(search_result.get("track_id") or record.get("track_id") or 0)
    generation = int(search_result.get("track_generation") or record.get("track_generation") or 0)

    return VehicleResultCard(
        rank=int(search_result.get("rank") or 0),
        record_id=str(search_result.get("record_id") or record.get("record_id") or ""),
        source_id=str(search_result.get("source_id") or record.get("source_id") or ""),
        track_id=track_id,
        track_generation=generation,
        title=_build_title(colour, object_class, plate_text, plate_status),
        subtitle=f"Track {track_id} · {_format_range(first_seen, last_seen)}",
        time_label=_format_range(first_seen, last_seen),
        plate_label=_build_plate_label(plate_text, plate_status),
        colour_label=_title_or_unknown(colour),
        status_badge=_status_badge(plate_status),
        confidence_label=_confidence_label(confidence),
        object_class=object_class,
        colour=colour,
        plate_text=plate_text,
        plate_status=plate_status,
        plate_confidence=confidence,
        first_seen_sec=first_seen,
        last_seen_sec=last_seen,
        duration_sec=duration,
        thumbnail_path=str(vehicle_path) if vehicle_path else None,
        secondary_image_path=str(plate_path) if plate_path else None,
        search_score=float(search_result.get("score") or 0.0),
        matched_filters=list(search_result.get("matched_filters") or []),
        matched_tokens=list(search_result.get("matched_tokens") or []),
        warnings=sorted(set(warnings)),
        metadata={
            "score_components": search_result.get("score_components") or {},
            "source_record_warnings": record.get("warnings") or [],
            "result_warnings": search_result.get("warnings") or [],
        },
    )


def build_vehicle_result_card_package(
    search_response: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    *,
    top_k: int = 20,
) -> VehicleResultCardPackage:
    cards = [
        build_vehicle_result_card(result, records_by_id.get(str(result.get("record_id") or "")))
        for result in list(search_response.get("results") or [])[:top_k]
    ]
    return VehicleResultCardPackage(
        raw_query=str((search_response.get("query") or {}).get("raw_query") or ""),
        parsed_query=dict(search_response.get("query") or {}),
        total_matches=int(search_response.get("total_matches") or 0),
        returned_cards=len(cards),
        cards=cards,
        runtime_sec=float(search_response.get("runtime_sec") or 0.0),
        warnings=list(search_response.get("warnings") or []),
    )


def _display_plate_text(value: Any, plate_status: str) -> str | None:
    if plate_status in {"verified", "weak"} and value:
        return str(value)
    return None


def _build_title(colour: str | None, object_class: str | None, plate_text: str | None, plate_status: str) -> str:
    vehicle = " ".join(part for part in [_title_or_none(colour), _title_or_none(object_class)] if part) or "Vehicle"
    plate = plate_text if plate_text else _build_plate_label(None, plate_status)
    return f"{vehicle} - {plate}"


def _build_plate_label(plate_text: str | None, plate_status: str) -> str:
    if plate_status == "verified" and plate_text:
        return plate_text
    if plate_status == "weak" and plate_text:
        return f"{plate_text} (Weak OCR)"
    if plate_status == "invalid":
        return "Invalid plate candidate"
    if plate_status == "no_plate_detected":
        return "No plate detected"
    return plate_status.replace("_", " ").title()


def _status_badge(plate_status: str) -> str:
    return {
        "verified": "Verified",
        "weak": "Weak OCR",
        "invalid": "Invalid",
        "no_plate_detected": "No Plate",
    }.get(plate_status, plate_status.replace("_", " ").title())


def _confidence_label(value: float | None) -> str:
    if value is None:
        return "Confidence unavailable"
    return f"{value:.3f}"


def _format_range(first_seen: float | None, last_seen: float | None) -> str:
    if first_seen is None and last_seen is None:
        return "time unavailable"
    if first_seen is None:
        return f"?-{last_seen:.1f}s"
    if last_seen is None:
        return f"{first_seen:.1f}s-?"
    return f"{first_seen:.1f}s-{last_seen:.1f}s"


def _title_or_unknown(value: Any) -> str:
    return _title_or_none(value) or "Unknown"


def _title_or_none(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).replace("_", " ").title()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
