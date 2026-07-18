from __future__ import annotations

import re
from typing import Any

from .class_normalization import normalize_class_name
from .searchable_object_schemas import SearchableVehicleRecord


VEHICLE_RECORD_PLATE_STATUSES = {"verified", "weak", "invalid", "no_plate_detected", "ocr_empty", "ocr_failed", "insufficient_evidence"}


def identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id", "")), int(row.get("track_id", 0) or 0), int(row.get("track_generation", 0) or 0))


def build_record_id(source_id: str, track_id: int, track_generation: int) -> str:
    return f"{source_id}:track_{track_id:06d}:gen_{track_generation:03d}"


def build_searchable_object_record(
    lifecycle: dict[str, Any],
    *,
    video_path: str | None,
    selected_crop_set: dict[str, Any] | None = None,
    final_anpr: dict[str, Any] | None = None,
    person_clothing: dict[str, Any] | None = None,
    include_weak_plate_text: bool = True,
) -> SearchableVehicleRecord:
    raw_class_name = (
        (final_anpr or {}).get("raw_class_name")
        or lifecycle.get("raw_class_name")
        or lifecycle.get("last_class_name")
        or _dominant_class(lifecycle.get("class_votes"))
    )
    normalized = normalize_class_name((final_anpr or {}).get("normalized_class_name") or (final_anpr or {}).get("object_class") or raw_class_name)
    if normalized.object_group == "person":
        return build_searchable_person_record(lifecycle, video_path=video_path, selected_crop_set=selected_crop_set, person_clothing=person_clothing)
    return build_searchable_vehicle_record(
        lifecycle,
        video_path=video_path,
        selected_crop_set=selected_crop_set,
        final_anpr=final_anpr,
        include_weak_plate_text=include_weak_plate_text,
    )


def build_searchable_person_record(
    lifecycle: dict[str, Any],
    *,
    video_path: str | None,
    selected_crop_set: dict[str, Any] | None = None,
    person_clothing: dict[str, Any] | None = None,
) -> SearchableVehicleRecord:
    source_id, track_id, track_generation = identity(lifecycle)
    selected = selected_crop_set or {}
    warnings: list[str] = []

    first_frame = _optional_int(lifecycle.get("first_seen_frame"))
    last_frame = _optional_int(lifecycle.get("last_seen_frame"))
    first_sec = _optional_float(lifecycle.get("first_seen_sec"))
    last_sec = _optional_float(lifecycle.get("last_seen_sec"))
    duration = round(last_sec - first_sec, 6) if first_sec is not None and last_sec is not None else None
    if first_frame is None or last_frame is None or first_sec is None or last_sec is None:
        warnings.append("missing_track_times")
    primary_paths = _crop_paths(selected.get("primary_crops"))
    fallback_paths = _crop_paths([selected.get("fallback_crop")] if selected.get("fallback_crop") else [])
    representative_crop = _first_non_empty(primary_paths + fallback_paths)
    if not representative_crop:
        warnings.append("missing_person_crop")
    selected_crop = _selected_crop_for_path(selected, representative_crop)
    selected_frame = _optional_int((selected_crop or {}).get("frame_index")) or first_frame
    selected_timestamp = _optional_float((selected_crop or {}).get("timestamp_sec")) or first_sec
    full_frame_path = _first_full_frame_path(selected, object_crop_path=representative_crop)
    clothing = person_clothing or {}
    tokens = build_person_search_tokens(
        source_id=source_id,
        track_id=track_id,
        track_generation=track_generation,
        first_seen_sec=first_sec,
        upper_clothing_color=clothing.get("upper_clothing_color"),
        lower_clothing_color=clothing.get("lower_clothing_color"),
        dominant_clothing_color=clothing.get("dominant_clothing_color"),
        clothing_color_status=clothing.get("clothing_color_status"),
    )
    return SearchableVehicleRecord(
        record_id=build_record_id(source_id, track_id, track_generation),
        source_id=source_id,
        video_path=video_path,
        track_id=track_id,
        track_generation=track_generation,
        object_class="person",
        first_frame_index=first_frame,
        last_frame_index=last_frame,
        first_seen_sec=first_sec,
        last_seen_sec=last_sec,
        duration_sec=duration,
        plate_text=None,
        plate_status="not_applicable",
        plate_confidence=0.0,
        plate_support_count=0,
        normalized_colour=None,
        raw_colour=None,
        representative_frame_index=selected_frame,
        representative_timestamp_sec=selected_timestamp,
        representative_vehicle_crop_path=representative_crop,
        representative_plate_crop_path=None,
        selected_frame_index=selected_frame,
        selected_timestamp_sec=selected_timestamp,
        primary_crop_paths=primary_paths,
        fallback_crop_paths=fallback_paths,
        object_type="person",
        object_group="person",
        full_frame_path=full_frame_path,
        object_crop_path=representative_crop,
        event_eligible=True,
        raw_class_name=lifecycle.get("raw_class_name") or lifecycle.get("last_class_name") or "person",
        normalized_class_name="person",
        dominant_colour=clothing.get("dominant_clothing_color"),
        colour_confidence=_optional_float(clothing.get("clothing_color_confidence")),
        colour_method=clothing.get("method"),
        colour_region="visible_clothing",
        upper_clothing_color=clothing.get("upper_clothing_color"),
        lower_clothing_color=clothing.get("lower_clothing_color"),
        dominant_clothing_color=clothing.get("dominant_clothing_color"),
        clothing_color_confidence=_optional_float(clothing.get("clothing_color_confidence")),
        clothing_color_status=clothing.get("clothing_color_status") or "not_visible",
        vehicle_colour_status="not_applicable",
        search_text=" ".join(tokens),
        searchable_tokens=tokens,
        warnings=sorted(set(warnings)),
        metadata={
            "source_track_id": lifecycle.get("source_track_id"),
            "lifecycle_status": lifecycle.get("status"),
            "lifecycle_completion_reason": lifecycle.get("completion_reason"),
            "class_votes": lifecycle.get("class_votes", {}),
            "selection_status": selected.get("selection_status"),
            "anpr_bypassed": True,
            "colour_bypassed": True,
            "vehicle_colour_bypassed": True,
            "person_clothing_colour": clothing,
        },
    )


def build_searchable_vehicle_record(
    lifecycle: dict[str, Any],
    *,
    video_path: str | None,
    selected_crop_set: dict[str, Any] | None = None,
    final_anpr: dict[str, Any] | None = None,
    include_weak_plate_text: bool = True,
) -> SearchableVehicleRecord:
    source_id, track_id, track_generation = identity(lifecycle)
    final = final_anpr or {}
    selected = selected_crop_set or {}
    warnings: list[str] = []

    first_frame = _optional_int(lifecycle.get("first_seen_frame"))
    last_frame = _optional_int(lifecycle.get("last_seen_frame"))
    first_sec = _optional_float(lifecycle.get("first_seen_sec"))
    last_sec = _optional_float(lifecycle.get("last_seen_sec"))
    duration = round(last_sec - first_sec, 6) if first_sec is not None and last_sec is not None else None
    if first_frame is None or last_frame is None or first_sec is None or last_sec is None:
        warnings.append("missing_track_times")

    raw_class_name = final.get("raw_class_name") or lifecycle.get("raw_class_name") or lifecycle.get("last_class_name") or _dominant_class(lifecycle.get("class_votes"))
    normalized_class = normalize_class_name(final.get("normalized_class_name") or final.get("object_class") or raw_class_name)
    object_class = normalized_class.normalized_class_name
    if not object_class:
        warnings.append("missing_object_class")

    plate_status = str(final.get("plate_status") or "no_plate_detected")
    if plate_status not in VEHICLE_RECORD_PLATE_STATUSES:
        warnings.append(f"unexpected_plate_status:{plate_status}")
    final_plate = final.get("final_plate_text")
    plate_text = final_plate if plate_status == "verified" else None
    if plate_status == "weak" and include_weak_plate_text:
        plate_text = final_plate
    if plate_status in {"invalid", "no_plate_detected", "ocr_empty", "ocr_failed", "insufficient_evidence"}:
        plate_text = None

    normalized_colour = final.get("normalized_colour")
    raw_colour = final.get("raw_colour")
    if not normalized_colour:
        colour_meta = (final.get("metadata") or {}).get("colour") or {}
        normalized_colour = colour_meta.get("normalized_colour")
        raw_colour = raw_colour or colour_meta.get("raw_colour")
    if not normalized_colour:
        warnings.append("missing_colour")
    colour_meta = (final.get("metadata") or {}).get("colour") or {}
    dominant_colour = final.get("dominant_colour") or colour_meta.get("dominant_colour") or normalized_colour
    colour_confidence = _optional_float(final.get("colour_confidence") or colour_meta.get("colour_confidence"))
    colour_coverage = _optional_float(final.get("colour_coverage") or colour_meta.get("colour_coverage"))
    colour_method = final.get("colour_method") or colour_meta.get("colour_method") or ("florence_raw" if normalized_colour else None)
    colour_region = final.get("colour_region") or colour_meta.get("colour_region")
    colour_warnings = list(final.get("colour_warnings") or colour_meta.get("colour_warnings") or [])

    primary_paths = _crop_paths(selected.get("primary_crops"))
    fallback_paths = _crop_paths([selected.get("fallback_crop")] if selected.get("fallback_crop") else [])
    representative_vehicle_crop_path = final.get("representative_vehicle_crop_path") or _first_non_empty(primary_paths + fallback_paths)
    representative_plate_crop_path = final.get("representative_plate_crop_path")
    if not representative_vehicle_crop_path:
        warnings.append("missing_vehicle_crop")
    if not representative_plate_crop_path and plate_status in {"verified", "weak", "invalid"}:
        warnings.append("missing_plate_crop")

    tokens = build_search_tokens(
        source_id=source_id,
        track_id=track_id,
        track_generation=track_generation,
        object_class=object_class,
        normalized_colour=normalized_colour,
        plate_text=plate_text,
        plate_status=plate_status,
        first_seen_sec=first_sec,
        representative_timestamp_sec=_optional_float(final.get("representative_timestamp_sec")),
        include_weak_plate_text=include_weak_plate_text,
    )
    search_text = " ".join(tokens)
    selected_crop = _selected_crop_for_path(selected, representative_vehicle_crop_path)
    selected_frame = _optional_int(final.get("representative_frame_index")) or _optional_int((selected_crop or {}).get("frame_index")) or first_frame
    selected_timestamp = _optional_float(final.get("representative_timestamp_sec")) or _optional_float((selected_crop or {}).get("timestamp_sec")) or first_sec
    return SearchableVehicleRecord(
        record_id=build_record_id(source_id, track_id, track_generation),
        source_id=source_id,
        video_path=video_path,
        track_id=track_id,
        track_generation=track_generation,
        object_class=object_class,
        first_frame_index=first_frame,
        last_frame_index=last_frame,
        first_seen_sec=first_sec,
        last_seen_sec=last_sec,
        duration_sec=duration,
        plate_text=plate_text,
        plate_status=plate_status,
        plate_confidence=round(float(final.get("confidence") or 0.0), 6),
        plate_support_count=int(final.get("support_count") or 0),
        normalized_colour=dominant_colour,
        raw_colour=raw_colour,
        representative_frame_index=selected_frame,
        representative_timestamp_sec=selected_timestamp,
        representative_vehicle_crop_path=representative_vehicle_crop_path,
        representative_plate_crop_path=representative_plate_crop_path,
        selected_frame_index=selected_frame,
        selected_timestamp_sec=selected_timestamp,
        primary_crop_paths=primary_paths,
        fallback_crop_paths=fallback_paths,
        object_type="vehicle",
        object_group="vehicle",
        full_frame_path=_first_full_frame_path(selected, object_crop_path=representative_vehicle_crop_path),
        object_crop_path=representative_vehicle_crop_path,
        event_eligible=True,
        raw_class_name=raw_class_name,
        normalized_class_name=object_class,
        dominant_colour=dominant_colour,
        colour_confidence=colour_confidence,
        colour_coverage=colour_coverage,
        colour_method=colour_method,
        colour_region=colour_region,
        colour_warnings=colour_warnings,
        vehicle_colour_status=(final.get("metadata") or {}).get("colour", {}).get("colour_status") or final.get("colour_status"),
        search_text=search_text,
        searchable_tokens=tokens,
        warnings=sorted(set(warnings)),
        metadata={
            "source_track_id": lifecycle.get("source_track_id"),
            "lifecycle_status": lifecycle.get("status"),
            "lifecycle_completion_reason": lifecycle.get("completion_reason"),
            "class_votes": lifecycle.get("class_votes", {}),
            "selection_status": selected.get("selection_status"),
            "plate_validation_metadata": final.get("metadata", {}),
        },
    )


def build_search_tokens(
    *,
    source_id: str,
    track_id: int,
    track_generation: int,
    object_class: str | None,
    normalized_colour: str | None,
    plate_text: str | None,
    plate_status: str,
    first_seen_sec: float | None,
    representative_timestamp_sec: float | None,
    include_weak_plate_text: bool = True,
) -> list[str]:
    tokens: list[str] = ["vehicle", plate_status, f"track_{track_id}", f"generation_{track_generation}", f"{source_id}_{track_id}g{track_generation}"]
    if object_class:
        tokens.append(str(object_class).lower())
    if normalized_colour:
        tokens.append(str(normalized_colour).lower())
    if plate_status == "verified" and plate_text:
        tokens.extend(["verified_plate", _tokenize_plate(plate_text)])
        if len(plate_text) >= 2:
            tokens.append(f"state_{plate_text[:2].lower()}")
            tokens.append(plate_text[:2].lower())
    elif plate_status == "weak" and plate_text and include_weak_plate_text:
        tokens.extend(["weak_plate", _tokenize_plate(plate_text)])
        if len(plate_text) >= 2:
            tokens.append(f"state_{plate_text[:2].lower()}")
    elif plate_status == "no_plate_detected":
        tokens.extend(["no_plate", "missing_plate"])
    else:
        tokens.append(f"{plate_status}_plate")
    time_value = representative_timestamp_sec if representative_timestamp_sec is not None else first_seen_sec
    if time_value is not None:
        bucket = int(float(time_value) // 60) * 60
        tokens.extend([f"{round(float(time_value), 1)}s", f"time_{bucket}_{bucket + 60}s"])
    return _dedupe_tokens(tokens)


def build_person_search_tokens(
    *,
    source_id: str,
    track_id: int,
    track_generation: int,
    first_seen_sec: float | None,
    upper_clothing_color: str | None = None,
    lower_clothing_color: str | None = None,
    dominant_clothing_color: str | None = None,
    clothing_color_status: str | None = None,
) -> list[str]:
    tokens = ["person", "people", "pedestrian", f"track_{track_id}", f"generation_{track_generation}", f"{source_id}_{track_id}g{track_generation}"]
    if clothing_color_status:
        tokens.append(f"clothing_{clothing_color_status}")
    if upper_clothing_color and upper_clothing_color != "unknown":
        tokens.extend([upper_clothing_color, f"{upper_clothing_color}_upper", "upper_clothing", f"{upper_clothing_color}_shirt"])
    if lower_clothing_color and lower_clothing_color != "unknown":
        tokens.extend([lower_clothing_color, f"{lower_clothing_color}_lower", "lower_clothing", f"{lower_clothing_color}_trousers"])
    if dominant_clothing_color and dominant_clothing_color != "unknown":
        tokens.extend([dominant_clothing_color, f"{dominant_clothing_color}_clothing"])
    if first_seen_sec is not None:
        bucket = int(float(first_seen_sec) // 60) * 60
        tokens.extend([f"{round(float(first_seen_sec), 1)}s", f"time_{bucket}_{bucket + 60}s"])
    return _dedupe_tokens(tokens)


def run_validation_queries(records: list[SearchableVehicleRecord]) -> dict[str, dict[str, Any]]:
    queries = {
        "white car": lambda r: _has(r, "white") and _has(r, "car"),
        "verified plates": lambda r: r.plate_status == "verified",
        "UP81CH4158": lambda r: _has(r, "up81ch4158"),
        "UP81": lambda r: any(token.startswith("up81") for token in r.searchable_tokens),
        "red vehicle": lambda r: _has(r, "red") and _has(r, "vehicle"),
        "vehicles between 60 and 120 seconds": lambda r: r.first_seen_sec is not None and 60.0 <= r.first_seen_sec <= 120.0,
        "weak OCR": lambda r: r.plate_status == "weak",
        "no plate": lambda r: r.plate_status == "no_plate_detected",
    }
    payload: dict[str, dict[str, Any]] = {}
    for query, predicate in queries.items():
        matches = [record.record_id for record in records if predicate(record)]
        payload[query] = {"count": len(matches), "record_ids": matches[:25]}
    return payload


def _has(record: SearchableVehicleRecord, token: str) -> bool:
    return token.lower() in record.searchable_tokens


def _tokenize_plate(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    retained: list[str] = []
    for token in tokens:
        normalized = str(token).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        retained.append(normalized)
    return retained


def _crop_paths(crops: Any) -> list[str]:
    paths: list[str] = []
    for crop in crops or []:
        if not isinstance(crop, dict):
            continue
        path = crop.get("vehicle_crop_path")
        if path:
            paths.append(str(path))
    return paths


def _first_non_empty(values: list[str]) -> str | None:
    return next((value for value in values if value), None)


def _first_full_frame_path(selected: dict[str, Any], *, object_crop_path: str | None = None) -> str | None:
    crops: list[Any] = []
    crops.extend(selected.get("primary_crops") or [])
    if selected.get("fallback_crop"):
        crops.append(selected.get("fallback_crop"))
    if object_crop_path:
        for crop in crops:
            if isinstance(crop, dict) and crop.get("vehicle_crop_path") == object_crop_path:
                for key in ("full_frame_path", "source_frame_path", "evidence_frame_path"):
                    if crop.get(key):
                        return str(crop[key])
    for crop in crops:
        if not isinstance(crop, dict):
            continue
        for key in ("full_frame_path", "source_frame_path", "evidence_frame_path"):
            if crop.get(key):
                return str(crop[key])
    return None


def _selected_crop_for_path(selected: dict[str, Any], object_crop_path: str | None) -> dict[str, Any] | None:
    if not object_crop_path:
        return None
    crops: list[Any] = []
    crops.extend(selected.get("primary_crops") or [])
    if selected.get("fallback_crop"):
        crops.append(selected.get("fallback_crop"))
    for crop in crops:
        if isinstance(crop, dict) and crop.get("vehicle_crop_path") == object_crop_path:
            return crop
    return None


def _dominant_class(class_votes: Any) -> str | None:
    if not isinstance(class_votes, dict) or not class_votes:
        return None
    return sorted(class_votes.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[0][0]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
