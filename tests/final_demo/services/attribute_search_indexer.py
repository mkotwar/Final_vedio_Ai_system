from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


VEHICLE_CLASS_MAP = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "auto": "vehicle",
    "rickshaw": "vehicle",
    "auto_rickshaw": "vehicle",
    "vehicle": "vehicle",
}
PERSON_CLASS_MAP = {"person": "person"}
OBJECT_ALIASES = {
    "backpack": ["backpack", "bag"],
    "suitcase": ["suitcase", "bag", "luggage"],
    "handbag": ["handbag", "bag"],
    "laptop": ["laptop", "computer"],
}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def entity_family_for_class(class_name: str) -> str:
    normalized = str(class_name or "").strip().lower()
    if normalized in VEHICLE_CLASS_MAP:
        return "vehicle"
    if normalized in PERSON_CLASS_MAP:
        return "person"
    if normalized:
        return "object"
    return "unknown"


def normalize_color(color_value: Any) -> tuple[str, list[str]]:
    color = str(color_value or "").strip().lower()
    if not color or color == "unknown":
        return "", []
    mapping = {
        "white": ("white", ["white", "light"]),
        "silver": ("silver", ["silver", "gray", "grey", "light", "white_possible"]),
        "grey": ("gray", ["gray", "grey", "silver", "light", "white_possible"]),
        "gray": ("gray", ["gray", "grey", "silver", "light", "white_possible"]),
        "black": ("black", ["black", "dark"]),
        "brown": ("brown", ["brown", "dark"]),
        "red": ("red", ["red"]),
        "blue": ("blue", ["blue"]),
        "yellow": ("yellow", ["yellow"]),
        "green": ("green", ["green"]),
    }
    return mapping.get(color, (color, [color]))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in result:
            result.append(text)
    return result


def build_search_text(keywords: list[str]) -> str:
    return " ".join(dict.fromkeys([item for item in keywords if item]))


def build_match_facets(
    *,
    entity_family: str,
    entity_type: str,
    class_name: str,
    vehicle_type: str,
    normalized_color: str,
    color_family: list[str],
    plate_text_normalized: str,
    record_type: str,
    needs_review: bool,
    person_attribute_tokens: list[str],
) -> dict[str, list[str]]:
    return {
        "entity_family": clean_list([entity_family]),
        "entity_type": clean_list([entity_type]),
        "class": clean_list([class_name]),
        "vehicle_type": clean_list([vehicle_type]),
        "person_attributes": clean_list(person_attribute_tokens),
        "object_type": clean_list([entity_type if entity_family == "object" else ""]),
        "color": clean_list([normalized_color]),
        "color_family": clean_list(color_family),
        "plate": clean_list([plate_text_normalized]),
        "review_status": clean_list(["needs_review" if needs_review else "confirmed"]),
        "record_type": clean_list([record_type]),
    }


def event_attribute_block(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event.get("attributes") or {})


def event_evidence_block(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event.get("evidence") or {})


def build_event_record(event: dict[str, Any], index: int) -> dict[str, Any]:
    attrs = event_attribute_block(event)
    evidence = event_evidence_block(event)
    class_name = str(event.get("class_name") or attrs.get("final_class_name") or "unknown").lower()
    safe_class_name = str(
        attrs.get("safe_display_class_name")
        or ("vehicle" if attrs.get("vehicle_subtype_needs_review") else class_name)
        or class_name
    ).lower()
    raw_class_name = str(
        attrs.get("raw_vehicle_class_name")
        or attrs.get("source_detection_class_name")
        or attrs.get("matched_track_class_name")
        or class_name
    ).lower()
    entity_family = entity_family_for_class(safe_class_name or class_name)
    entity_type = safe_class_name or class_name
    vehicle_color = attrs.get("vehicle_color") or ""
    normalized_color, color_family = normalize_color(vehicle_color)
    plate_text = str(attrs.get("candidate_plate_text") or "").strip()
    plate_text_normalized = plate_text.lower()
    person_tokens = clean_list(
        [
            attrs.get("upper_clothing_color"),
            attrs.get("lower_clothing_color"),
            attrs.get("carrying_object"),
            attrs.get("carrying_object_type"),
        ]
    )
    keywords = clean_list(
        [
            entity_family,
            entity_type,
            class_name,
            safe_class_name,
            raw_class_name,
            attrs.get("vehicle_type"),
            attrs.get("resolved_vehicle_type"),
            attrs.get("vehicle_category"),
            vehicle_color,
            normalized_color,
            plate_text_normalized,
            attrs.get("plate_ocr_status"),
            attrs.get("plate_format_status"),
            attrs.get("candidate_source"),
            event.get("event_type"),
            "needs_review" if event.get("needs_review") else "",
        ]
        + color_family
        + person_tokens
    )
    return {
        "search_id": f"search_{index:06d}",
        "record_type": "event_record",
        "source_event_candidate_id": event.get("event_candidate_id"),
        "source_track_id": event.get("source_track_id"),
        "attribute_track_id": event.get("attribute_track_id"),
        "source_detection_id": attrs.get("source_detection_id"),
        "entity_family": entity_family,
        "entity_type": entity_type,
        "class_name": class_name,
        "safe_class_name": safe_class_name,
        "raw_class_name": raw_class_name,
        "start_time": as_float(event.get("start_time")),
        "end_time": as_float(event.get("end_time")),
        "representative_timestamp": as_float(event.get("representative_timestamp")),
        "duration_seconds": as_float(event.get("duration_seconds")),
        "confidence": as_float(event.get("confidence")),
        "risk_score": as_float(event.get("risk_score")),
        "needs_review": bool(event.get("needs_review")),
        "attributes": {
            **attrs,
            "vehicle_type": attrs.get("safe_display_vehicle_type") or attrs.get("resolved_vehicle_type") or attrs.get("vehicle_type"),
            "normalized_color": normalized_color,
            "color_family": color_family,
            "plate_text": plate_text,
            "plate_text_normalized": plate_text_normalized,
            "attribute_status": attrs.get("attribute_status") or "",
        },
        "evidence": {
            "best_frame_id": evidence.get("best_frame_id"),
            "best_image_path": evidence.get("best_image_path"),
            "crop_path": evidence.get("crop_path"),
            "plate_crop_path": evidence.get("plate_crop_path"),
            "ocr_debug_crop_dir": evidence.get("ocr_debug_crop_dir"),
            "bbox": evidence.get("bbox"),
            "supporting_frame_ids": list(evidence.get("supporting_frame_ids") or []),
            "supporting_timestamps": list(evidence.get("supporting_timestamps") or []),
        },
        "search_keywords": keywords,
        "search_text": build_search_text(keywords),
        "match_facets": build_match_facets(
            entity_family=entity_family,
            entity_type=entity_type,
            class_name=class_name,
            vehicle_type=str(attrs.get("safe_display_vehicle_type") or attrs.get("resolved_vehicle_type") or attrs.get("vehicle_type") or "").lower(),
            normalized_color=normalized_color,
            color_family=color_family,
            plate_text_normalized=plate_text_normalized,
            record_type="event_record",
            needs_review=bool(event.get("needs_review")),
            person_attribute_tokens=person_tokens,
        ),
    }


def build_track_record(track: dict[str, Any], attribute: dict[str, Any] | None, index: int) -> dict[str, Any]:
    class_name = str(track.get("class_name") or "unknown").lower()
    entity_family = entity_family_for_class(class_name)
    entity_type = class_name
    person_attributes = dict((attribute or {}).get("person_attributes") or {})
    vehicle_attributes = dict((attribute or {}).get("vehicle_attributes") or {})
    basic_attributes = dict((attribute or {}).get("basic_attributes") or {})
    vehicle_color = vehicle_attributes.get("vehicle_color") or basic_attributes.get("dominant_color") or ""
    normalized_color, color_family = normalize_color(vehicle_color)
    person_tokens = clean_list(
        [
            person_attributes.get("upper_clothing_color"),
            person_attributes.get("lower_clothing_color"),
        ]
    )
    if list(person_attributes.get("carried_object_candidates") or []):
        person_tokens.extend(clean_list(list(person_attributes.get("carried_object_candidates") or [])))
    keywords = clean_list(
        [
            entity_family,
            entity_type,
            class_name,
            vehicle_attributes.get("vehicle_type"),
            vehicle_attributes.get("vehicle_category"),
            vehicle_color,
            normalized_color,
            "needs_review" if (track.get("needs_review") or (attribute or {}).get("needs_review")) else "",
            "not_extracted" if (entity_family in {"person", "object"} and not (person_attributes or basic_attributes)) else "",
        ]
        + color_family
        + person_tokens
    )
    evidence_bbox = None
    bbox_sequence = list(track.get("bbox_sequence") or [])
    if bbox_sequence and isinstance(bbox_sequence[0], dict):
        evidence_bbox = bbox_sequence[0].get("bbox_xyxy")
    record_attributes = {
        "vehicle_type": vehicle_attributes.get("vehicle_type"),
        "vehicle_category": vehicle_attributes.get("vehicle_category"),
        "vehicle_color": vehicle_color,
        "normalized_color": normalized_color,
        "color_family": color_family,
        "object_type": entity_type if entity_family == "object" else "",
        "person_id": str((attribute or {}).get("attribute_track_id") or track.get("clean_track_id") or ""),
        "clothing_top_color": person_attributes.get("upper_clothing_color", ""),
        "clothing_bottom_color": person_attributes.get("lower_clothing_color", ""),
        "clothing_color_family": color_family if entity_family == "person" else [],
        "carrying_object": ",".join(clean_list(list(person_attributes.get("carried_object_candidates") or []))),
        "carrying_object_type": ",".join(clean_list(list(person_attributes.get("carried_object_candidates") or []))),
        "person_track_quality": track.get("cleanup_status"),
        "person_attributes_need_review": entity_family == "person" and not bool(person_attributes),
        "object_attributes_need_review": entity_family == "object" and not bool(basic_attributes),
        "attribute_status": "not_extracted" if ((entity_family == "person" and not person_attributes) or (entity_family == "object" and not basic_attributes)) else "extracted",
    }
    return {
        "search_id": f"search_{index:06d}",
        "record_type": "track_record",
        "source_event_candidate_id": None,
        "source_track_id": track.get("clean_track_id") or track.get("source_track_id"),
        "attribute_track_id": (attribute or {}).get("attribute_track_id"),
        "source_detection_id": None,
        "entity_family": entity_family,
        "entity_type": entity_type,
        "class_name": class_name,
        "safe_class_name": class_name,
        "raw_class_name": class_name,
        "start_time": as_float(track.get("start_time")),
        "end_time": as_float(track.get("end_time")),
        "representative_timestamp": round((as_float(track.get("start_time")) + as_float(track.get("end_time"))) / 2.0, 3),
        "duration_seconds": as_float(track.get("duration_seconds")),
        "confidence": as_float(track.get("average_confidence") or track.get("max_confidence")),
        "risk_score": 0.15 if track.get("needs_review") else 0.05,
        "needs_review": bool(track.get("needs_review")) or bool((attribute or {}).get("needs_review")),
        "attributes": record_attributes,
        "evidence": {
            "best_frame_id": track.get("best_frame_id"),
            "best_image_path": track.get("best_image_path"),
            "crop_path": (attribute or {}).get("attribute_crop_path"),
            "plate_crop_path": vehicle_attributes.get("possible_plate_crop_path"),
            "ocr_debug_crop_dir": None,
            "bbox": evidence_bbox,
            "supporting_frame_ids": clean_list([item.get("frame_id") for item in bbox_sequence if isinstance(item, dict)]),
            "supporting_timestamps": [as_float(item.get("timestamp")) for item in bbox_sequence if isinstance(item, dict)],
        },
        "search_keywords": keywords,
        "search_text": build_search_text(keywords),
        "match_facets": build_match_facets(
            entity_family=entity_family,
            entity_type=entity_type,
            class_name=class_name,
            vehicle_type=str(vehicle_attributes.get("vehicle_type") or "").lower(),
            normalized_color=normalized_color,
            color_family=color_family,
            plate_text_normalized="",
            record_type="track_record",
            needs_review=bool(track.get("needs_review")) or bool((attribute or {}).get("needs_review")),
            person_attribute_tokens=person_tokens,
        ),
    }


def build_detection_record(detection: dict[str, Any], frame_lookup: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
    class_name = str(detection.get("class_name") or "unknown").lower()
    entity_family = entity_family_for_class(class_name)
    entity_type = class_name
    frame_id = str(detection.get("frame_id") or "")
    frame_item = frame_lookup.get(frame_id, {})
    timestamp = as_float(
        detection.get("global_timestamp_seconds")
        or frame_item.get("global_timestamp_seconds")
        or frame_item.get("timestamp")
    )
    aliases = OBJECT_ALIASES.get(class_name, [])
    keywords = clean_list(
        [
            entity_family,
            entity_type,
            class_name,
            "needs_review",
            "not_extracted",
        ]
        + aliases
    )
    return {
        "search_id": f"search_{index:06d}",
        "record_type": "detection_record",
        "source_event_candidate_id": None,
        "source_track_id": None,
        "attribute_track_id": None,
        "source_detection_id": detection.get("detection_id"),
        "entity_family": entity_family,
        "entity_type": entity_type,
        "class_name": class_name,
        "safe_class_name": class_name,
        "raw_class_name": class_name,
        "start_time": timestamp,
        "end_time": timestamp,
        "representative_timestamp": timestamp,
        "duration_seconds": 0.0,
        "confidence": as_float(detection.get("confidence")),
        "risk_score": 0.10,
        "needs_review": True,
        "attributes": {
            "attribute_status": "not_extracted",
            "object_type": entity_type if entity_family == "object" else "",
            "vehicle_type": entity_type if entity_family == "vehicle" else "",
            "person_attributes_need_review": entity_family == "person",
            "object_attributes_need_review": entity_family == "object",
        },
        "evidence": {
            "best_frame_id": frame_id,
            "best_image_path": frame_item.get("image_path"),
            "crop_path": None,
            "plate_crop_path": None,
            "ocr_debug_crop_dir": None,
            "bbox": detection.get("bbox_xyxy"),
            "supporting_frame_ids": [frame_id] if frame_id else [],
            "supporting_timestamps": [timestamp],
        },
        "search_keywords": keywords,
        "search_text": build_search_text(keywords),
        "match_facets": build_match_facets(
            entity_family=entity_family,
            entity_type=entity_type,
            class_name=class_name,
            vehicle_type=entity_type if entity_family == "vehicle" else "",
            normalized_color="",
            color_family=[],
            plate_text_normalized="",
            record_type="detection_record",
            needs_review=True,
            person_attribute_tokens=[],
        ),
    }


def query_matches_record(query: str, record: dict[str, Any]) -> tuple[str | None, str | None]:
    query_text = query.lower()
    keywords = set(clean_list(record.get("search_keywords")))
    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "")
    attrs = dict(record.get("attributes") or {})
    notes = None

    wants_review = "review" in query_text
    wants_vehicle = "vehicle" in query_text
    wants_person = "person" in query_text
    wants_object = "object" in query_text
    wants_car = "car" in query_text
    wants_white = "white" in query_text
    wants_silver = "silver" in query_text
    wants_backpack = "backpack" in query_text
    wants_suitcase = "suitcase" in query_text
    wants_plate = "plate " in query_text or "plate" in query_text

    if wants_vehicle and entity_family != "vehicle":
        return None, None
    if wants_person and entity_family != "person":
        return None, None
    if wants_object and entity_family != "object":
        return None, None
    if wants_review and not bool(record.get("needs_review")):
        return None, None
    if wants_car and entity_type != "car":
        if entity_family == "vehicle" and bool(attrs.get("vehicle_subtype_needs_review")):
            return "review", "vehicle subtype needs review"
        return None, None
    if wants_backpack and entity_type != "backpack":
        return None, None
    if wants_suitcase and entity_type != "suitcase":
        return None, None
    if wants_plate:
        plate_token = query_text.split("plate", 1)[1].strip()
        plate_text = str(attrs.get("plate_text_normalized") or "").lower()
        if not plate_token or plate_token not in plate_text:
            return None, None
    if wants_white:
        normalized_color = str(attrs.get("normalized_color") or "").lower()
        color_family = clean_list(attrs.get("color_family"))
        if normalized_color == "white":
            return "strong", None
        if "white_possible" in color_family:
            if bool(attrs.get("vehicle_subtype_needs_review")):
                return "review", "white possible only via uncertain subtype/color evidence"
            return "possible", None
        return None, None
    if wants_silver:
        normalized_color = str(attrs.get("normalized_color") or "").lower()
        if normalized_color == "silver":
            return "strong", None
        return None, None
    if wants_person and "red shirt" in query_text:
        top_color = str(attrs.get("clothing_top_color") or "").lower()
        if top_color == "red":
            return "strong", None
        notes = "person attribute extraction not available"
        return "review" if record.get("needs_review") else "possible", notes
    if wants_person and entity_family == "person" and attrs.get("attribute_status") == "not_extracted":
        notes = "person attribute extraction not available"
        return "review", notes
    if wants_object and attrs.get("attribute_status") == "not_extracted":
        notes = "object attribute extraction not available"
        return "review", notes
    if wants_car or wants_vehicle or wants_person or wants_object or wants_backpack or wants_suitcase or wants_plate:
        return "strong", notes
    return None, None


def run_smoke_tests(records: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [
        "find white car",
        "find silver car",
        "find all cars",
        "find vehicle with plate HR38AE1442",
        "find all persons",
        "find person wearing red shirt",
        "find backpack",
        "find suitcase",
        "find all objects",
        "find vehicles needing review",
        "find objects needing review",
    ]
    results: list[dict[str, Any]] = []
    for query in queries:
        strong_matches: list[str] = []
        possible_matches: list[str] = []
        review_matches: list[str] = []
        notes: list[str] = []
        for record in records:
            match_level, note = query_matches_record(query, record)
            if match_level == "strong":
                strong_matches.append(str(record.get("search_id")))
            elif match_level == "possible":
                possible_matches.append(str(record.get("search_id")))
            elif match_level == "review":
                review_matches.append(str(record.get("search_id")))
            if note and note not in notes:
                notes.append(note)
        results.append(
            {
                "query": query,
                "total_matches": len(strong_matches) + len(possible_matches) + len(review_matches),
                "strong_matches": strong_matches,
                "possible_matches": possible_matches,
                "review_matches": review_matches,
                "notes": notes,
            }
        )
    return {"created_at": current_timestamp(), "queries": results}


def update_run_manifest_for_attribute_search_index(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "08_attribute_search_index" not in completed_steps:
        completed_steps.append("08_attribute_search_index")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "09_search_query_engine"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_attribute_search_index_outputs(run_dir: Path) -> dict[str, Any]:
    events_payload = read_optional_json(run_dir / "07B_event_candidates.json")
    tracks_payload = read_optional_json(run_dir / "05B_clean_tracks.json")
    attributes_payload = read_optional_json(run_dir / "06_track_attributes.json")
    detections_payload = read_optional_json(run_dir / "04_yolo_detections.json")
    ocr_payload = read_optional_json(run_dir / "07A_plate_ocr_results.json")
    event_report_payload = read_optional_json(run_dir / "07B_event_candidate_report.json")
    frames_payload = read_optional_json(run_dir / "03_sampled_frames_index.json")

    warnings: list[str] = []
    recommendations: list[str] = []
    frame_lookup = {}
    if isinstance(frames_payload, dict):
        for frame in list(frames_payload.get("frames") or []):
            if isinstance(frame, dict) and str(frame.get("frame_id") or ""):
                frame_lookup[str(frame.get("frame_id"))] = frame

    events = list((events_payload or {}).get("events") or [])
    tracks = list((tracks_payload or {}).get("clean_tracks") or [])
    attributes = list((attributes_payload or {}).get("attributes") or [])
    detections = list((detections_payload or {}).get("detections") or [])
    attribute_by_track = {
        str(item.get("source_track_id") or ""): item
        for item in attributes
        if isinstance(item, dict) and str(item.get("source_track_id") or "")
    }

    records: list[dict[str, Any]] = []
    covered_track_ids: set[str] = set()
    covered_detection_ids: set[str] = set()

    next_index = 1
    for event in events:
        if not isinstance(event, dict):
            continue
        record = build_event_record(event, next_index)
        next_index += 1
        records.append(record)
        if record.get("source_track_id"):
            covered_track_ids.add(str(record["source_track_id"]))
        if record.get("source_detection_id"):
            covered_detection_ids.add(str(record["source_detection_id"]))

    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("clean_track_id") or track.get("source_track_id") or "")
        if track_id and track_id in covered_track_ids:
            continue
        record = build_track_record(track, attribute_by_track.get(track_id), next_index)
        next_index += 1
        records.append(record)
        if track_id:
            covered_track_ids.add(track_id)

    for detection in detections:
        if not isinstance(detection, dict):
            continue
        detection_id = str(detection.get("detection_id") or "")
        if detection_id and detection_id in covered_detection_ids:
            continue
        record = build_detection_record(detection, frame_lookup, next_index)
        next_index += 1
        records.append(record)
        if detection_id:
            covered_detection_ids.add(detection_id)

    records_by_type: dict[str, int] = defaultdict(int)
    records_by_family: dict[str, int] = defaultdict(int)
    records_by_class: dict[str, int] = defaultdict(int)
    records_by_entity_type: dict[str, int] = defaultdict(int)
    records_by_color: dict[str, int] = defaultdict(int)
    records_by_normalized_color: dict[str, int] = defaultdict(int)
    records_with_plate_text = 0
    records_with_persons = 0
    records_with_objects = 0
    records_needing_review = 0
    records_with_vehicle_subtype_review = 0
    records_with_missing_attributes = 0

    for record in records:
        records_by_type[str(record.get("record_type") or "unknown")] += 1
        records_by_family[str(record.get("entity_family") or "unknown")] += 1
        records_by_class[str(record.get("class_name") or "unknown")] += 1
        records_by_entity_type[str(record.get("entity_type") or "unknown")] += 1
        attrs = dict(record.get("attributes") or {})
        if attrs.get("vehicle_color"):
            records_by_color[str(attrs.get("vehicle_color")).lower()] += 1
        if attrs.get("normalized_color"):
            records_by_normalized_color[str(attrs.get("normalized_color")).lower()] += 1
        if attrs.get("plate_text_normalized"):
            records_with_plate_text += 1
        if str(record.get("entity_family")) == "person":
            records_with_persons += 1
        if str(record.get("entity_family")) == "object":
            records_with_objects += 1
        if bool(record.get("needs_review")):
            records_needing_review += 1
        if bool(attrs.get("vehicle_subtype_needs_review")):
            records_with_vehicle_subtype_review += 1
        if str(attrs.get("attribute_status") or "") == "not_extracted":
            records_with_missing_attributes += 1

    if records_with_persons == 0:
        warnings.append("No person records were indexed.")
        recommendations.append("Enable person tracking focus if person search is required.")
    if records_with_objects == 0:
        warnings.append("No object records were indexed.")
        recommendations.append("Keep detection-level fallback records for broad search.")
    if any(str(item.get("class_name") or "").lower() == "person" for item in detections) and records_with_persons == 0:
        warnings.append("Person detections exist in Step 4 but no person tracks/events were indexed.")
    if any(entity_family_for_class(str(item.get("class_name") or "")) == "object" for item in detections) and records_with_objects == 0:
        warnings.append("Object detections exist in Step 4 but no object tracks/events were indexed.")
    if not records_by_normalized_color:
        warnings.append("Color attributes are missing.")
    if not any(
        str(dict(record.get("attributes") or {}).get("clothing_top_color") or "")
        or str(dict(record.get("attributes") or {}).get("clothing_bottom_color") or "")
        for record in records
    ):
        warnings.append("Person clothing attributes are missing.")
        recommendations.append("Add person attribute extraction for clothing/action/carrying object.")
    if not any(str(record.get("entity_family") or "") == "object" and str(dict(record.get("attributes") or {}).get("attribute_status") or "") != "not_extracted" for record in records):
        recommendations.append("Add generic object attribute extraction for bags/laptops/suitcases.")

    smoke_payload = run_smoke_tests(records)
    smoke_tests_passed = any(
        item.get("query") == "find vehicle with plate HR38AE1442" and item.get("total_matches", 0) > 0
        for item in list(smoke_payload.get("queries") or [])
    ) if records_with_plate_text else True

    index_payload = {
        "created_at": current_timestamp(),
        "source": {
            "events": "07B_event_candidates.json" if events_payload is not None else None,
            "tracks": "05B_clean_tracks.json" if tracks_payload is not None else None,
            "attributes": "06_track_attributes.json" if attributes_payload is not None else None,
            "detections": "04_yolo_detections.json" if detections_payload is not None else None,
            "ocr": "07A_plate_ocr_results.json" if ocr_payload is not None else None,
        },
        "records": records,
    }
    report_payload = {
        "created_at": current_timestamp(),
        "overall_status": "completed",
        "total_events_input": len(events),
        "total_tracks_input": len(tracks),
        "total_detections_input": len(detections),
        "total_search_records": len(records),
        "records_by_type": dict(sorted(records_by_type.items())),
        "records_by_family": dict(sorted(records_by_family.items())),
        "records_by_class": dict(sorted(records_by_class.items())),
        "records_by_entity_type": dict(sorted(records_by_entity_type.items())),
        "records_by_color": dict(sorted(records_by_color.items())),
        "records_by_normalized_color": dict(sorted(records_by_normalized_color.items())),
        "records_with_plate_text": records_with_plate_text,
        "records_with_persons": records_with_persons,
        "records_with_objects": records_with_objects,
        "records_needing_review": records_needing_review,
        "records_with_vehicle_subtype_review": records_with_vehicle_subtype_review,
        "records_with_missing_attributes": records_with_missing_attributes,
        "smoke_tests_passed": smoke_tests_passed,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recommendations)),
    }
    return {
        "index_payload": index_payload,
        "report_payload": report_payload,
        "smoke_payload": smoke_payload,
    }
