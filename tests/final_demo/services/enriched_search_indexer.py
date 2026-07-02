from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


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


def is_detection_id(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("det_")


def first_non_detection_track_id(values: Any) -> str:
    for value in list(values or []):
        text = str(value or "").strip()
        if text and not is_detection_id(text):
            return text
    return ""


def first_detection_id(values: Any) -> str:
    for value in list(values or []):
        text = str(value or "").strip()
        if text and is_detection_id(text):
            return text
    return ""


def merge_unique_list(*values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, list):
            items = value
        elif value is None:
            items = []
        else:
            items = [value]
        for item in items:
            if item not in result:
                result.append(item)
    return result


def normalize_color_family(values: Any) -> list[str]:
    return clean_list(values)


def base_status(record: dict[str, Any]) -> str:
    if bool(record.get("needs_review")):
        return "review"
    confidence = as_float(record.get("confidence"), 0.0)
    if confidence >= 0.70:
        return "confirmed"
    if confidence >= 0.45:
        return "possible"
    return "review"


def strength_from_confidence(confidence: float, needs_review: bool) -> str:
    if needs_review or confidence < 0.55:
        return "review"
    if confidence < 0.70:
        return "possible"
    return "strong"


def build_source_ids(
    *,
    source_track_id: Any = "",
    attribute_track_id: Any = "",
    source_detection_id: Any = "",
    source_event_id: Any = "",
    person_attribute_id: Any = "",
    object_attribute_id: Any = "",
    association_id: Any = "",
    base_search_id: Any = "",
) -> dict[str, str]:
    return {
        "source_track_id": str(source_track_id or ""),
        "attribute_track_id": str(attribute_track_id or ""),
        "source_detection_id": str(source_detection_id or ""),
        "source_event_id": str(source_event_id or ""),
        "person_attribute_id": str(person_attribute_id or ""),
        "object_attribute_id": str(object_attribute_id or ""),
        "association_id": str(association_id or ""),
        "base_search_id": str(base_search_id or ""),
    }


def build_evidence(
    *,
    frame_id: Any = "",
    image_path: Any = "",
    crop_path: Any = "",
    subject_crop_path: Any = "",
    object_crop_path: Any = "",
    supporting_frame_ids: Any = None,
    supporting_timestamps: Any = None,
    source_detection_ids: Any = None,
    source_track_ids: Any = None,
) -> dict[str, Any]:
    return {
        "frame_id": str(frame_id or ""),
        "image_path": str(image_path or ""),
        "crop_path": str(crop_path or ""),
        "subject_crop_path": str(subject_crop_path or ""),
        "object_crop_path": str(object_crop_path or ""),
        "supporting_frame_ids": clean_list(supporting_frame_ids),
        "supporting_timestamps": list(supporting_timestamps or []),
        "source_detection_ids": clean_list(source_detection_ids),
        "source_track_ids": clean_list(source_track_ids),
    }


def relationship_summary_from_association(association: dict[str, Any]) -> dict[str, Any]:
    return {
        "association_id": str(association.get("association_id") or ""),
        "relationship": str(association.get("relationship") or ""),
        "object_entity_type": str(association.get("object_entity_type") or ""),
        "subject_entity_type": str(association.get("subject_entity_type") or ""),
        "confidence": round(as_float(association.get("confidence"), 0.0), 3),
        "needs_review": bool(association.get("needs_review")),
    }


def build_base_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    attributes = dict(record.get("attributes") or {})
    evidence = dict(record.get("evidence") or {})
    return {
        "search_id": f"enriched_search_{index:06d}",
        "base_search_id": str(record.get("search_id") or ""),
        "record_type": "base_record",
        "entity_family": str(record.get("entity_family") or ""),
        "entity_type": str(record.get("entity_type") or ""),
        "class_name": str(record.get("class_name") or ""),
        "safe_class_name": str(record.get("safe_class_name") or ""),
        "raw_class_name": str(record.get("raw_class_name") or ""),
        "start_time": round(as_float(record.get("start_time"), 0.0), 3),
        "end_time": round(as_float(record.get("end_time"), 0.0), 3),
        "representative_timestamp": round(as_float(record.get("representative_timestamp"), 0.0), 3),
        "duration_seconds": round(as_float(record.get("duration_seconds"), 0.0), 3),
        "confidence": round(as_float(record.get("confidence"), 0.0), 3),
        "match_strength_default": "review" if bool(record.get("needs_review")) else "strong",
        "needs_review": bool(record.get("needs_review")),
        "review_reason": str(attributes.get("attribute_status") or ""),
        "status": base_status(record),
        "source_ids": build_source_ids(
            source_track_id=record.get("source_track_id"),
            attribute_track_id=record.get("attribute_track_id"),
            source_detection_id=record.get("source_detection_id"),
            source_event_id=record.get("source_event_candidate_id"),
            base_search_id=record.get("search_id"),
        ),
        "evidence": build_evidence(
            frame_id=evidence.get("best_frame_id"),
            image_path=evidence.get("best_image_path"),
            crop_path=evidence.get("crop_path"),
            supporting_frame_ids=evidence.get("supporting_frame_ids"),
            supporting_timestamps=evidence.get("supporting_timestamps"),
            source_detection_ids=[record.get("source_detection_id")] if record.get("source_detection_id") else [],
            source_track_ids=[record.get("source_track_id")] if record.get("source_track_id") else [],
        ),
        "attributes": attributes,
        "relationships": [],
        "search_keywords": clean_list(record.get("search_keywords")),
        "match_facets": dict(record.get("match_facets") or {}),
    }


def build_person_attribute_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    confidence = round(as_float(record.get("person_attribute_confidence"), 0.0), 3)
    needs_review = bool(record.get("needs_review"))
    top_color = str(record.get("normalized_top_color") or "")
    bottom_color = str(record.get("normalized_bottom_color") or "")
    overall_color = str(record.get("overall_clothing_color") or "")
    keywords = clean_list(
        [
            "person",
            top_color,
            bottom_color,
            overall_color,
            f"wearing_{top_color}" if top_color else "",
            f"top_{top_color}" if top_color else "",
            f"bottom_{bottom_color}" if bottom_color else "",
            f"clothing_{overall_color}" if overall_color and overall_color != "unknown" else "",
            "carrying_object" if bool(record.get("carrying_object_possible")) else "",
            f"carrying_{record.get('carrying_object_type')}" if bool(record.get("carrying_object_possible")) and str(record.get("carrying_object_type") or "") else "",
            "needs_review" if needs_review else "",
        ]
        + clean_list(record.get("clothing_color_family"))
    )
    return {
        "search_id": f"enriched_search_{index:06d}",
        "base_search_id": "",
        "record_type": "person_attribute_record",
        "entity_family": "person",
        "entity_type": "person",
        "class_name": "person",
        "safe_class_name": "person",
        "raw_class_name": "person",
        "start_time": round(as_float(record.get("start_time"), 0.0), 3),
        "end_time": round(as_float(record.get("end_time"), 0.0), 3),
        "representative_timestamp": round(as_float(record.get("representative_timestamp"), 0.0), 3),
        "duration_seconds": round(as_float(record.get("duration_seconds"), 0.0), 3),
        "confidence": confidence,
        "match_strength_default": strength_from_confidence(confidence, needs_review),
        "needs_review": needs_review,
        "review_reason": str(record.get("review_reason") or ""),
        "status": "review" if needs_review else ("possible" if confidence < 0.70 else "confirmed"),
        "source_ids": build_source_ids(
            source_track_id=record.get("source_track_id"),
            attribute_track_id=record.get("attribute_track_id"),
            source_detection_id=record.get("source_detection_id"),
            person_attribute_id=record.get("person_attribute_id"),
        ),
        "evidence": build_evidence(
            frame_id=record.get("frame_id") or record.get("best_frame_id"),
            image_path=record.get("best_image_path"),
            crop_path=record.get("crop_path"),
            source_detection_ids=[record.get("source_detection_id")] if record.get("source_detection_id") else [],
            source_track_ids=[record.get("source_track_id"), record.get("attribute_track_id")],
        ),
        "attributes": {
            "top_clothing_color": record.get("top_clothing_color"),
            "bottom_clothing_color": record.get("bottom_clothing_color"),
            "overall_clothing_color": record.get("overall_clothing_color"),
            "normalized_top_color": record.get("normalized_top_color"),
            "normalized_bottom_color": record.get("normalized_bottom_color"),
            "clothing_color_family": clean_list(record.get("clothing_color_family")),
            "carrying_object_possible": bool(record.get("carrying_object_possible")),
            "carrying_object_type": record.get("carrying_object_type"),
            "carrying_object_confidence": round(as_float(record.get("carrying_object_confidence"), 0.0), 3),
            "person_attribute_confidence": confidence,
            "attribute_status": record.get("attribute_status"),
        },
        "relationships": [],
        "search_keywords": keywords,
        "match_facets": {
            "entity_family": ["person"],
            "entity_type": ["person"],
            "top_clothing_color": clean_list([record.get("normalized_top_color")]),
            "bottom_clothing_color": clean_list([record.get("normalized_bottom_color")]),
            "clothing_color_family": clean_list(record.get("clothing_color_family")),
            "review_status": ["needs_review" if needs_review else "confirmed"],
        },
    }


def build_object_attribute_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    confidence = round(as_float(record.get("object_attribute_confidence"), 0.0), 3)
    needs_review = bool(record.get("needs_review"))
    attribute_status = str(record.get("attribute_status") or "")
    object_class_needs_review = bool(record.get("object_class_needs_review"))
    entity_type = str(record.get("normalized_object_type") or record.get("object_type") or "")
    if attribute_status in {"possible_vehicle_misclassification", "possible_false_positive"} or object_class_needs_review:
        default_strength = "review"
        status = attribute_status or "review"
    else:
        default_strength = strength_from_confidence(confidence, needs_review)
        status = "review" if needs_review else ("possible" if confidence < 0.70 else "confirmed")
    object_type = str(record.get("object_type") or record.get("class_name") or "")
    normalized_color = str(record.get("normalized_color") or "")
    keywords = clean_list(
        [
            "object",
            object_type,
            entity_type,
            normalized_color,
            f"{normalized_color}_{object_type}" if normalized_color and object_type else "",
            "needs_review" if needs_review else "",
            "possible_vehicle_misclassification" if attribute_status == "possible_vehicle_misclassification" else "",
            "vehicle_like_object" if attribute_status == "possible_vehicle_misclassification" or object_class_needs_review else "",
            "uncertain_object" if entity_type == "uncertain_object" else "",
        ]
    )
    return {
        "search_id": f"enriched_search_{index:06d}",
        "base_search_id": "",
        "record_type": "object_attribute_record",
        "entity_family": "object",
        "entity_type": entity_type,
        "class_name": str(record.get("class_name") or ""),
        "safe_class_name": entity_type,
        "raw_class_name": str(record.get("class_name") or ""),
        "start_time": round(as_float(record.get("start_time"), 0.0), 3),
        "end_time": round(as_float(record.get("end_time"), 0.0), 3),
        "representative_timestamp": round(as_float(record.get("representative_timestamp"), 0.0), 3),
        "duration_seconds": round(as_float(record.get("duration_seconds"), 0.0), 3),
        "confidence": confidence,
        "match_strength_default": default_strength,
        "needs_review": needs_review or object_class_needs_review,
        "review_reason": str(record.get("review_reason") or ""),
        "status": status,
        "source_ids": build_source_ids(
            source_track_id=record.get("source_track_id"),
            attribute_track_id=record.get("attribute_track_id"),
            source_detection_id=record.get("source_detection_id"),
            object_attribute_id=record.get("object_attribute_id"),
        ),
        "evidence": build_evidence(
            frame_id=record.get("best_frame_id"),
            image_path=record.get("best_image_path"),
            crop_path=record.get("crop_path"),
            source_detection_ids=[record.get("source_detection_id")] if record.get("source_detection_id") else [],
            source_track_ids=[record.get("source_track_id"), record.get("attribute_track_id")],
        ),
        "attributes": {
            "object_type": object_type,
            "normalized_object_type": entity_type,
            "object_color": record.get("object_color"),
            "normalized_color": normalized_color,
            "color_family": clean_list(record.get("color_family")),
            "object_attribute_confidence": confidence,
            "attribute_status": attribute_status,
            "possible_actual_family": record.get("possible_actual_family"),
            "possible_actual_types": clean_list(record.get("possible_actual_types")),
            "false_positive_risk_score": round(as_float(record.get("false_positive_risk_score"), 0.0), 3),
            "object_class_needs_review": object_class_needs_review,
        },
        "relationships": [],
        "search_keywords": keywords,
        "match_facets": {
            "entity_family": ["object"],
            "entity_type": clean_list([entity_type]),
            "object_type": clean_list([object_type]),
            "normalized_color": clean_list([normalized_color]),
            "review_status": ["needs_review" if (needs_review or object_class_needs_review) else "confirmed"],
        },
    }


def build_association_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    confidence = round(as_float(record.get("confidence"), 0.0), 3)
    needs_review = bool(record.get("needs_review"))
    relationship = str(record.get("relationship") or "")
    association_type = str(record.get("association_type") or "")
    subject_type = str(record.get("subject_entity_type") or "")
    object_type = str(record.get("object_entity_type") or "")
    keywords = clean_list(
        [
            association_type,
            relationship,
            subject_type,
            object_type,
            f"{subject_type}_{relationship}" if subject_type and relationship else "",
            f"{subject_type}_{object_type}" if subject_type and object_type else "",
            "needs_review" if needs_review else "",
            "possible_vehicle_misclassification" if relationship == "possible_vehicle_misclassification" else "",
            "vehicle_like_object" if relationship == "possible_vehicle_misclassification" else "",
            "false_suitcase_possible" if relationship == "possible_vehicle_misclassification" else "",
            "object_overlaps_vehicle" if relationship == "possible_vehicle_misclassification" else "",
        ]
    )
    evidence = dict(record.get("evidence") or {})
    evidence_source_track_ids = clean_list(evidence.get("source_track_ids"))
    evidence_source_detection_ids = clean_list(evidence.get("source_detection_ids"))
    subject_id = str(record.get("subject_id") or "")
    object_id = str(record.get("object_id") or "")
    corrected_source_track_id = first_non_detection_track_id(evidence_source_track_ids)
    primary_detection_id = (
        first_detection_id(evidence_source_detection_ids)
        or (str(record.get("object_source_id") or "") if is_detection_id(record.get("object_source_id")) else "")
        or (str(record.get("subject_source_id") or "") if is_detection_id(record.get("subject_source_id")) else "")
    )
    person_attribute_id = subject_id if subject_id.startswith("person_attr_") else ""
    object_attribute_id = subject_id if subject_id.startswith("object_attr_") else ""
    return {
        "search_id": f"enriched_search_{index:06d}",
        "base_search_id": "",
        "record_type": "association_record",
        "entity_family": "association",
        "entity_type": association_type,
        "class_name": relationship,
        "safe_class_name": relationship,
        "raw_class_name": relationship,
        "start_time": round(as_float(record.get("start_time"), 0.0), 3),
        "end_time": round(as_float(record.get("end_time"), 0.0), 3),
        "representative_timestamp": round(as_float(record.get("representative_timestamp"), 0.0), 3),
        "duration_seconds": round(as_float(record.get("duration_seconds"), 0.0), 3),
        "confidence": confidence,
        "match_strength_default": "review" if needs_review else ("possible" if confidence < 0.70 else "strong"),
        "needs_review": needs_review,
        "review_reason": str(record.get("review_reason") or ""),
        "status": "review" if needs_review else ("possible" if confidence < 0.70 else "confirmed"),
        "source_ids": build_source_ids(
            source_track_id=corrected_source_track_id,
            attribute_track_id="",
            source_detection_id=primary_detection_id,
            person_attribute_id=person_attribute_id,
            object_attribute_id=object_attribute_id,
            association_id=record.get("association_id"),
        ),
        "evidence": build_evidence(
            frame_id=evidence.get("frame_id"),
            image_path=evidence.get("image_path"),
            subject_crop_path=evidence.get("subject_crop_path"),
            object_crop_path=evidence.get("object_crop_path"),
            supporting_frame_ids=evidence.get("supporting_frame_ids"),
            supporting_timestamps=evidence.get("supporting_timestamps"),
            source_detection_ids=evidence_source_detection_ids,
            source_track_ids=evidence_source_track_ids,
        ),
        "attributes": {
            "association_type": association_type,
            "relationship": relationship,
            "subject_entity_family": record.get("subject_entity_family"),
            "subject_entity_type": subject_type,
            "subject_id": subject_id,
            "object_entity_family": record.get("object_entity_family"),
            "object_entity_type": object_type,
            "object_id": object_id,
            "association_status": record.get("association_status"),
            "geometry": dict(record.get("geometry") or {}),
            "alternate_vehicle_evidence": list(record.get("alternate_vehicle_evidence") or []),
            "alternate_vehicle_types": clean_list(record.get("alternate_vehicle_types")),
            "alternate_vehicle_count": int(record.get("alternate_vehicle_count") or 0),
        },
        "relationships": [],
        "search_keywords": keywords,
        "match_facets": {
            "entity_family": ["association"],
            "association_type": clean_list([association_type]),
            "relationship": clean_list([relationship]),
            "subject_entity_type": clean_list([subject_type]),
            "object_entity_type": clean_list([object_type]),
            "review_status": ["needs_review" if needs_review else "confirmed"],
        },
    }


def attach_relationship_summaries(
    records: list[dict[str, Any]],
    associations: list[dict[str, Any]],
) -> None:
    person_lookup = {
        str(record.get("source_ids", {}).get("person_attribute_id") or ""): record
        for record in records
        if str(record.get("record_type") or "") == "person_attribute_record"
    }
    object_lookup = {
        str(record.get("source_ids", {}).get("object_attribute_id") or ""): record
        for record in records
        if str(record.get("record_type") or "") == "object_attribute_record"
    }
    for association in associations:
        attrs = dict(association.get("attributes") or {})
        subject_id = str(attrs.get("subject_id") or "")
        object_id = str(attrs.get("object_id") or "")
        summary = relationship_summary_from_association(association)
        if subject_id and subject_id in person_lookup:
            person_lookup[subject_id]["relationships"] = merge_unique_list(person_lookup[subject_id].get("relationships"), [summary])
        if subject_id and subject_id in object_lookup:
            object_lookup[subject_id]["relationships"] = merge_unique_list(object_lookup[subject_id].get("relationships"), [summary])
        if object_id and object_id in object_lookup:
            object_lookup[object_id]["relationships"] = merge_unique_list(object_lookup[object_id].get("relationships"), [summary])
        if object_id and object_id in person_lookup:
            person_lookup[object_id]["relationships"] = merge_unique_list(person_lookup[object_id].get("relationships"), [summary])


def dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates_collapsed = 0
    for record in records:
        source_ids = dict(record.get("source_ids") or {})
        key = (
            str(record.get("record_type") or ""),
            str(record.get("entity_family") or ""),
            str(record.get("entity_type") or ""),
            str(source_ids.get("source_detection_id") or ""),
            str(source_ids.get("source_track_id") or ""),
            str(source_ids.get("person_attribute_id") or ""),
            str(source_ids.get("object_attribute_id") or ""),
            str(source_ids.get("association_id") or ""),
            round(as_float(record.get("representative_timestamp"), 0.0), 2),
        )
        if key not in grouped:
            grouped[key] = record
            continue
        duplicates_collapsed += 1
        current = grouped[key]
        if as_float(record.get("confidence"), 0.0) > as_float(current.get("confidence"), 0.0):
            winner = record
            loser = current
        else:
            winner = current
            loser = record
        winner["search_keywords"] = merge_unique_list(winner.get("search_keywords"), loser.get("search_keywords"))
        winner["relationships"] = merge_unique_list(winner.get("relationships"), loser.get("relationships"))
        winner_evidence = dict(winner.get("evidence") or {})
        loser_evidence = dict(loser.get("evidence") or {})
        winner_evidence["supporting_frame_ids"] = clean_list(
            list(winner_evidence.get("supporting_frame_ids") or []) + list(loser_evidence.get("supporting_frame_ids") or [])
        )
        winner_evidence["supporting_timestamps"] = merge_unique_list(
            winner_evidence.get("supporting_timestamps"),
            loser_evidence.get("supporting_timestamps"),
        )
        winner_evidence["source_detection_ids"] = clean_list(
            list(winner_evidence.get("source_detection_ids") or []) + list(loser_evidence.get("source_detection_ids") or [])
        )
        winner_evidence["source_track_ids"] = clean_list(
            list(winner_evidence.get("source_track_ids") or []) + list(loser_evidence.get("source_track_ids") or [])
        )
        winner["evidence"] = winner_evidence
        grouped[key] = winner
    deduped = list(grouped.values())
    deduped.sort(
        key=lambda item: (
            str(item.get("record_type") or ""),
            str(item.get("entity_family") or ""),
            as_float(item.get("representative_timestamp"), 0.0),
            -as_float(item.get("confidence"), 0.0),
        )
    )
    return deduped, duplicates_collapsed


def query_matches_record(query: str, record: dict[str, Any]) -> tuple[str | None, str | None]:
    query_text = query.lower()
    keywords = set(clean_list(record.get("search_keywords")))
    attributes = dict(record.get("attributes") or {})
    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "")
    match_strength = str(record.get("match_strength_default") or "review")
    note = None

    if "plate hr38ae1442" in query_text:
        plate_text = str(attributes.get("plate_text") or attributes.get("candidate_plate_text") or "").lower()
        if "hr38ae1442" not in plate_text and "hr38ae1442" not in " ".join(keywords):
            return None, None
    if "all persons" in query_text and entity_family != "person":
        return None, None
    if "gray person" in query_text or "person wearing gray" in query_text:
        if entity_family != "person":
            return None, None
        top_color = str(attributes.get("top_clothing_color") or attributes.get("normalized_top_color") or "").lower()
        bottom_color = str(attributes.get("bottom_clothing_color") or attributes.get("normalized_bottom_color") or "").lower()
        if "gray" not in {top_color, bottom_color} and "gray" not in keywords and "grey" not in keywords:
            return None, None
    if "black person" in query_text:
        if entity_family != "person":
            return None, None
        top_color = str(attributes.get("top_clothing_color") or attributes.get("normalized_top_color") or "").lower()
        bottom_color = str(attributes.get("bottom_clothing_color") or attributes.get("normalized_bottom_color") or "").lower()
        if "black" not in {top_color, bottom_color} and "black" not in keywords:
            return None, None
    if "confirmed suitcase" in query_text:
        if "suitcase" not in keywords:
            return None, None
        if match_strength != "strong":
            return None, "suitcase exists only as review/possible evidence"
    if "find suitcase" in query_text and "confirmed suitcase" not in query_text:
        if "suitcase" not in keywords:
            return None, None
    if "object needing review" in query_text and not bool(record.get("needs_review")):
        return None, None
    if "vehicle-like object" in query_text:
        if "vehicle_like_object" not in keywords and str(attributes.get("possible_actual_family") or "") != "vehicle":
            return None, None
    if "possible vehicle misclassification" in query_text:
        if str(attributes.get("relationship") or "") != "possible_vehicle_misclassification" and str(attributes.get("attribute_status") or "") != "possible_vehicle_misclassification":
            return None, None
    if "object overlapping vehicle" in query_text:
        if "object_overlaps_vehicle" not in keywords and str(attributes.get("relationship") or "") not in {"possible_vehicle_misclassification", "overlapping"}:
            return None, None
    if "uncertain object" in query_text and entity_type != "uncertain_object":
        return None, None
    if "all vehicles" in query_text and entity_family != "vehicle":
        return None, None

    if "needs_review" in keywords or bool(record.get("needs_review")):
        if match_strength == "strong":
            return "review", note
        return match_strength or "review", note
    return match_strength or "review", note


def run_smoke_tests(records: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [
        "find all persons",
        "find gray person",
        "find person wearing gray",
        "find black person",
        "find suitcase",
        "find confirmed suitcase",
        "find object needing review",
        "find vehicle-like object",
        "find possible vehicle misclassification",
        "find object overlapping vehicle",
        "find uncertain object",
        "find all vehicles",
        "find plate HR38AE1442 if present",
    ]
    results: list[dict[str, Any]] = []
    for query in queries:
        strong_matches = 0
        possible_matches = 0
        review_matches = 0
        notes: list[str] = []
        for record in records:
            match_level, note = query_matches_record(query, record)
            if match_level == "strong":
                strong_matches += 1
            elif match_level == "possible":
                possible_matches += 1
            elif match_level == "review":
                review_matches += 1
            if note and note not in notes:
                notes.append(note)
        results.append(
            {
                "query": query,
                "total_matches": strong_matches + possible_matches + review_matches,
                "strong_matches": strong_matches,
                "possible_matches": possible_matches,
                "review_matches": review_matches,
                "notes": notes,
            }
        )
    return {"created_at": current_timestamp(), "queries": results}


def build_enriched_search_index_outputs(run_dir: Path) -> dict[str, Any]:
    base_index_path = run_dir / "08_attribute_search_index.json"
    if not base_index_path.exists():
        raise FileNotFoundError(f"Missing required Step 13 input: {base_index_path}")

    person_path = run_dir / "10_person_attributes.json"
    object_path = run_dir / "11_object_attributes.json"
    association_path = run_dir / "12_entity_associations.json"
    event_path = run_dir / "07B_event_candidates.json"
    search_results_path = run_dir / "09_search_results.json"

    base_payload = read_json(base_index_path)
    person_payload = read_optional_json(person_path)
    object_payload = read_optional_json(object_path)
    association_payload = read_optional_json(association_path)
    event_payload = read_optional_json(event_path)
    _search_results_payload = read_optional_json(search_results_path)

    warnings: list[str] = []
    recommendations: list[str] = []
    missing_optional_inputs: list[str] = []
    if person_payload is None:
        missing_optional_inputs.append(person_path.name)
        warnings.append("10_person_attributes.json is missing.")
    if object_payload is None:
        missing_optional_inputs.append(object_path.name)
        warnings.append("11_object_attributes.json is missing.")
    if association_payload is None:
        missing_optional_inputs.append(association_path.name)
        warnings.append("12_entity_associations.json is missing.")

    base_records = list(base_payload.get("records") or [])
    person_records = list((person_payload or {}).get("person_attributes") or [])
    object_records = list((object_payload or {}).get("object_attributes") or [])
    association_records = list((association_payload or {}).get("associations") or [])
    records_with_detection_id_in_track_id = 0
    for record in association_records:
        if not isinstance(record, dict):
            continue
        evidence = dict(record.get("evidence") or {})
        source_track_ids = clean_list(evidence.get("source_track_ids"))
        if any(is_detection_id(item) for item in source_track_ids) or is_detection_id(record.get("subject_source_id")):
            records_with_detection_id_in_track_id += 1

    enriched_records: list[dict[str, Any]] = []
    next_index = 1
    for record in base_records:
        if isinstance(record, dict):
            enriched_records.append(build_base_record(record, next_index))
            next_index += 1
    for record in person_records:
        if isinstance(record, dict):
            enriched_records.append(build_person_attribute_record(record, next_index))
            next_index += 1
    for record in object_records:
        if isinstance(record, dict):
            enriched_records.append(build_object_attribute_record(record, next_index))
            next_index += 1
    association_enriched_records: list[dict[str, Any]] = []
    for record in association_records:
        if isinstance(record, dict):
            association_enriched_records.append(build_association_record(record, next_index))
            next_index += 1
    enriched_records.extend(association_enriched_records)

    attach_relationship_summaries(enriched_records, association_enriched_records)
    enriched_records, duplicates_collapsed = dedupe_records(enriched_records)

    if any(
        str(record.get("status") or "") == "possible_vehicle_misclassification"
        for record in enriched_records
    ):
        warnings.append("Some object records are possible vehicle misclassifications and remain review evidence.")
    if not person_records:
        warnings.append("No enriched person attributes exist.")
    if not association_records:
        warnings.append("No association records exist.")
    if records_with_detection_id_in_track_id > 0:
        warnings.append("Some records had detection IDs incorrectly placed in source_track_id and were corrected.")

    if person_records:
        recommendations.append("Person clothing attributes exist. Use the enriched query engine next.")
    if any(
        str(record.get("status") or "") == "possible_vehicle_misclassification"
        for record in enriched_records
    ):
        recommendations.append("Object false positives exist. Consider a custom traffic/e-rickshaw model.")
    if sum(1 for record in enriched_records if bool(record.get("needs_review"))) >= max(5, len(enriched_records) // 3):
        recommendations.append("Many records need review. Consider VLM verification for selected review records later.")

    smoke_payload = run_smoke_tests(enriched_records)
    created_at = current_timestamp()
    output_payload = {
        "created_at": created_at,
        "source": {
            "base_index": base_index_path.name,
            "person_attributes": person_path.name,
            "object_attributes": object_path.name,
            "entity_associations": association_path.name,
        },
        "index_version": "final_demo_enriched_v1",
        "records": enriched_records,
    }
    report_payload = {
        "overall_status": "completed",
        "base_records_loaded": len(base_records),
        "person_attribute_records_loaded": len(person_records),
        "object_attribute_records_loaded": len(object_records),
        "association_records_loaded": len(association_records),
        "enriched_records_created": len(enriched_records),
        "records_by_record_type": dict(sorted(Counter(str(item.get("record_type") or "") for item in enriched_records).items())),
        "records_by_entity_family": dict(sorted(Counter(str(item.get("entity_family") or "") for item in enriched_records).items())),
        "records_by_entity_type": dict(sorted(Counter(str(item.get("entity_type") or "") for item in enriched_records).items())),
        "records_by_status": dict(sorted(Counter(str(item.get("status") or "") for item in enriched_records).items())),
        "records_by_match_strength_default": dict(sorted(Counter(str(item.get("match_strength_default") or "") for item in enriched_records).items())),
        "records_needing_review": sum(1 for item in enriched_records if bool(item.get("needs_review"))),
        "person_records_with_clothing": sum(
            1
            for item in enriched_records
            if str(item.get("record_type") or "") == "person_attribute_record"
            and (
                str(dict(item.get("attributes") or {}).get("top_clothing_color") or "") not in {"", "unknown"}
                or str(dict(item.get("attributes") or {}).get("bottom_clothing_color") or "") not in {"", "unknown"}
            )
        ),
        "object_records_with_color": sum(
            1
            for item in enriched_records
            if str(item.get("record_type") or "") == "object_attribute_record"
            and str(dict(item.get("attributes") or {}).get("object_color") or "") not in {"", "unknown"}
        ),
        "object_records_possible_vehicle_misclassification": sum(
            1
            for item in enriched_records
            if str(item.get("record_type") or "") == "object_attribute_record"
            and str(item.get("status") or "") == "possible_vehicle_misclassification"
        ),
        "association_records_possible_vehicle_misclassification": sum(
            1
            for item in enriched_records
            if str(item.get("record_type") or "") == "association_record"
            and str(dict(item.get("attributes") or {}).get("relationship") or "") == "possible_vehicle_misclassification"
        ),
        "records_with_detection_id_in_track_id": records_with_detection_id_in_track_id,
        "records_with_relationships": sum(1 for item in enriched_records if list(item.get("relationships") or [])),
        "duplicates_collapsed": duplicates_collapsed,
        "missing_optional_inputs": missing_optional_inputs,
        "warnings": warnings,
        "recommendations": recommendations,
        "created_at": created_at,
    }
    return {
        "index_payload": output_payload,
        "report_payload": report_payload,
        "smoke_payload": smoke_payload,
    }


def update_run_manifest_for_enriched_search_index(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "13_enriched_search_index" not in completed_steps:
        completed_steps.append("13_enriched_search_index")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "14_enriched_query_engine"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
