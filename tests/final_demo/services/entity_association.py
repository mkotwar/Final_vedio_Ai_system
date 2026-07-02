from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_ASSOC_TIME_TOLERANCE_SECONDS = "FINAL_DEMO_ASSOC_TIME_TOLERANCE_SECONDS"
ENV_FINAL_DEMO_ASSOC_PERSON_OBJECT_MAX_DISTANCE = "FINAL_DEMO_ASSOC_PERSON_OBJECT_MAX_DISTANCE"
ENV_FINAL_DEMO_ASSOC_PERSON_VEHICLE_MAX_DISTANCE = "FINAL_DEMO_ASSOC_PERSON_VEHICLE_MAX_DISTANCE"
ENV_FINAL_DEMO_ASSOC_MIN_CONFIDENCE = "FINAL_DEMO_ASSOC_MIN_CONFIDENCE"
ENV_FINAL_DEMO_ASSOC_MAX_RECORDS = "FINAL_DEMO_ASSOC_MAX_RECORDS"
ENV_FINAL_DEMO_ASSOC_DEBUG_FULL = "FINAL_DEMO_ASSOC_DEBUG_FULL"

DEFAULT_ASSOC_TIME_TOLERANCE_SECONDS = 0.75
DEFAULT_ASSOC_PERSON_OBJECT_MAX_DISTANCE = 1.25
DEFAULT_ASSOC_PERSON_VEHICLE_MAX_DISTANCE = 1.50
DEFAULT_ASSOC_MIN_CONFIDENCE = 0.30
DEFAULT_ASSOC_MAX_RECORDS = 200

CARRYING_OBJECT_TYPES = {"bag", "backpack", "handbag", "suitcase", "laptop", "phone", "bottle"}
VEHICLE_TYPES = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "auto", "rickshaw", "auto_rickshaw"}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def compute_bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def compute_bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def compute_bbox_diagonal(bbox_xyxy: list[float]) -> float:
    return math.dist((float(bbox_xyxy[0]), float(bbox_xyxy[1])), (float(bbox_xyxy[2]), float(bbox_xyxy[3])))


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_x1 = max(float(box_a[0]), float(box_b[0]))
    intersection_y1 = max(float(box_a[1]), float(box_b[1]))
    intersection_x2 = min(float(box_a[2]), float(box_b[2]))
    intersection_y2 = min(float(box_a[3]), float(box_b[3]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = compute_bbox_area(box_a) + compute_bbox_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_inside_bbox(center: tuple[float, float], bbox_xyxy: list[float]) -> bool:
    return float(bbox_xyxy[0]) <= center[0] <= float(bbox_xyxy[2]) and float(bbox_xyxy[1]) <= center[1] <= float(bbox_xyxy[3])


def bbox_inside_bbox(inner_bbox: list[float], outer_bbox: list[float]) -> bool:
    return (
        float(outer_bbox[0]) <= float(inner_bbox[0]) <= float(inner_bbox[2]) <= float(outer_bbox[2])
        and float(outer_bbox[1]) <= float(inner_bbox[1]) <= float(inner_bbox[3]) <= float(outer_bbox[3])
    )


def read_settings() -> dict[str, Any]:
    return {
        "time_tolerance_seconds": read_float_env(
            ENV_FINAL_DEMO_ASSOC_TIME_TOLERANCE_SECONDS,
            DEFAULT_ASSOC_TIME_TOLERANCE_SECONDS,
        ),
        "person_object_max_distance": read_float_env(
            ENV_FINAL_DEMO_ASSOC_PERSON_OBJECT_MAX_DISTANCE,
            DEFAULT_ASSOC_PERSON_OBJECT_MAX_DISTANCE,
        ),
        "person_vehicle_max_distance": read_float_env(
            ENV_FINAL_DEMO_ASSOC_PERSON_VEHICLE_MAX_DISTANCE,
            DEFAULT_ASSOC_PERSON_VEHICLE_MAX_DISTANCE,
        ),
        "min_confidence": read_float_env(
            ENV_FINAL_DEMO_ASSOC_MIN_CONFIDENCE,
            DEFAULT_ASSOC_MIN_CONFIDENCE,
        ),
        "max_records": read_positive_int_env(
            ENV_FINAL_DEMO_ASSOC_MAX_RECORDS,
            DEFAULT_ASSOC_MAX_RECORDS,
        ),
        "debug_full": read_bool_env(ENV_FINAL_DEMO_ASSOC_DEBUG_FULL, False),
    }


def build_person_records(person_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(person_payload, dict):
        return records
    for item in list(person_payload.get("person_attributes") or []):
        if not isinstance(item, dict):
            continue
        bbox = list(item.get("bbox") or [])
        if len(bbox) < 4:
            continue
        records.append(
            {
                "entity_id": str(item.get("person_attribute_id") or ""),
                "entity_family": "person",
                "entity_type": "person",
                "record_source": str(item.get("record_source") or ""),
                "source_track_id": str(item.get("source_track_id") or ""),
                "attribute_track_id": str(item.get("attribute_track_id") or ""),
                "source_detection_id": str(item.get("source_detection_id") or ""),
                "start_time": round(as_float(item.get("start_time"), 0.0), 3),
                "end_time": round(as_float(item.get("end_time"), 0.0), 3),
                "representative_timestamp": round(as_float(item.get("representative_timestamp"), 0.0), 3),
                "bbox": [round(float(value), 3) for value in bbox[:4]],
                "frame_id": str(item.get("frame_id") or item.get("best_frame_id") or ""),
                "image_path": str(item.get("best_image_path") or ""),
                "crop_path": str(item.get("crop_path") or ""),
                "needs_review": bool(item.get("needs_review")),
                "quality_score": round(as_float(item.get("person_attribute_confidence"), 0.0), 3),
                "object_hints": clean_list(
                    [
                        item.get("carrying_object_type"),
                        *list(item.get("nearby_object_types") or []),
                    ]
                ),
                "raw": item,
            }
        )
    return records


def build_object_records(object_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(object_payload, dict):
        return records
    for item in list(object_payload.get("object_attributes") or []):
        if not isinstance(item, dict):
            continue
        bbox = list(item.get("bbox") or [])
        if len(bbox) < 4:
            continue
        records.append(
            {
                "entity_id": str(item.get("object_attribute_id") or ""),
                "entity_family": "object",
                "entity_type": str(item.get("normalized_object_type") or item.get("object_type") or item.get("class_name") or ""),
                "display_type": str(item.get("object_type") or item.get("class_name") or ""),
                "record_source": "object_attribute",
                "source_track_id": str(item.get("source_track_id") or ""),
                "attribute_track_id": str(item.get("attribute_track_id") or ""),
                "source_detection_id": str(item.get("source_detection_id") or ""),
                "start_time": round(as_float(item.get("start_time"), 0.0), 3),
                "end_time": round(as_float(item.get("end_time"), 0.0), 3),
                "representative_timestamp": round(as_float(item.get("representative_timestamp"), 0.0), 3),
                "bbox": [round(float(value), 3) for value in bbox[:4]],
                "frame_id": str(item.get("best_frame_id") or ""),
                "image_path": str(item.get("best_image_path") or ""),
                "crop_path": str(item.get("crop_path") or ""),
                "needs_review": bool(item.get("needs_review")),
                "quality_score": round(as_float(item.get("object_attribute_confidence"), 0.0), 3),
                "possible_actual_family": str(item.get("possible_actual_family") or ""),
                "possible_actual_types": clean_list(item.get("possible_actual_types")),
                "object_class_needs_review": bool(item.get("object_class_needs_review")),
                "false_positive_risk_score": round(as_float(item.get("false_positive_risk_score"), 0.0), 3),
                "attribute_status": str(item.get("attribute_status") or ""),
                "raw": item,
            }
        )
    return records


def build_vehicle_records(
    search_index_payload: dict[str, Any] | None,
    event_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    if isinstance(search_index_payload, dict):
        for item in list(search_index_payload.get("records") or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("entity_family") or "") != "vehicle":
                continue
            evidence = dict(item.get("evidence") or {})
            bbox = list(evidence.get("bbox") or [])
            if len(bbox) < 4:
                continue
            entity_id = str(item.get("source_event_candidate_id") or item.get("attribute_track_id") or item.get("source_track_id") or item.get("search_id") or "")
            key = (entity_id, str(item.get("record_type") or ""), str(item.get("entity_type") or ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(
                {
                    "entity_id": entity_id,
                    "entity_family": "vehicle",
                    "entity_type": str(item.get("entity_type") or item.get("class_name") or "vehicle"),
                    "record_source": str(item.get("record_type") or ""),
                    "source_track_id": str(item.get("source_track_id") or ""),
                    "attribute_track_id": str(item.get("attribute_track_id") or ""),
                    "source_detection_id": str(item.get("source_detection_id") or ""),
                    "source_event_candidate_id": str(item.get("source_event_candidate_id") or ""),
                    "start_time": round(as_float(item.get("start_time"), 0.0), 3),
                    "end_time": round(as_float(item.get("end_time"), 0.0), 3),
                    "representative_timestamp": round(as_float(item.get("representative_timestamp"), 0.0), 3),
                    "bbox": [round(float(value), 3) for value in bbox[:4]],
                    "frame_id": str(evidence.get("best_frame_id") or ""),
                    "image_path": str(evidence.get("best_image_path") or ""),
                    "crop_path": str(evidence.get("crop_path") or ""),
                    "needs_review": bool(item.get("needs_review")),
                    "quality_score": round(as_float(item.get("confidence"), 0.0), 3),
                    "raw": item,
                }
            )
    if isinstance(event_payload, dict):
        for item in list(event_payload.get("events") or event_payload.get("event_candidates") or []):
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class_name") or "")
            if class_name.lower() not in VEHICLE_TYPES:
                continue
            evidence = dict(item.get("evidence") or {})
            bbox = list(evidence.get("bbox") or [])
            if len(bbox) < 4:
                continue
            entity_id = str(item.get("event_candidate_id") or item.get("source_track_id") or item.get("attribute_track_id") or "")
            key = (entity_id, "event_candidate", class_name.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(
                {
                    "entity_id": entity_id,
                    "entity_family": "vehicle",
                    "entity_type": class_name.lower(),
                    "record_source": "event_candidate",
                    "source_track_id": str(item.get("source_track_id") or ""),
                    "attribute_track_id": str(item.get("attribute_track_id") or ""),
                    "source_detection_id": str(dict(item.get("attributes") or {}).get("source_detection_id") or ""),
                    "source_event_candidate_id": str(item.get("event_candidate_id") or ""),
                    "start_time": round(as_float(item.get("start_time"), 0.0), 3),
                    "end_time": round(as_float(item.get("end_time"), 0.0), 3),
                    "representative_timestamp": round(as_float(item.get("representative_timestamp"), 0.0), 3),
                    "bbox": [round(float(value), 3) for value in bbox[:4]],
                    "frame_id": str(evidence.get("best_frame_id") or ""),
                    "image_path": str(evidence.get("best_image_path") or ""),
                    "crop_path": str(evidence.get("crop_path") or ""),
                    "needs_review": bool(item.get("needs_review")),
                    "quality_score": round(as_float(item.get("confidence"), 0.0), 3),
                    "raw": item,
                }
            )
    records.sort(key=lambda item: (float(item["representative_timestamp"]), item["entity_type"], item["entity_id"]))
    return records


def geometry_metrics(subject_bbox: list[float], object_bbox: list[float], time_delta_seconds: float) -> dict[str, Any]:
    iou = round(compute_iou(subject_bbox, object_bbox), 4)
    subject_center = compute_bbox_center(subject_bbox)
    object_center = compute_bbox_center(object_bbox)
    center_distance = round(math.dist(subject_center, object_center), 3)
    subject_diagonal = max(1.0, compute_bbox_diagonal(subject_bbox))
    normalized_center_distance = round(center_distance / subject_diagonal, 4)
    return {
        "time_delta_seconds": round(abs(time_delta_seconds), 3),
        "bbox_iou": iou,
        "center_distance_pixels": center_distance,
        "normalized_center_distance": normalized_center_distance,
        "object_center_inside_subject_bbox": center_inside_bbox(object_center, subject_bbox),
        "object_bbox_inside_subject_bbox": bbox_inside_bbox(object_bbox, subject_bbox),
        "subject_bbox": [round(float(value), 3) for value in subject_bbox[:4]],
        "object_bbox": [round(float(value), 3) for value in object_bbox[:4]],
    }


def person_side_or_torso_region(person_bbox: list[float], object_bbox: list[float]) -> bool:
    px1, py1, px2, py2 = [float(value) for value in person_bbox[:4]]
    ox1, oy1, ox2, oy2 = [float(value) for value in object_bbox[:4]]
    person_center_x = (px1 + px2) / 2.0
    object_center_x = (ox1 + ox2) / 2.0
    object_center_y = (oy1 + oy2) / 2.0
    person_width = max(1.0, px2 - px1)
    torso_band = py1 + (py2 - py1) * 0.18 <= object_center_y <= py1 + (py2 - py1) * 0.88
    near_side = abs(object_center_x - person_center_x) <= person_width * 0.65
    return torso_band and near_side


def relationship_status(confidence: float, relationship: str) -> tuple[str, bool]:
    if confidence >= 0.75:
        return "confirmed_candidate", relationship == "carrying_candidate"
    if confidence >= 0.45:
        return "review_candidate", True
    return "weak_candidate", True


def build_search_keywords(
    association_type: str,
    relationship: str,
    subject_type: str,
    object_type: str,
    needs_review: bool,
) -> list[str]:
    items = [
        association_type,
        relationship,
        subject_type,
        object_type,
        "needs_review" if needs_review else "",
        f"{subject_type}_{relationship}",
        f"{subject_type}_{object_type}",
    ]
    result: list[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text and text not in result:
            result.append(text)
    return result


def build_match_facets(
    *,
    association_type: str,
    relationship: str,
    subject_family: str,
    object_family: str,
    subject_type: str,
    object_type: str,
    needs_review: bool,
) -> dict[str, list[str]]:
    return {
        "association_type": [association_type],
        "relationship": [relationship],
        "subject_entity_family": [subject_family],
        "object_entity_family": [object_family],
        "subject_entity_type": [subject_type],
        "object_entity_type": [object_type],
        "review_status": ["needs_review" if needs_review else "confirmed"],
    }


def build_association_record(
    *,
    association_id: str,
    association_type: str,
    relationship: str,
    subject: dict[str, Any],
    object_item: dict[str, Any],
    geometry: dict[str, Any],
    confidence: float,
    needs_review: bool,
    review_reason: str,
) -> dict[str, Any]:
    start_time = min(as_float(subject.get("start_time")), as_float(object_item.get("start_time")))
    end_time = max(as_float(subject.get("end_time")), as_float(object_item.get("end_time")))
    representative_timestamp = round(
        (as_float(subject.get("representative_timestamp")) + as_float(object_item.get("representative_timestamp"))) / 2.0,
        3,
    )
    association_status, default_review = relationship_status(confidence, relationship)
    final_needs_review = bool(needs_review or default_review)
    search_keywords = build_search_keywords(
        association_type,
        relationship,
        str(subject.get("entity_type") or ""),
        str(object_item.get("entity_type") or ""),
        final_needs_review,
    )
    return {
        "association_id": association_id,
        "association_type": association_type,
        "relationship": relationship,
        "subject_entity_family": str(subject.get("entity_family") or ""),
        "subject_entity_type": str(subject.get("entity_type") or ""),
        "subject_id": str(subject.get("entity_id") or ""),
        "subject_source_id": str(subject.get("source_detection_id") or subject.get("source_track_id") or subject.get("attribute_track_id") or ""),
        "subject_record_source": str(subject.get("record_source") or ""),
        "object_entity_family": str(object_item.get("entity_family") or ""),
        "object_entity_type": str(object_item.get("entity_type") or object_item.get("display_type") or ""),
        "object_id": str(object_item.get("entity_id") or ""),
        "object_source_id": str(object_item.get("source_detection_id") or object_item.get("source_track_id") or object_item.get("attribute_track_id") or object_item.get("source_event_candidate_id") or ""),
        "object_record_source": str(object_item.get("record_source") or ""),
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "representative_timestamp": representative_timestamp,
        "duration_seconds": round(max(0.0, end_time - start_time), 3),
        "confidence": round(confidence, 3),
        "needs_review": final_needs_review,
        "review_reason": review_reason,
        "association_status": association_status,
        "geometry": geometry,
        "evidence": {
            "frame_id": str(subject.get("frame_id") or object_item.get("frame_id") or ""),
            "image_path": str(subject.get("image_path") or object_item.get("image_path") or ""),
            "subject_crop_path": str(subject.get("crop_path") or ""),
            "object_crop_path": str(object_item.get("crop_path") or ""),
            "supporting_frame_ids": clean_list([subject.get("frame_id"), object_item.get("frame_id")]),
            "supporting_timestamps": [
                round(as_float(subject.get("representative_timestamp")), 3),
                round(as_float(object_item.get("representative_timestamp")), 3),
            ],
            "source_detection_ids": clean_list([subject.get("source_detection_id"), object_item.get("source_detection_id")]),
            "source_track_ids": clean_list([subject.get("source_track_id"), object_item.get("source_track_id"), object_item.get("attribute_track_id")]),
        },
        "search_keywords": search_keywords,
        "match_facets": build_match_facets(
            association_type=association_type,
            relationship=relationship,
            subject_family=str(subject.get("entity_family") or ""),
            object_family=str(object_item.get("entity_family") or ""),
            subject_type=str(subject.get("entity_type") or ""),
            object_type=str(object_item.get("entity_type") or object_item.get("display_type") or ""),
            needs_review=final_needs_review,
        ),
    }


def evaluate_person_object(
    subject: dict[str, Any],
    object_item: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[str | None, float, str, dict[str, Any]]:
    time_delta = abs(
        as_float(subject.get("representative_timestamp")) - as_float(object_item.get("representative_timestamp"))
    )
    if time_delta > settings["time_tolerance_seconds"]:
        return None, 0.0, "time_delta_exceeds_tolerance", {}
    geometry = geometry_metrics(list(subject["bbox"]), list(object_item["bbox"]), time_delta)
    object_type = str(object_item.get("entity_type") or object_item.get("display_type") or "").lower()
    carrying_like = object_type in CARRYING_OBJECT_TYPES
    same_frame = str(subject.get("frame_id") or "") and str(subject.get("frame_id") or "") == str(object_item.get("frame_id") or "")
    torso_region = person_side_or_torso_region(list(subject["bbox"]), list(object_item["bbox"]))
    strong_carry = (
        carrying_like
        and time_delta <= 0.5
        and (
            geometry["object_center_inside_subject_bbox"]
            or geometry["bbox_iou"] > 0.05
            or geometry["normalized_center_distance"] <= 0.45
            or torso_region
        )
    )
    if strong_carry:
        confidence = 0.55
        if same_frame:
            confidence += 0.08
        if geometry["object_center_inside_subject_bbox"]:
            confidence += 0.10
        if geometry["bbox_iou"] > 0.08:
            confidence += 0.08
        if torso_region:
            confidence += 0.05
        return "carrying_candidate", min(0.80, confidence), "carrying_candidate_needs_review", geometry

    near_match = geometry["normalized_center_distance"] <= settings["person_object_max_distance"] or geometry["bbox_iou"] > 0.01
    if near_match:
        relationship = "near"
        if same_frame and geometry["normalized_center_distance"] <= 0.8:
            relationship = "beside"
        confidence = 0.35
        if same_frame:
            confidence += 0.08
        if geometry["bbox_iou"] > 0.01:
            confidence += 0.08
        if geometry["normalized_center_distance"] <= 0.75:
            confidence += 0.10
        return relationship, min(0.65, confidence), "near_association_needs_review", geometry

    return None, 0.0, "geometry_too_weak", geometry


def evaluate_person_vehicle(
    subject: dict[str, Any],
    vehicle: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[str | None, float, str, dict[str, Any]]:
    time_delta = abs(as_float(subject.get("representative_timestamp")) - as_float(vehicle.get("representative_timestamp")))
    if time_delta > max(1.0, settings["time_tolerance_seconds"]):
        return None, 0.0, "time_delta_exceeds_vehicle_tolerance", {}
    geometry = geometry_metrics(list(subject["bbox"]), list(vehicle["bbox"]), time_delta)
    if geometry["normalized_center_distance"] > settings["person_vehicle_max_distance"] and geometry["bbox_iou"] <= 0.01:
        return None, 0.0, "person_not_near_vehicle", geometry
    relationship = "near"
    if geometry["normalized_center_distance"] <= 0.95 or geometry["bbox_iou"] > 0.02:
        relationship = "beside"
    confidence = 0.38
    if str(subject.get("frame_id") or "") and str(subject.get("frame_id") or "") == str(vehicle.get("frame_id") or ""):
        confidence += 0.10
    if geometry["bbox_iou"] > 0.01:
        confidence += 0.06
    if geometry["normalized_center_distance"] <= 0.75:
        confidence += 0.10
    return relationship, min(0.68, confidence), "person_vehicle_association_needs_review", geometry


def evaluate_object_vehicle(
    object_item: dict[str, Any],
    vehicle: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[str | None, float, str, dict[str, Any]]:
    time_delta = abs(as_float(object_item.get("representative_timestamp")) - as_float(vehicle.get("representative_timestamp")))
    if time_delta > max(1.0, settings["time_tolerance_seconds"]):
        return None, 0.0, "time_delta_exceeds_vehicle_tolerance", {}
    geometry = geometry_metrics(list(object_item["bbox"]), list(vehicle["bbox"]), time_delta)
    special_misclassification = (
        str(object_item.get("possible_actual_family") or "") == "vehicle"
        and bool(object_item.get("object_class_needs_review"))
        and (
            geometry["bbox_iou"] > 0.30
            or geometry["object_center_inside_subject_bbox"]
            or str(object_item.get("attribute_status") or "") == "possible_vehicle_misclassification"
        )
    )
    if special_misclassification:
        return "possible_vehicle_misclassification", 0.72, "object_overlaps_vehicle_detection", geometry
    if geometry["bbox_iou"] > 0.08 or geometry["object_center_inside_subject_bbox"]:
        return "overlapping", 0.52, "object_overlaps_vehicle_detection", geometry
    if geometry["normalized_center_distance"] <= 1.0:
        return "near", 0.40, "object_near_vehicle_needs_review", geometry
    return None, 0.0, "object_not_near_vehicle", geometry


def dedupe_associations(
    associations: list[dict[str, Any]],
    debug_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for association in associations:
        key = (
            str(association.get("subject_id") or ""),
            str(association.get("object_id") or ""),
            str(association.get("relationship") or ""),
            round(as_float(association.get("representative_timestamp"), 0.0), 1),
        )
        if key not in grouped:
            grouped[key] = association
            continue
        current = grouped[key]
        if as_float(association.get("confidence"), 0.0) > as_float(current.get("confidence"), 0.0):
            previous = grouped[key]
            association["evidence"]["supporting_frame_ids"] = clean_list(
                list(association["evidence"].get("supporting_frame_ids") or [])
                + list(previous["evidence"].get("supporting_frame_ids") or [])
            )
            association["evidence"]["supporting_timestamps"] = list(
                dict.fromkeys(
                    list(association["evidence"].get("supporting_timestamps") or [])
                    + list(previous["evidence"].get("supporting_timestamps") or [])
                )
            )
            grouped[key] = association
            debug_payload["dedup_decisions"].append({"dedup_key": key, "kept": association["association_id"], "dropped": previous["association_id"]})
        else:
            current["evidence"]["supporting_frame_ids"] = clean_list(
                list(current["evidence"].get("supporting_frame_ids") or [])
                + list(association["evidence"].get("supporting_frame_ids") or [])
            )
            current["evidence"]["supporting_timestamps"] = list(
                dict.fromkeys(
                    list(current["evidence"].get("supporting_timestamps") or [])
                    + list(association["evidence"].get("supporting_timestamps") or [])
                )
            )
            debug_payload["dedup_decisions"].append({"dedup_key": key, "kept": current["association_id"], "dropped": association["association_id"]})
    return list(grouped.values())


def misclassification_priority_key(association: dict[str, Any]) -> tuple[float, int, float, float, float]:
    geometry = dict(association.get("geometry") or {})
    return (
        as_float(geometry.get("bbox_iou"), 0.0),
        1 if bool(geometry.get("object_center_inside_subject_bbox")) else 0,
        -as_float(geometry.get("time_delta_seconds"), 999.0),
        -as_float(geometry.get("normalized_center_distance"), 999.0),
        as_float(association.get("confidence"), 0.0),
    )


def compact_possible_vehicle_misclassification_associations(
    associations: list[dict[str, Any]],
    debug_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    stats = {
        "duplicate_associations_collapsed": 0,
        "alternate_vehicle_evidence_count": 0,
    }

    for association in associations:
        if str(association.get("relationship") or "") != "possible_vehicle_misclassification":
            kept.append(association)
            continue
        key = (
            str(association.get("subject_id") or ""),
            str(association.get("relationship") or ""),
        )
        grouped.setdefault(key, []).append(association)

    for key, items in grouped.items():
        if len(items) == 1:
            item = items[0]
            item["alternate_vehicle_evidence"] = []
            item["alternate_vehicle_types"] = []
            item["alternate_vehicle_detection_ids"] = []
            item["alternate_vehicle_record_ids"] = []
            item["alternate_vehicle_count"] = 0
            kept.append(item)
            continue

        ordered = sorted(
            items,
            key=misclassification_priority_key,
            reverse=True,
        )
        best = ordered[0]
        alternates = ordered[1:]
        alternate_vehicle_evidence: list[dict[str, Any]] = []
        alternate_vehicle_types: list[str] = []
        alternate_vehicle_detection_ids: list[str] = []
        alternate_vehicle_record_ids: list[str] = []

        for alternate in alternates:
            alternate_geometry = dict(alternate.get("geometry") or {})
            alternate_evidence = dict(alternate.get("evidence") or {})
            alternate_type = str(alternate.get("object_entity_type") or "")
            alternate_record_id = str(alternate.get("object_id") or "")
            alternate_detection_ids = clean_list(alternate_evidence.get("source_detection_ids"))
            alternate_track_ids = clean_list(alternate_evidence.get("source_track_ids"))
            alternate_vehicle_evidence.append(
                {
                    "association_id": str(alternate.get("association_id") or ""),
                    "vehicle_record_id": alternate_record_id,
                    "vehicle_entity_type": alternate_type,
                    "confidence": round(as_float(alternate.get("confidence"), 0.0), 3),
                    "frame_id": str(alternate_evidence.get("frame_id") or ""),
                    "image_path": str(alternate_evidence.get("image_path") or ""),
                    "source_detection_ids": alternate_detection_ids,
                    "source_track_ids": alternate_track_ids,
                    "geometry": alternate_geometry,
                }
            )
            if alternate_type and alternate_type not in alternate_vehicle_types:
                alternate_vehicle_types.append(alternate_type)
            for detection_id in alternate_detection_ids:
                if detection_id not in alternate_vehicle_detection_ids:
                    alternate_vehicle_detection_ids.append(detection_id)
            if alternate_record_id and alternate_record_id not in alternate_vehicle_record_ids:
                alternate_vehicle_record_ids.append(alternate_record_id)
            best_evidence = best.get("evidence") or {}
            best_evidence["supporting_frame_ids"] = clean_list(
                list(best_evidence.get("supporting_frame_ids") or [])
                + list(alternate_evidence.get("supporting_frame_ids") or [])
                + [alternate_evidence.get("frame_id")]
            )
            best_evidence["supporting_timestamps"] = list(
                dict.fromkeys(
                    list(best_evidence.get("supporting_timestamps") or [])
                    + list(alternate_evidence.get("supporting_timestamps") or [])
                    + [alternate.get("representative_timestamp")]
                )
            )
            best_evidence["source_detection_ids"] = clean_list(
                list(best_evidence.get("source_detection_ids") or []) + alternate_detection_ids
            )
            best_evidence["source_track_ids"] = clean_list(
                list(best_evidence.get("source_track_ids") or []) + alternate_track_ids
            )
            best["evidence"] = best_evidence
            debug_payload["dedup_decisions"].append(
                {
                    "dedup_key": key,
                    "type": "possible_vehicle_misclassification_compaction",
                    "kept": str(best.get("association_id") or ""),
                    "dropped": str(alternate.get("association_id") or ""),
                }
            )

        best["alternate_vehicle_evidence"] = alternate_vehicle_evidence
        best["alternate_vehicle_types"] = alternate_vehicle_types
        best["alternate_vehicle_detection_ids"] = alternate_vehicle_detection_ids
        best["alternate_vehicle_record_ids"] = alternate_vehicle_record_ids
        best["alternate_vehicle_count"] = len(alternate_vehicle_evidence)
        stats["duplicate_associations_collapsed"] += len(alternates)
        stats["alternate_vehicle_evidence_count"] += len(alternate_vehicle_evidence)
        kept.append(best)

    kept.sort(
        key=lambda item: (
            -as_float(item.get("confidence"), 0.0),
            as_float(item.get("representative_timestamp"), 0.0),
            str(item.get("association_id") or ""),
        )
    )
    return kept, stats


def build_entity_association_outputs(run_dir: Path) -> dict[str, Any]:
    settings = read_settings()
    source_paths = {
        "person_attributes": run_dir / "10_person_attributes.json",
        "object_attributes": run_dir / "11_object_attributes.json",
        "search_index": run_dir / "08_attribute_search_index.json",
        "detections": run_dir / "04_yolo_detections.json",
        "tracks": run_dir / "05B_clean_tracks.json",
        "events": run_dir / "07B_event_candidates.json",
    }
    person_payload = read_optional_json(source_paths["person_attributes"])
    object_payload = read_optional_json(source_paths["object_attributes"])
    search_index_payload = read_optional_json(source_paths["search_index"])
    event_payload = read_optional_json(source_paths["events"])

    persons = build_person_records(person_payload)
    objects = build_object_records(object_payload)
    vehicles = build_vehicle_records(search_index_payload, event_payload)

    warnings: list[str] = []
    recommendations: list[str] = []
    if not persons:
        warnings.append("No person records exist for Step 12 association.")
    if not objects:
        warnings.append("No object records exist for Step 12 association.")
    if not vehicles:
        warnings.append("No vehicle records exist for Step 12 association.")

    debug_payload: dict[str, Any] = {
        "threshold_values": settings,
        "candidate_comparisons": [],
        "rejected_comparisons": [],
        "dedup_decisions": [],
    }
    associations: list[dict[str, Any]] = []
    association_counter = 1

    for person in persons:
        for object_item in objects:
            relationship, confidence, review_reason, geometry = evaluate_person_object(person, object_item, settings)
            debug_row = {
                "candidate_type": "person_object",
                "subject_id": person["entity_id"],
                "object_id": object_item["entity_id"],
                "relationship": relationship,
                "confidence": round(confidence, 3),
                "reason": review_reason,
                "geometry": geometry,
            }
            if relationship is None or confidence < settings["min_confidence"]:
                if settings["debug_full"]:
                    debug_payload["rejected_comparisons"].append(debug_row)
                continue
            debug_payload["candidate_comparisons"].append(debug_row)
            associations.append(
                build_association_record(
                    association_id=f"assoc_{association_counter:06d}",
                    association_type="person_object",
                    relationship=relationship,
                    subject=person,
                    object_item=object_item,
                    geometry=geometry,
                    confidence=confidence,
                    needs_review=bool(person.get("needs_review") or object_item.get("needs_review")),
                    review_reason=review_reason,
                )
            )
            association_counter += 1

    for person in persons:
        for vehicle in vehicles:
            relationship, confidence, review_reason, geometry = evaluate_person_vehicle(person, vehicle, settings)
            debug_row = {
                "candidate_type": "person_vehicle",
                "subject_id": person["entity_id"],
                "object_id": vehicle["entity_id"],
                "relationship": relationship,
                "confidence": round(confidence, 3),
                "reason": review_reason,
                "geometry": geometry,
            }
            if relationship is None or confidence < settings["min_confidence"]:
                if settings["debug_full"]:
                    debug_payload["rejected_comparisons"].append(debug_row)
                continue
            debug_payload["candidate_comparisons"].append(debug_row)
            associations.append(
                build_association_record(
                    association_id=f"assoc_{association_counter:06d}",
                    association_type="person_vehicle",
                    relationship=relationship,
                    subject=person,
                    object_item=vehicle,
                    geometry=geometry,
                    confidence=confidence,
                    needs_review=bool(person.get("needs_review") or vehicle.get("needs_review")),
                    review_reason=review_reason,
                )
            )
            association_counter += 1

    for object_item in objects:
        for vehicle in vehicles:
            relationship, confidence, review_reason, geometry = evaluate_object_vehicle(object_item, vehicle, settings)
            debug_row = {
                "candidate_type": "object_vehicle",
                "subject_id": object_item["entity_id"],
                "object_id": vehicle["entity_id"],
                "relationship": relationship,
                "confidence": round(confidence, 3),
                "reason": review_reason,
                "geometry": geometry,
            }
            if relationship is None or confidence < settings["min_confidence"]:
                if settings["debug_full"]:
                    debug_payload["rejected_comparisons"].append(debug_row)
                continue
            debug_payload["candidate_comparisons"].append(debug_row)
            associations.append(
                build_association_record(
                    association_id=f"assoc_{association_counter:06d}",
                    association_type="object_vehicle",
                    relationship=relationship,
                    subject=object_item,
                    object_item=vehicle,
                    geometry=geometry,
                    confidence=confidence,
                    needs_review=True,
                    review_reason=review_reason,
                )
            )
            association_counter += 1

    raw_candidate_associations_before_dedup = len(associations)
    associations = dedupe_associations(associations, debug_payload)
    associations, compaction_stats = compact_possible_vehicle_misclassification_associations(
        associations,
        debug_payload,
    )
    associations_after_dedup = len(associations)
    limited_associations = associations[: settings["max_records"]]

    if objects and not any(item.get("association_type") == "person_object" for item in limited_associations):
        recommendations.append("Person-object associations were weak. Consider better person/object tracking.")
    if any(
        str(item.get("relationship") or "") == "possible_vehicle_misclassification"
        for item in limited_associations
    ):
        recommendations.append("Vehicle-like object misclassifications exist. Consider a custom traffic/e-rickshaw model.")
    if any(
        str(item.get("relationship") or "") == "carrying_candidate"
        and bool(item.get("needs_review"))
        for item in limited_associations
    ):
        recommendations.append("Carrying-object links remain review-only. Consider VLM verification on selected crops later.")
    if not limited_associations:
        warnings.append("No associations were created in Step 12.")

    records_with_missing_ids = sum(
        1
        for association in limited_associations
        if not str(association.get("subject_source_id") or "").strip()
        or not str(association.get("object_source_id") or "").strip()
    )
    if records_with_missing_ids > 0:
        warnings.append("Some association records are missing source IDs.")

    possible_vehicle_review_count = sum(
        1
        for object_item in objects
        if bool(object_item.get("object_class_needs_review"))
        and str(object_item.get("possible_actual_family") or "") == "vehicle"
    )
    if possible_vehicle_review_count > 0:
        warnings.append("Many object records look vehicle-like and were associated for review.")

    associations_by_type = dict(sorted(Counter(str(item.get("association_type") or "") for item in limited_associations).items()))
    associations_by_relationship = dict(sorted(Counter(str(item.get("relationship") or "") for item in limited_associations).items()))

    created_at = current_timestamp()
    associations_payload = {
        "created_at": created_at,
        "source": {
            "person_attributes": source_paths["person_attributes"].name,
            "object_attributes": source_paths["object_attributes"].name,
            "search_index": source_paths["search_index"].name,
            "detections": source_paths["detections"].name,
            "tracks": source_paths["tracks"].name,
            "events": source_paths["events"].name,
        },
        "associations": limited_associations,
    }
    report_payload = {
        "overall_status": "completed",
        "person_records_loaded": len(persons),
        "object_records_loaded": len(objects),
        "vehicle_records_loaded": len(vehicles),
        "raw_candidate_associations_before_dedup": raw_candidate_associations_before_dedup,
        "associations_after_dedup": associations_after_dedup,
        "associations_created": len(limited_associations),
        "associations_by_type": associations_by_type,
        "associations_by_relationship": associations_by_relationship,
        "associations_needing_review": sum(1 for item in limited_associations if bool(item.get("needs_review"))),
        "person_object_associations": sum(1 for item in limited_associations if str(item.get("association_type")) == "person_object"),
        "person_vehicle_associations": sum(1 for item in limited_associations if str(item.get("association_type")) == "person_vehicle"),
        "object_vehicle_associations": sum(1 for item in limited_associations if str(item.get("association_type")) == "object_vehicle"),
        "possible_vehicle_misclassification_associations": sum(
            1 for item in limited_associations if str(item.get("relationship")) == "possible_vehicle_misclassification"
        ),
        "duplicate_associations_collapsed": compaction_stats["duplicate_associations_collapsed"],
        "alternate_vehicle_evidence_count": compaction_stats["alternate_vehicle_evidence_count"],
        "records_with_missing_ids": records_with_missing_ids,
        "warnings": warnings,
        "recommendations": recommendations,
        "created_at": created_at,
    }
    if not settings["debug_full"]:
        debug_payload["rejected_comparisons"] = debug_payload["rejected_comparisons"][:0]
        debug_payload["candidate_comparisons"] = debug_payload["candidate_comparisons"][:80]
        debug_payload["dedup_decisions"] = debug_payload["dedup_decisions"][:80]
    return {
        "associations_payload": associations_payload,
        "report_payload": report_payload,
        "debug_payload": debug_payload,
    }


def update_run_manifest_for_entity_association(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "12_entity_association" not in completed_steps:
        completed_steps.append("12_entity_association")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "13_enriched_search_index"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
