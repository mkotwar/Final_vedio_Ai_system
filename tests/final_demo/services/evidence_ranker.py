from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_RANK_TOP_N = "FINAL_DEMO_RANK_TOP_N"
ENV_FINAL_DEMO_RANK_TOP_VLM_CANDIDATES = "FINAL_DEMO_RANK_TOP_VLM_CANDIDATES"
ENV_FINAL_DEMO_RANK_GROUP_GAP_SECONDS = "FINAL_DEMO_RANK_GROUP_GAP_SECONDS"
ENV_FINAL_DEMO_RANK_MIN_SCORE = "FINAL_DEMO_RANK_MIN_SCORE"
ENV_FINAL_DEMO_RANK_DEBUG_FULL = "FINAL_DEMO_RANK_DEBUG_FULL"

DEFAULT_RANK_TOP_N = 50
DEFAULT_RANK_TOP_VLM_CANDIDATES = 10
DEFAULT_RANK_GROUP_GAP_SECONDS = 2.0
DEFAULT_RANK_MIN_SCORE = 0.20
DEFAULT_RANK_DEBUG_FULL = False


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


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


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    value = as_int(raw_value, -1)
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {raw_value!r}")
    return value


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    value = as_float(raw_value, -1.0)
    if value < 0:
        raise ValueError(f"Environment variable {env_name} must be >= 0. Received: {raw_value!r}")
    return value


def safe_round(value: Any, digits: int = 3) -> float:
    return round(as_float(value, 0.0), digits)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def flatten_strings(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            values.extend(flatten_strings(nested_key))
            values.extend(flatten_strings(nested_value))
        return values
    if isinstance(value, list):
        for item in value:
            values.extend(flatten_strings(item))
        return values
    text = str(value or "").strip()
    if text:
        values.append(text)
    return values


def merge_unique_list(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        for item in flatten_strings(value):
            if item not in merged:
                merged.append(item)
    return merged


def relationship_terms(record: dict[str, Any]) -> list[str]:
    attributes = dict(record.get("attributes") or {})
    relationships = list(record.get("relationships") or [])
    terms = [
        str(attributes.get("relationship") or ""),
        str(attributes.get("association_type") or ""),
    ]
    for relationship in relationships:
        if isinstance(relationship, dict):
            terms.append(str(relationship.get("relationship") or ""))
    return [term for term in terms if term]


def build_dedup_key(record: dict[str, Any]) -> tuple[Any, ...]:
    source_ids = dict(record.get("source_ids") or {})
    return (
        str(record.get("source_search_id") or record.get("search_id") or ""),
        str(source_ids.get("source_detection_id") or ""),
        str(source_ids.get("source_track_id") or ""),
        str(source_ids.get("person_attribute_id") or ""),
        str(source_ids.get("object_attribute_id") or ""),
        str(source_ids.get("association_id") or ""),
        round(as_float(record.get("representative_timestamp"), 0.0), 1),
    )


def compute_match_strength_score(match_strength: str) -> float:
    return {
        "strong": 1.0,
        "possible": 0.65,
        "review": 0.45,
    }.get(str(match_strength or "").strip().lower(), 0.45)


def compute_evidence_richness_score(record: dict[str, Any], attributes: dict[str, Any]) -> float:
    evidence = dict(record.get("evidence") or {})
    points = 0.0
    total = 9.0
    if has_text(evidence.get("image_path")):
        points += 1.0
    if has_text(evidence.get("crop_path")):
        points += 1.0
    if has_text(evidence.get("subject_crop_path")):
        points += 1.0
    if has_text(evidence.get("object_crop_path")):
        points += 1.0
    if clean_list(evidence.get("supporting_frame_ids")):
        points += 1.0
    if clean_list(evidence.get("source_detection_ids")):
        points += 1.0
    if clean_list(evidence.get("source_track_ids")):
        points += 1.0
    if has_text(attributes.get("plate_text")) or has_text(attributes.get("candidate_plate_text")):
        points += 1.0
    if list(record.get("relationships") or []):
        points += 1.0
    return clip01(points / total)


def compute_attribute_score(attributes: dict[str, Any]) -> float:
    points = 0.0
    total = 8.0
    if has_text(attributes.get("top_clothing_color")) or has_text(attributes.get("normalized_top_color")):
        points += 1.0
    if has_text(attributes.get("bottom_clothing_color")) or has_text(attributes.get("normalized_bottom_color")):
        points += 1.0
    if has_text(attributes.get("overall_clothing_color")):
        points += 1.0
    if has_text(attributes.get("object_color")) or has_text(attributes.get("normalized_color")):
        points += 1.0
    if has_text(attributes.get("vehicle_color")):
        points += 1.0
    if has_text(attributes.get("plate_text")) or has_text(attributes.get("candidate_plate_text")):
        points += 1.0
    if has_text(attributes.get("plate_ocr_status")) or has_text(attributes.get("plate_format_status")):
        points += 1.0
    if (
        has_text(attributes.get("class_correction_reason"))
        or has_text(attributes.get("safe_display_class_name"))
        or bool(attributes.get("vehicle_subtype_needs_review"))
        or bool(attributes.get("object_class_needs_review"))
    ):
        points += 1.0
    return clip01(points / total)


def compute_relationship_score(record: dict[str, Any], attributes: dict[str, Any]) -> float:
    points = 0.0
    total = 5.0
    if str(record.get("record_type") or "") == "association_record":
        points += 1.0
    if has_text(attributes.get("relationship")):
        points += 1.0
    if isinstance(attributes.get("geometry"), dict) and bool(attributes.get("geometry")):
        points += 1.0
    if list(attributes.get("alternate_vehicle_evidence") or []):
        points += 1.0
    if has_text(attributes.get("subject_entity_type")) and has_text(attributes.get("object_entity_type")):
        points += 1.0
    return clip01(points / total)


def compute_review_priority_score(record: dict[str, Any], attributes: dict[str, Any]) -> float:
    status = str(record.get("status") or "")
    class_name = str(record.get("class_name") or "")
    search_keywords = set(clean_list(record.get("search_keywords")))
    score = 0.0
    if status == "possible_vehicle_misclassification":
        score = max(score, 1.0)
    if class_name == "possible_vehicle_misclassification":
        score = max(score, 1.0)
    if "vehicle_like_object" in search_keywords:
        score = max(score, 0.95)
    if "object_overlaps_vehicle" in search_keywords:
        score = max(score, 0.95)
    if "false_suitcase_possible" in search_keywords:
        score = max(score, 0.95)
    if bool(record.get("needs_review")) and has_text(record.get("review_reason")):
        score = max(score, 0.70)
    if bool(attributes.get("vehicle_subtype_needs_review")):
        score = max(score, 0.85)
    if bool(attributes.get("object_class_needs_review")):
        score = max(score, 0.85)
    return clip01(score)


def is_important_review_evidence(record: dict[str, Any], attributes: dict[str, Any]) -> bool:
    status = str(record.get("status") or "")
    class_name = str(record.get("class_name") or "")
    review_reason = str(record.get("review_reason") or "")
    relationship = str(attributes.get("relationship") or "")
    association_type = str(attributes.get("association_type") or "")
    plate_ocr_status = str(attributes.get("plate_ocr_status") or "")
    search_keywords = set(clean_list(record.get("search_keywords")))
    return any(
        [
            status == "possible_vehicle_misclassification",
            class_name == "possible_vehicle_misclassification",
            "possible_vehicle_misclassification" in search_keywords,
            "vehicle_like_object" in search_keywords,
            "object_overlaps_vehicle" in search_keywords,
            "false_suitcase_possible" in search_keywords,
            "object_overlaps_vehicle" in review_reason,
            relationship == "possible_vehicle_misclassification",
            association_type == "object_vehicle" and bool(record.get("needs_review")),
            plate_ocr_status == "read_needs_review",
            bool(attributes.get("vehicle_subtype_needs_review")),
        ]
    )


def compute_plate_score(attributes: dict[str, Any]) -> float:
    score = 0.0
    if has_text(attributes.get("candidate_plate_text")):
        score += 0.35
    if has_text(attributes.get("plate_text")):
        score += 0.35
    if str(attributes.get("plate_format_status") or "") == "valid_indian_plate":
        score += 0.20
    elif str(attributes.get("plate_format_status") or "") == "possible_pattern":
        score += 0.10
    if has_text(attributes.get("plate_ocr_status")):
        score += 0.05
    if as_float(attributes.get("final_plate_confidence"), 0.0) > 0:
        score += 0.05
    return clip01(score)


def compute_time_context_score(record: dict[str, Any]) -> float:
    evidence = dict(record.get("evidence") or {})
    start_time = as_float(record.get("start_time"), 0.0)
    end_time = as_float(record.get("end_time"), 0.0)
    duration_seconds = as_float(record.get("duration_seconds"), 0.0)
    score = 0.0
    if start_time > 0 or end_time > 0:
        score += 0.30
    if duration_seconds > 0:
        score += 0.30
    if len(list(evidence.get("supporting_timestamps") or [])) > 1:
        score += 0.20
    if len(clean_list(evidence.get("supporting_frame_ids"))) > 1:
        score += 0.20
    return clip01(score)


def infer_match_strength(record: dict[str, Any]) -> str:
    explicit = str(record.get("match_strength") or "").strip().lower()
    if explicit in {"strong", "possible", "review"}:
        return explicit
    default_strength = str(record.get("match_strength_default") or "").strip().lower()
    if default_strength in {"strong", "possible", "review"}:
        return default_strength
    if bool(record.get("needs_review")):
        return "review"
    confidence = as_float(record.get("confidence"), 0.0)
    if confidence >= 0.70:
        return "strong"
    if confidence >= 0.45:
        return "possible"
    return "review"


def build_importance_reasons(
    record: dict[str, Any],
    attributes: dict[str, Any],
    ranking_factors: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if ranking_factors["plate_score"] >= 0.50:
        reasons.append("plate evidence present")
    if str(attributes.get("plate_format_status") or "") == "valid_indian_plate":
        reasons.append("valid Indian plate pattern")
    if str(record.get("record_type") or "") == "association_record":
        reasons.append("association evidence links entities")
    if ranking_factors["review_priority_score"] >= 0.85:
        reasons.append("important review evidence for demo honesty")
    if ranking_factors["evidence_richness_score"] >= 0.60:
        reasons.append("rich image or crop evidence available")
    if ranking_factors["attribute_score"] >= 0.50:
        reasons.append("useful extracted attributes available")
    if bool(record.get("needs_review")) and has_text(record.get("review_reason")):
        reasons.append(f"needs review: {record.get('review_reason')}")
    if not reasons:
        reasons.append("useful supporting evidence record")
    return reasons


def build_display_title(record: dict[str, Any], attributes: dict[str, Any]) -> str:
    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "")
    class_name = str(record.get("class_name") or "")
    plate_text = str(attributes.get("plate_text") or attributes.get("candidate_plate_text") or "")
    if plate_text:
        return f"Vehicle with plate {plate_text}"
    if class_name == "possible_vehicle_misclassification":
        return "Possible vehicle-like object misclassified as suitcase"
    if entity_family == "person":
        top_color = str(attributes.get("normalized_top_color") or attributes.get("top_clothing_color") or "")
        if top_color:
            return f"Person wearing {top_color}"
        return "Person evidence"
    if entity_family == "association":
        subject = str(attributes.get("subject_entity_type") or "entity")
        obj = str(attributes.get("object_entity_type") or "entity")
        relationship = str(attributes.get("relationship") or class_name or "association")
        return f"{subject} {relationship} {obj}".strip()
    if entity_family == "object":
        return f"Object evidence: {entity_type or class_name or 'object'}"
    if entity_family == "vehicle":
        return f"Vehicle evidence: {entity_type or class_name or 'vehicle'}"
    return f"Evidence: {entity_type or class_name or entity_family or 'record'}"


def build_time_label(record: dict[str, Any]) -> str:
    start_time = as_float(record.get("start_time"), 0.0)
    end_time = as_float(record.get("end_time"), 0.0)
    timestamp = as_float(record.get("representative_timestamp"), 0.0)
    if end_time > start_time:
        return f"{start_time:.2f}s-{end_time:.2f}s"
    return f"{timestamp:.2f}s"


def build_display_subtitle(record: dict[str, Any]) -> str:
    label = build_time_label(record)
    if bool(record.get("needs_review")):
        return f"{label} | Needs review"
    return f"{label} | {infer_match_strength(record)}"


def build_display_description(record: dict[str, Any], attributes: dict[str, Any]) -> str:
    class_name = str(record.get("class_name") or "")
    plate_text = str(attributes.get("plate_text") or attributes.get("candidate_plate_text") or "")
    if plate_text:
        vehicle_type = str(attributes.get("safe_display_vehicle_type") or attributes.get("vehicle_type") or "vehicle")
        ocr_status = str(attributes.get("plate_ocr_status") or "OCR result available")
        return f"Plate text detected as {plate_text}; {vehicle_type} evidence status: {ocr_status}."
    if class_name == "possible_vehicle_misclassification":
        return "YOLO object evidence overlaps vehicle detections; likely not a confirmed suitcase."
    if str(record.get("entity_family") or "") == "person":
        return "Rule-based clothing color extracted from person crop."
    if str(record.get("entity_family") or "") == "association":
        return "Rule-based association evidence links related entities at a shared time."
    return "Ranked evidence record from the enriched demo index."


def build_display_payload(record: dict[str, Any], attributes: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(record.get("evidence") or {})
    primary_image_path = (
        str(evidence.get("image_path") or "")
        or str(evidence.get("crop_path") or "")
        or str(evidence.get("subject_crop_path") or "")
        or str(evidence.get("object_crop_path") or "")
    )
    crop_path = (
        str(evidence.get("crop_path") or "")
        or str(evidence.get("subject_crop_path") or "")
        or str(evidence.get("object_crop_path") or "")
    )
    return {
        "title": build_display_title(record, attributes),
        "subtitle": build_display_subtitle(record),
        "description": build_display_description(record, attributes),
        "time_label": build_time_label(record),
        "review_badge": "Needs review" if bool(record.get("needs_review")) else "",
        "primary_image_path": primary_image_path,
        "crop_path": crop_path,
    }


def rank_bucket(score: float, needs_review: bool, important_review_evidence: bool) -> str:
    if important_review_evidence and needs_review and score >= 0.45:
        return "review_priority"
    if score >= 0.75 and not needs_review:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def build_ranked_evidence_record(
    record: dict[str, Any],
    *,
    evidence_id: str,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes = dict(record.get("attributes") or {})
    evidence = dict(record.get("evidence") or {})
    relationships = list(record.get("relationships") or [])
    match_strength = infer_match_strength(record)
    ranking_factors = {
        "match_strength_score": round(compute_match_strength_score(match_strength), 3),
        "confidence_score": round(clip01(as_float(record.get("confidence"), 0.3) or 0.3), 3),
        "evidence_richness_score": round(compute_evidence_richness_score(record, attributes), 3),
        "attribute_score": round(compute_attribute_score(attributes), 3),
        "relationship_score": round(compute_relationship_score(record, attributes), 3),
        "plate_score": round(compute_plate_score(attributes), 3),
        "review_priority_score": round(compute_review_priority_score(record, attributes), 3),
        "time_context_score": round(compute_time_context_score(record), 3),
    }
    score = (
        0.20 * ranking_factors["match_strength_score"]
        + 0.20 * ranking_factors["confidence_score"]
        + 0.15 * ranking_factors["evidence_richness_score"]
        + 0.10 * ranking_factors["attribute_score"]
        + 0.10 * ranking_factors["relationship_score"]
        + 0.10 * ranking_factors["plate_score"]
        + 0.10 * ranking_factors["review_priority_score"]
        + 0.05 * ranking_factors["time_context_score"]
    )
    score = round(clip01(score), 3)
    needs_review = bool(record.get("needs_review")) or match_strength == "review"
    if needs_review:
        match_strength = "review" if match_strength != "strong" else "review"
    important_review_evidence = is_important_review_evidence(record, attributes)
    ranked = {
        "rank": rank,
        "evidence_id": evidence_id,
        "source_search_id": str(record.get("search_id") or record.get("source_search_id") or ""),
        "source_record_type": str(record.get("record_type") or ""),
        "entity_family": str(record.get("entity_family") or ""),
        "entity_type": str(record.get("entity_type") or ""),
        "class_name": str(record.get("class_name") or ""),
        "start_time": safe_round(record.get("start_time")),
        "end_time": safe_round(record.get("end_time")),
        "representative_timestamp": safe_round(record.get("representative_timestamp")),
        "duration_seconds": safe_round(record.get("duration_seconds")),
        "ranking_score": score,
        "ranking_bucket": rank_bucket(score, needs_review, important_review_evidence),
        "match_strength": match_strength,
        "confidence": safe_round(record.get("confidence")),
        "needs_review": needs_review,
        "review_reason": str(record.get("review_reason") or ""),
        "status": str(record.get("status") or ""),
        "importance_reasons": build_importance_reasons(record, attributes, ranking_factors),
        "ranking_factors": ranking_factors,
        "source_ids": dict(record.get("source_ids") or {}),
        "evidence": evidence,
        "attributes": attributes,
        "relationships": relationships,
        "search_keywords": clean_list(record.get("search_keywords")),
        "match_facets": dict(record.get("match_facets") or {}),
        "display": build_display_payload(record, attributes),
    }
    debug_row = {
        "source_search_id": ranked["source_search_id"],
        "evidence_id": evidence_id,
        "ranking_score": score,
        "ranking_factors": ranking_factors,
        "importance_reasons": ranked["importance_reasons"],
    }
    return ranked, debug_row


def merge_ranked_records(winner: dict[str, Any], loser: dict[str, Any]) -> dict[str, Any]:
    winner["importance_reasons"] = merge_unique_list(winner.get("importance_reasons"), loser.get("importance_reasons"))
    winner["search_keywords"] = merge_unique_list(winner.get("search_keywords"), loser.get("search_keywords"))
    winner["relationships"] = list(winner.get("relationships") or []) + [
        item for item in list(loser.get("relationships") or []) if item not in list(winner.get("relationships") or [])
    ]
    winner_evidence = dict(winner.get("evidence") or {})
    loser_evidence = dict(loser.get("evidence") or {})
    for field in ["supporting_frame_ids", "supporting_timestamps", "source_detection_ids", "source_track_ids"]:
        winner_evidence[field] = merge_unique_list(winner_evidence.get(field), loser_evidence.get(field))
    for field in ["image_path", "crop_path", "subject_crop_path", "object_crop_path", "frame_id"]:
        if not has_text(winner_evidence.get(field)) and has_text(loser_evidence.get(field)):
            winner_evidence[field] = loser_evidence.get(field)
    winner["evidence"] = winner_evidence
    winner["display"] = build_display_payload(winner, dict(winner.get("attributes") or {}))
    return winner


def dedupe_ranked_records(
    ranked_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    debug_rows: list[dict[str, Any]] = []
    duplicates_collapsed = 0
    for record in ranked_records:
        key = build_dedup_key(record)
        if key not in grouped:
            grouped[key] = record
            continue
        duplicates_collapsed += 1
        current = grouped[key]
        if as_float(record.get("ranking_score"), 0.0) > as_float(current.get("ranking_score"), 0.0):
            winner = record
            loser = current
        else:
            winner = current
            loser = record
        grouped[key] = merge_ranked_records(winner, loser)
        debug_rows.append(
            {
                "dedup_key": list(key),
                "winner": grouped[key].get("evidence_id"),
                "loser": loser.get("evidence_id"),
                "reason": "higher_ranking_score",
            }
        )
    deduped = list(grouped.values())
    deduped.sort(
        key=lambda item: (
            -as_float(item.get("ranking_score"), 0.0),
            {"high": 0, "review_priority": 1, "medium": 2, "low": 3}.get(str(item.get("ranking_bucket") or ""), 9),
            -as_float(item.get("confidence"), 0.0),
            as_float(item.get("representative_timestamp"), 0.0),
        )
    )
    for index, record in enumerate(deduped, start=1):
        record["rank"] = index
    return deduped, duplicates_collapsed, debug_rows


def select_top_records(records: list[dict[str, Any]], top_n: int, min_score: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        if as_float(record.get("ranking_score"), 0.0) < min_score:
            rejected.append(
                {
                    "source_search_id": record.get("source_search_id"),
                    "reason": "below_min_score",
                    "ranking_score": record.get("ranking_score"),
                }
            )
            continue
        if len(kept) >= top_n:
            rejected.append(
                {
                    "source_search_id": record.get("source_search_id"),
                    "reason": "beyond_top_n",
                    "ranking_score": record.get("ranking_score"),
                }
            )
            continue
        kept.append(record)
    for index, record in enumerate(kept, start=1):
        record["rank"] = index
        record["evidence_id"] = f"ranked_evt_{index:06d}"
    return kept, rejected


def summarize_relationships(records: list[dict[str, Any]]) -> list[str]:
    terms = Counter()
    for record in records:
        for term in relationship_terms(record):
            if term:
                terms[str(term)] += 1
    return [item[0] for item in terms.most_common(3)]


def build_timeline_groups(records: list[dict[str, Any]], gap_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_records = sorted(records, key=lambda item: as_float(item.get("representative_timestamp"), 0.0))
    groups: list[list[dict[str, Any]]] = []
    debug_rows: list[dict[str, Any]] = []
    for record in sorted_records:
        if not groups:
            groups.append([record])
            continue
        previous = groups[-1][-1]
        if as_float(record.get("representative_timestamp"), 0.0) - as_float(previous.get("representative_timestamp"), 0.0) <= gap_seconds:
            groups[-1].append(record)
        else:
            groups.append([record])
    timeline_groups: list[dict[str, Any]] = []
    for group_index, group_records in enumerate(groups, start=1):
        start_time = min(as_float(item.get("start_time"), as_float(item.get("representative_timestamp"), 0.0)) for item in group_records)
        end_time = max(as_float(item.get("end_time"), as_float(item.get("representative_timestamp"), 0.0)) for item in group_records)
        timestamps = [as_float(item.get("representative_timestamp"), 0.0) for item in group_records]
        entity_families = Counter(str(item.get("entity_family") or "") for item in group_records if str(item.get("entity_family") or ""))
        entity_types = Counter(str(item.get("entity_type") or "") for item in group_records if str(item.get("entity_type") or ""))
        relationships = summarize_relationships(group_records)
        top_records = sorted(group_records, key=lambda item: -as_float(item.get("ranking_score"), 0.0))
        display_title = top_records[0].get("display", {}).get("title", "Evidence group")
        timeline_groups.append(
            {
                "timeline_group_id": f"timeline_grp_{group_index:06d}",
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "representative_timestamp": round(sum(timestamps) / max(len(timestamps), 1), 3),
                "evidence_count": len(group_records),
                "top_evidence_ids": [str(item.get("evidence_id") or "") for item in top_records[:5]],
                "dominant_entity_families": [item[0] for item in entity_families.most_common(3)],
                "dominant_entity_types": [item[0] for item in entity_types.most_common(3)],
                "dominant_relationships": relationships,
                "needs_review": any(bool(item.get("needs_review")) for item in group_records),
                "group_score": round(max(as_float(item.get("ranking_score"), 0.0) for item in group_records), 3),
                "display_title": display_title,
                "display_description": f"{len(group_records)} ranked evidence records grouped within {gap_seconds:.1f}s.",
            }
        )
        debug_rows.append(
            {
                "timeline_group_id": f"timeline_grp_{group_index:06d}",
                "evidence_ids": [str(item.get("evidence_id") or "") for item in group_records],
                "reason": "grouped_by_time_gap",
            }
        )
    return timeline_groups, debug_rows


def build_vlm_candidates(records: list[dict[str, Any]], top_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for record in records:
        attributes = dict(record.get("attributes") or {})
        search_keywords = set(clean_list(record.get("search_keywords")))
        candidate_reason = ""
        prompt_type = ""
        if has_text(attributes.get("candidate_plate_text")) or has_text(attributes.get("plate_text")):
            candidate_reason = "plate OCR evidence needs verification"
            prompt_type = "verify_plate"
        elif str(record.get("status") or "") == "possible_vehicle_misclassification" or "vehicle_like_object" in search_keywords:
            candidate_reason = "vehicle-like object review evidence needs type verification"
            prompt_type = "verify_object_type"
        elif bool(attributes.get("vehicle_subtype_needs_review")):
            candidate_reason = "vehicle subtype needs review"
            prompt_type = "verify_vehicle_type"
        elif str(record.get("entity_family") or "") == "person" and bool(record.get("needs_review")):
            candidate_reason = "person attribute evidence needs review"
            prompt_type = "verify_person_attribute"
        elif str(record.get("entity_family") or "") == "association" and bool(record.get("needs_review")):
            candidate_reason = "association evidence needs explanation"
            prompt_type = "explain_event"
        elif bool(record.get("needs_review")) and as_float(record.get("confidence"), 0.0) >= 0.70:
            candidate_reason = "high confidence review evidence needs explanation"
            prompt_type = "explain_event"
        if not candidate_reason:
            continue
        evidence = dict(record.get("evidence") or {})
        priority = "high" if as_float(record.get("ranking_score"), 0.0) >= 0.70 else ("medium" if as_float(record.get("ranking_score"), 0.0) >= 0.50 else "low")
        candidate = {
            "vlm_candidate_id": f"vlm_cand_{len(candidates) + 1:06d}",
            "source_evidence_id": str(record.get("evidence_id") or ""),
            "candidate_reason": candidate_reason,
            "priority": priority,
            "image_path": str(
                evidence.get("image_path")
                or evidence.get("crop_path")
                or evidence.get("subject_crop_path")
                or evidence.get("object_crop_path")
                or ""
            ),
            "crop_path": str(
                evidence.get("crop_path")
                or evidence.get("subject_crop_path")
                or evidence.get("object_crop_path")
                or ""
            ),
            "supporting_frame_ids": clean_list(evidence.get("supporting_frame_ids")),
            "start_time": safe_round(record.get("start_time")),
            "end_time": safe_round(record.get("end_time")),
            "suggested_prompt_type": prompt_type,
        }
        candidates.append(candidate)
        debug_rows.append(
            {
                "source_evidence_id": candidate["source_evidence_id"],
                "priority": priority,
                "candidate_reason": candidate_reason,
            }
        )
    candidates.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("priority") or ""), 9),
            as_float(next((record.get("ranking_score") for record in records if record.get("evidence_id") == item.get("source_evidence_id")), 0.0), 0.0) * -1,
        )
    )
    return candidates[:top_n], debug_rows[:top_n]


def rank_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_records: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        ranked, debug_row = build_ranked_evidence_record(record, evidence_id=f"ranked_evt_src_{index:06d}", rank=index)
        ranked_records.append(ranked)
        debug_rows.append(debug_row)
    ranked_records.sort(
        key=lambda item: (
            -as_float(item.get("ranking_score"), 0.0),
            {"high": 0, "review_priority": 1, "medium": 2, "low": 3}.get(str(item.get("ranking_bucket") or ""), 9),
            -as_float(item.get("confidence"), 0.0),
            as_float(item.get("representative_timestamp"), 0.0),
        )
    )
    for index, record in enumerate(ranked_records, start=1):
        record["rank"] = index
    return ranked_records, debug_rows


def build_query_ranked_results(
    search_results_payload: dict[str, Any] | None,
    *,
    top_n: int,
    min_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not search_results_payload:
        return [], []
    query_results = list(search_results_payload.get("queries") or [])
    ranked_queries: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for query_entry in query_results:
        query = str(query_entry.get("query") or "")
        combined_records = (
            list(query_entry.get("strong_results") or [])
            + list(query_entry.get("possible_results") or [])
            + list(query_entry.get("review_results") or [])
        )
        ranked_records, score_debug = rank_records(combined_records)
        deduped_records, _, dedup_debug = dedupe_ranked_records(ranked_records)
        trimmed_records, rejected_rows = select_top_records(deduped_records, top_n, min_score)
        query_payload = {
            "query": query,
            "total_results": len(trimmed_records),
            "ranked_results": trimmed_records,
            "best_result": trimmed_records[0] if trimmed_records else {},
            "notes": list(query_entry.get("notes") or []) + ([str(query_entry.get("no_match_reason") or "")] if not trimmed_records and has_text(query_entry.get("no_match_reason")) else []),
        }
        ranked_queries.append(query_payload)
        debug_rows.append(
            {
                "query": query,
                "score_factors": score_debug[:20],
                "dedup_decisions": dedup_debug[:20],
                "rejected_records": rejected_rows[:20],
            }
        )
    return ranked_queries, debug_rows


def build_evidence_ranking_outputs(
    run_dir: Path,
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    index_path = run_dir / "13_enriched_search_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required Step 15 input: {index_path}")

    search_results_path = run_dir / "14_enriched_search_results.json"
    search_report_path = run_dir / "14_enriched_search_query_report.json"
    associations_path = run_dir / "12_entity_associations.json"
    events_path = run_dir / "07B_event_candidates.json"
    frames_path = run_dir / "03_sampled_frames_index.json"

    enriched_index_payload = read_json(index_path)
    search_results_payload = read_optional_json(search_results_path)
    search_report_payload = read_optional_json(search_report_path)
    _associations_payload = read_optional_json(associations_path)
    _events_payload = read_optional_json(events_path)
    _frames_payload = read_optional_json(frames_path)

    top_n = read_positive_int_env(ENV_FINAL_DEMO_RANK_TOP_N, DEFAULT_RANK_TOP_N)
    top_vlm_n = read_positive_int_env(ENV_FINAL_DEMO_RANK_TOP_VLM_CANDIDATES, DEFAULT_RANK_TOP_VLM_CANDIDATES)
    group_gap_seconds = read_non_negative_float_env(ENV_FINAL_DEMO_RANK_GROUP_GAP_SECONDS, DEFAULT_RANK_GROUP_GAP_SECONDS)
    min_score = read_non_negative_float_env(ENV_FINAL_DEMO_RANK_MIN_SCORE, DEFAULT_RANK_MIN_SCORE)
    debug_full_enabled = debug_full or read_bool_env(ENV_FINAL_DEMO_RANK_DEBUG_FULL, DEFAULT_RANK_DEBUG_FULL)

    records = list(enriched_index_payload.get("records") or [])
    ranked_records, score_debug = rank_records(records)
    deduped_records, duplicates_collapsed, dedup_debug = dedupe_ranked_records(ranked_records)
    global_ranked_evidence, rejected_rows = select_top_records(deduped_records, top_n, min_score)
    timeline_groups, timeline_debug = build_timeline_groups(global_ranked_evidence, group_gap_seconds)
    top_vlm_candidates, vlm_debug = build_vlm_candidates(global_ranked_evidence, top_vlm_n)
    query_ranked_evidence, query_debug = build_query_ranked_results(
        search_results_payload,
        top_n=top_n,
        min_score=min_score,
    )

    report_records = global_ranked_evidence
    records_by_ranking_bucket = Counter(str(item.get("ranking_bucket") or "") for item in report_records)
    records_by_match_strength = Counter(str(item.get("match_strength") or "") for item in report_records)
    records_by_entity_family = Counter(str(item.get("entity_family") or "") for item in report_records)
    records_by_record_type = Counter(str(item.get("source_record_type") or "") for item in report_records)
    warnings: list[str] = []
    recommendations: list[str] = []

    if search_results_payload is None:
        warnings.append("Optional Step 14 search results were missing; query-specific ranking was not created.")
    if not global_ranked_evidence:
        warnings.append("No ranked evidence was created.")
    review_count = sum(1 for item in report_records if str(item.get("match_strength") or "") == "review")
    if report_records and review_count / len(report_records) >= 0.60:
        warnings.append("Many ranked evidence records are review-only; preserve uncertainty in demo presentation.")
    if top_vlm_candidates and sum(1 for item in global_ranked_evidence[: len(top_vlm_candidates)] if bool(item.get("needs_review"))) >= max(1, len(top_vlm_candidates) // 2):
        warnings.append("Top VLM candidates are mostly review records.")

    if any(has_text(item.get("attributes", {}).get("candidate_plate_text")) and bool(item.get("needs_review")) for item in report_records):
        recommendations.append("Plate OCR review evidence exists; recommend Step 16 VLM verification for selected plate evidence.")
    if any("vehicle_like_object" in clean_list(item.get("search_keywords")) for item in report_records):
        recommendations.append("Vehicle-like object review evidence exists; recommend Step 16 VLM verification for object type.")
    if sum(1 for item in report_records if bool(item.get("attributes", {}).get("vehicle_subtype_needs_review"))) >= 3:
        recommendations.append("Many vehicle subtype records need review; consider a custom traffic or e-rickshaw model.")

    results_payload = {
        "created_at": current_timestamp(),
        "source": {
            "enriched_index": "13_enriched_search_index.json",
            "enriched_search_results": "14_enriched_search_results.json" if search_results_payload is not None else None,
        },
        "ranking_version": "final_demo_evidence_rank_v1",
        "global_ranked_evidence": global_ranked_evidence,
        "query_ranked_evidence": query_ranked_evidence,
        "top_vlm_candidates": top_vlm_candidates,
        "timeline_groups": timeline_groups,
    }
    report_payload = {
        "overall_status": "completed",
        "enriched_records_loaded": len(records),
        "search_queries_loaded": len(list(search_results_payload.get("queries") or [])) if search_results_payload is not None else 0,
        "global_ranked_evidence_count": len(global_ranked_evidence),
        "query_ranked_evidence_count": len(query_ranked_evidence),
        "timeline_groups_created": len(timeline_groups),
        "top_vlm_candidates_created": len(top_vlm_candidates),
        "records_by_ranking_bucket": dict(sorted(records_by_ranking_bucket.items())),
        "records_by_match_strength": dict(sorted(records_by_match_strength.items())),
        "records_by_entity_family": dict(sorted(records_by_entity_family.items())),
        "records_by_record_type": dict(sorted(records_by_record_type.items())),
        "review_priority_records": sum(1 for item in report_records if str(item.get("ranking_bucket") or "") == "review_priority"),
        "important_review_records": sum(
            1
            for item in report_records
            if is_important_review_evidence(item, dict(item.get("attributes") or {}))
        ),
        "plate_review_priority_records": sum(
            1
            for item in report_records
            if str(item.get("ranking_bucket") or "") == "review_priority"
            and str(item.get("attributes", {}).get("plate_ocr_status") or "") == "read_needs_review"
        ),
        "vehicle_like_review_priority_records": sum(
            1
            for item in report_records
            if str(item.get("ranking_bucket") or "") == "review_priority"
            and (
                str(item.get("status") or "") == "possible_vehicle_misclassification"
                or str(item.get("class_name") or "") == "possible_vehicle_misclassification"
                or "vehicle_like_object" in clean_list(item.get("search_keywords"))
                or "object_overlaps_vehicle" in clean_list(item.get("search_keywords"))
                or str(item.get("attributes", {}).get("relationship") or "") == "possible_vehicle_misclassification"
            )
        ),
        "plate_evidence_records": sum(1 for item in report_records if has_text(item.get("attributes", {}).get("candidate_plate_text")) or has_text(item.get("attributes", {}).get("plate_text"))),
        "possible_vehicle_misclassification_records": sum(
            1
            for item in report_records
            if str(item.get("status") or "") == "possible_vehicle_misclassification"
            or str(item.get("class_name") or "") == "possible_vehicle_misclassification"
        ),
        "person_attribute_records_ranked": sum(1 for item in report_records if str(item.get("source_record_type") or "") == "person_attribute_record"),
        "object_attribute_records_ranked": sum(1 for item in report_records if str(item.get("source_record_type") or "") == "object_attribute_record"),
        "association_records_ranked": sum(1 for item in report_records if str(item.get("source_record_type") or "") == "association_record"),
        "duplicates_collapsed": duplicates_collapsed,
        "warnings": warnings,
        "recommendations": recommendations,
    }
    debug_payload = {
        "created_at": current_timestamp(),
        "score_factors_per_record": score_debug[: (len(score_debug) if debug_full_enabled else 80)],
        "rejected_records": rejected_rows if debug_full_enabled else rejected_rows[:80],
        "dedup_decisions": dedup_debug[: (len(dedup_debug) if debug_full_enabled else 80)],
        "timeline_grouping_decisions": timeline_debug[: (len(timeline_debug) if debug_full_enabled else 80)],
        "vlm_candidate_selection_reasons": vlm_debug[: (len(vlm_debug) if debug_full_enabled else 80)],
        "query_debug": query_debug[: (len(query_debug) if debug_full_enabled else 40)],
    }
    if not global_ranked_evidence and records:
        debug_payload["warnings"] = ["All records fell below ranking thresholds or were filtered."]
    return {
        "results_payload": results_payload,
        "report_payload": report_payload,
        "debug_payload": debug_payload,
    }


def update_run_manifest_for_evidence_ranking(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "15_evidence_ranking" not in completed_steps:
        completed_steps.append("15_evidence_ranking")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "16_vlm_verification"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
