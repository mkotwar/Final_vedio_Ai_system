from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


DEFAULT_ENRICHED_DEMO_QUERIES = [
    "find all persons",
    "find person wearing gray",
    "find person wearing black",
    "find gray person",
    "find all vehicles",
    "find car",
    "find truck",
    "find vehicle with plate HR38AE1442",
    "find suitcase",
    "find confirmed suitcase",
    "find object needing review",
    "find vehicle-like object",
    "find possible vehicle misclassification",
    "find object overlapping vehicle",
    "find uncertain object",
    "find all review records",
]

ENV_FINAL_DEMO_ENRICHED_SEARCH_DEBUG_FULL = "FINAL_DEMO_ENRICHED_SEARCH_DEBUG_FULL"

SINGULAR_NORMALIZATION = {
    "persons": "person",
    "people": "person",
    "human": "person",
    "humans": "person",
    "vehicles": "vehicle",
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "motorcycles": "motorcycle",
    "bikes": "bike",
    "objects": "object",
    "bags": "bag",
}
COLOR_TERMS = {
    "black", "white", "gray", "grey", "silver", "brown", "red", "blue",
    "green", "yellow", "orange", "pink", "purple",
}
PERSON_TERMS = {"person", "people", "human"}
VEHICLE_TERMS = {"vehicle", "car", "truck", "bus", "motorcycle", "bike"}
OBJECT_TERMS = {
    "object", "suitcase", "bag", "backpack", "handbag", "luggage", "laptop",
    "phone", "bottle", "uncertain object", "vehicle like object",
}
CONFIRM_TERMS = {"confirmed", "sure", "strong", "definite"}
REVIEW_TERMS = {
    "review", "needs review", "needing review", "uncertain", "possible",
    "suspicious", "misclassification", "false positive", "vehicle like",
}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


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


def normalize_text(value: str) -> str:
    text = str(value or "").lower().replace(",", " ").replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def normalize_plate_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


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


def extract_keywords(record: dict[str, Any]) -> set[str]:
    keywords = set(clean_list(record.get("search_keywords")))
    match_facets = dict(record.get("match_facets") or {})
    for value in match_facets.values():
        keywords.update(clean_list(value))
    attributes = dict(record.get("attributes") or {})
    for key, value in attributes.items():
        if isinstance(value, list):
            keywords.update(clean_list(value))
        else:
            text = str(value or "").strip().lower()
            if text and len(text) <= 64:
                keywords.add(text)
    return keywords


def _flatten_search_values(value: Any) -> list[str]:
    flattened: list[str] = []
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            flattened.extend(_flatten_search_values(nested_key))
            flattened.extend(_flatten_search_values(nested_value))
        return flattened
    if isinstance(value, list):
        for item in value:
            flattened.extend(_flatten_search_values(item))
        return flattened
    normalized_value = normalize_text(str(value or ""))
    if normalized_value:
        flattened.append(normalized_value)
        flattened.append(normalized_value.replace(" ", "_"))
    return flattened


def record_search_blob(record: dict[str, Any]) -> str:
    base_values: list[Any] = [
        record.get("record_type"),
        record.get("entity_family"),
        record.get("entity_type"),
        record.get("class_name"),
        record.get("safe_class_name"),
        record.get("raw_class_name"),
        record.get("status"),
        record.get("review_reason"),
        record.get("search_keywords"),
        record.get("match_facets"),
        record.get("attributes"),
        record.get("relationships"),
        record.get("source_ids"),
    ]
    searchable_terms: set[str] = set()
    for value in base_values:
        searchable_terms.update(_flatten_search_values(value))
    return " ".join(sorted(term for term in searchable_terms if term))


def parse_query(query: str) -> dict[str, Any]:
    normalized = normalize_text(query)
    raw_tokens = [token for token in normalized.split(" ") if token]
    tokens = [SINGULAR_NORMALIZATION.get(token, token) for token in raw_tokens]
    token_set = set(tokens)
    colors: list[str] = []
    for token in tokens:
        if token in COLOR_TERMS:
            normalized_color = "gray" if token == "grey" else token
            if normalized_color not in colors:
                colors.append(normalized_color)
    color = colors[0] if colors else ""
    plate_token = ""
    for token in raw_tokens:
        normalized_plate = normalize_plate_token(token)
        if len(normalized_plate) >= 6 and any(ch.isalpha() for ch in normalized_plate) and any(ch.isdigit() for ch in normalized_plate):
            plate_token = normalized_plate
            break
    query_lower = normalized
    wants_confirmed = any(term in query_lower for term in CONFIRM_TERMS)
    wants_review = any(term in query_lower for term in REVIEW_TERMS) or "all review records" in query_lower
    wants_vehicle_like = any(
        phrase in query_lower
        for phrase in ["vehicle like object", "vehicle_like_object"]
    )
    wants_misclassification = any(
        phrase in query_lower
        for phrase in [
            "possible vehicle misclassification",
            "vehicle misclassification",
            "misclassification",
            "false suitcase",
            "false positive",
        ]
    )
    wants_overlapping_vehicle = any(
        phrase in query_lower
        for phrase in [
            "object overlapping vehicle",
            "object overlaps vehicle",
            "overlapping vehicle",
            "overlaps vehicle",
            "object_overlaps_vehicle",
        ]
    )
    entity = ""
    if "person" in token_set or "people" in token_set or "human" in token_set:
        entity = "person"
    elif "truck" in token_set:
        entity = "truck"
    elif "bus" in token_set:
        entity = "bus"
    elif "motorcycle" in token_set or "bike" in token_set:
        entity = "motorcycle"
    elif "car" in token_set:
        entity = "car"
    elif "vehicle" in token_set:
        entity = "vehicle"
    elif "suitcase" in token_set or "luggage" in token_set:
        entity = "suitcase"
    elif "backpack" in token_set:
        entity = "backpack"
    elif "bag" in token_set or "handbag" in token_set:
        entity = "bag"
    elif "laptop" in token_set:
        entity = "laptop"
    elif "phone" in token_set:
        entity = "phone"
    elif "bottle" in token_set:
        entity = "bottle"
    elif "uncertain" in token_set and "object" in token_set:
        entity = "uncertain_object"
    elif "object" in token_set:
        entity = "object"
    wants_all = "all" in token_set
    wants_person_clothing = "wearing" in token_set or "shirt" in token_set or (entity == "person" and bool(color))
    return {
        "normalized_query": normalized,
        "tokens": tokens,
        "entity": entity,
        "color": color,
        "colors": colors,
        "plate_token": plate_token,
        "wants_confirmed": wants_confirmed,
        "wants_review": wants_review,
        "wants_all": wants_all,
        "wants_person_clothing": wants_person_clothing,
        "wants_vehicle_like": wants_vehicle_like,
        "wants_misclassification": wants_misclassification,
        "wants_overlapping_vehicle": wants_overlapping_vehicle,
    }


def group_key_for_record(record: dict[str, Any]) -> tuple[str, ...]:
    source_ids = dict(record.get("source_ids") or {})
    if str(record.get("record_type") or "") == "association_record" and str(source_ids.get("association_id") or ""):
        return ("association", str(source_ids.get("association_id") or ""))
    return (
        str(record.get("record_type") or ""),
        str(record.get("entity_family") or ""),
        str(record.get("entity_type") or ""),
        str(source_ids.get("source_detection_id") or ""),
        str(source_ids.get("source_track_id") or ""),
        str(source_ids.get("person_attribute_id") or ""),
        str(source_ids.get("object_attribute_id") or ""),
        str(source_ids.get("association_id") or ""),
        str(source_ids.get("base_search_id") or ""),
        f"{round(as_float(record.get('representative_timestamp'), 0.0), 1):.1f}",
    )


def record_priority(record: dict[str, Any], match_strength: str) -> tuple[int, int, float, int, int]:
    record_type = str(record.get("record_type") or "")
    type_rank = {
        "association_record": 0,
        "person_attribute_record": 1,
        "object_attribute_record": 2,
        "enriched_event_record": 3,
        "base_record": 4,
    }.get(record_type, 9)
    strength_rank = {"strong": 0, "possible": 1, "review": 2}.get(match_strength, 9)
    confidence = -as_float(record.get("confidence"), 0.0)
    evidence = dict(record.get("evidence") or {})
    rich_evidence_rank = 0 if any(str(evidence.get(key) or "").strip() for key in ["crop_path", "subject_crop_path", "object_crop_path"]) else 1
    source_ids = dict(record.get("source_ids") or {})
    missing_id_count = sum(1 for value in source_ids.values() if not str(value or "").strip())
    return (type_rank, strength_rank, confidence, rich_evidence_rank, missing_id_count)


def bucket_for_record(record: dict[str, Any], parsed_query: dict[str, Any], proposed_strength: str) -> str:
    default_strength = str(record.get("match_strength_default") or "review")
    status = str(record.get("status") or "")
    if status == "possible_vehicle_misclassification":
        return "review"
    if default_strength == "review":
        return "review"
    if bool(record.get("needs_review")) and not bool(parsed_query.get("wants_confirmed")):
        return "review"
    if bool(parsed_query.get("wants_review")) and proposed_strength == "strong":
        return "review"
    if default_strength == "possible" and proposed_strength == "strong":
        return "possible"
    return proposed_strength


def classify_record(record: dict[str, Any], parsed_query: dict[str, Any]) -> tuple[str | None, list[str]]:
    keywords = extract_keywords(record)
    searchable_text = record_search_blob(record)
    attributes = dict(record.get("attributes") or {})
    match_reasons: list[str] = []
    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "").lower()
    class_name = str(record.get("class_name") or "").lower()
    safe_class_name = str(record.get("safe_class_name") or "").lower()
    raw_class_name = str(record.get("raw_class_name") or "").lower()
    status = str(record.get("status") or "")

    entity = str(parsed_query.get("entity") or "")
    color = str(parsed_query.get("color") or "")
    colors = [str(item or "").lower() for item in list(parsed_query.get("colors") or []) if str(item or "").strip()]
    plate_token = str(parsed_query.get("plate_token") or "")
    wants_confirmed = bool(parsed_query.get("wants_confirmed"))
    wants_review = bool(parsed_query.get("wants_review"))
    wants_person_clothing = bool(parsed_query.get("wants_person_clothing"))
    wants_vehicle_like = bool(parsed_query.get("wants_vehicle_like"))
    wants_misclassification = bool(parsed_query.get("wants_misclassification"))
    wants_overlapping_vehicle = bool(parsed_query.get("wants_overlapping_vehicle"))
    wants_all = bool(parsed_query.get("wants_all"))
    is_intent_query = wants_vehicle_like or wants_misclassification or wants_overlapping_vehicle

    if plate_token:
        plate_candidates = {
            normalize_plate_token(attributes.get("plate_text") or ""),
            normalize_plate_token(attributes.get("candidate_plate_text") or ""),
            normalize_plate_token(attributes.get("plate_text_normalized") or ""),
        }
        plate_candidates.update(normalize_plate_token(item) for item in keywords)
        if plate_token not in plate_candidates:
            return None, []
        match_reasons.append(f"plate matched {plate_token.upper()}")

    if wants_misclassification:
        if not any(
            term in searchable_text
            for term in [
                "possible_vehicle_misclassification",
                "possible vehicle misclassification",
                "vehicle_like_object",
                "vehicle like object",
                "false_suitcase_possible",
                "false suitcase possible",
            ]
        ):
            return None, []
        match_reasons.append("matched possible vehicle misclassification intent")

    if wants_vehicle_like:
        if not any(
            term in searchable_text
            for term in [
                "vehicle_like_object",
                "vehicle like object",
                "possible_vehicle_misclassification",
                "possible vehicle misclassification",
                "uncertain_object",
                "uncertain object",
            ]
        ) and str(attributes.get("possible_actual_family") or "") != "vehicle":
            return None, []
        match_reasons.append("matched vehicle-like object intent")

    if wants_overlapping_vehicle:
        if not any(
            term in searchable_text
            for term in [
                "object_overlaps_vehicle",
                "object overlaps vehicle",
                "object_overlaps_vehicle_detection",
                "object overlaps vehicle detection",
                "object_vehicle",
                "object vehicle",
            ]
        ):
            return None, []
        match_reasons.append("matched object overlapping vehicle intent")

    if not is_intent_query and entity == "person":
        if entity_family != "person":
            return None, []
        match_reasons.append("entity_family matched person")
    elif not is_intent_query and entity in {"vehicle", "car", "truck", "bus", "motorcycle"}:
        vehicle_values = {entity_type, class_name, safe_class_name, raw_class_name}
        vehicle_values.update(clean_list(dict(record.get("match_facets") or {}).get("vehicle_type")))
        if entity == "vehicle":
            if entity_family != "vehicle":
                return None, []
        elif entity not in vehicle_values:
            return None, []
        match_reasons.append(f"vehicle entity matched {entity}")
    elif not is_intent_query and entity in {"object", "suitcase", "bag", "backpack", "laptop", "phone", "bottle", "uncertain_object"}:
        object_values = {entity_type, class_name, safe_class_name, raw_class_name}
        object_values.update(clean_list(dict(record.get("match_facets") or {}).get("object_type")))
        if entity == "object":
            if entity_family not in {"object", "association"}:
                return None, []
        elif entity not in object_values and entity not in keywords:
            return None, []
        match_reasons.append(f"object entity matched {entity}")

    if wants_confirmed:
        if entity in {"suitcase", "bag", "backpack", "laptop", "phone", "bottle"}:
            if entity_family != "object":
                return None, []
            if bool(record.get("needs_review")):
                return None, []
            if status != "confirmed":
                return None, []
            if str(record.get("match_strength_default") or "") != "strong":
                return None, []
            if str(attributes.get("attribute_status") or "") in {"possible_vehicle_misclassification", "possible_false_positive"}:
                return None, []
            if bool(attributes.get("object_class_needs_review")):
                return None, []
            match_reasons.append("confirmed object constraints satisfied")

    if wants_person_clothing:
        if entity_family != "person":
            return None, []
        person_colors = {
            str(attributes.get("top_clothing_color") or "").lower(),
            str(attributes.get("bottom_clothing_color") or "").lower(),
            str(attributes.get("overall_clothing_color") or "").lower(),
            str(attributes.get("normalized_top_color") or "").lower(),
            str(attributes.get("normalized_bottom_color") or "").lower(),
        }
        color_family = set(clean_list(attributes.get("clothing_color_family")))
        if colors:
            if not any(
                requested_color in person_colors or requested_color in color_family or requested_color in keywords
                for requested_color in colors
            ):
                return None, []
            match_reasons.append(f"person clothing color matched {','.join(colors)}")

    if color and not wants_person_clothing:
        record_colors = {
            str(attributes.get("normalized_color") or "").lower(),
            str(attributes.get("object_color") or "").lower(),
            str(attributes.get("vehicle_color") or "").lower(),
            str(attributes.get("top_clothing_color") or "").lower(),
            str(attributes.get("bottom_clothing_color") or "").lower(),
            str(attributes.get("overall_clothing_color") or "").lower(),
        }
        color_family = set(clean_list(attributes.get("color_family"))) | set(clean_list(attributes.get("clothing_color_family")))
        if not any(
            requested_color in record_colors or requested_color in color_family or requested_color in keywords
            for requested_color in colors or [color]
        ):
            return None, []
        match_reasons.append(f"color matched {','.join(colors or [color])}")

    if wants_review:
        review_like = bool(record.get("needs_review")) or status in {"review", "possible_vehicle_misclassification", "possible_false_positive"} or str(record.get("match_strength_default") or "") == "review"
        if not review_like:
            return None, []
        match_reasons.append("review-state matched")

    if entity == "vehicle" and wants_all and entity_family != "vehicle":
        return None, []
    if entity == "person" and wants_all and entity_family != "person":
        return None, []
    if entity == "object" and wants_all and entity_family not in {"object", "association"}:
        return None, []
    if normalized_query := str(parsed_query.get("normalized_query") or ""):
        if normalized_query == "find all review records":
            if not bool(record.get("needs_review")) and str(record.get("match_strength_default") or "") != "review":
                return None, []
            match_reasons.append("all review records filter matched")

    default_strength = str(record.get("match_strength_default") or "review")
    proposed_strength = default_strength if default_strength in {"strong", "possible", "review"} else "review"
    if (
        is_intent_query
        and (
            bool(record.get("needs_review"))
            or normalize_text(status) == "review"
            or default_strength == "review"
            or class_name == "possible_vehicle_misclassification"
            or normalize_text(str(attributes.get("relationship") or "")) == "possible vehicle misclassification"
            or "possible_vehicle_misclassification" in searchable_text
            or "possible vehicle misclassification" in searchable_text
        )
    ):
        proposed_strength = "review"
    if wants_confirmed and proposed_strength != "strong":
        return None, []
    return bucket_for_record(record, parsed_query, proposed_strength), match_reasons


def build_result_record(record: dict[str, Any], match_strength: str, match_reasons: list[str], matched_terms: list[str]) -> dict[str, Any]:
    return {
        "search_id": str(record.get("search_id") or ""),
        "record_type": str(record.get("record_type") or ""),
        "entity_family": str(record.get("entity_family") or ""),
        "entity_type": str(record.get("entity_type") or ""),
        "class_name": str(record.get("class_name") or ""),
        "start_time": round(as_float(record.get("start_time"), 0.0), 3),
        "end_time": round(as_float(record.get("end_time"), 0.0), 3),
        "representative_timestamp": round(as_float(record.get("representative_timestamp"), 0.0), 3),
        "confidence": round(as_float(record.get("confidence"), 0.0), 3),
        "match_strength": match_strength,
        "needs_review": bool(record.get("needs_review")),
        "review_reason": str(record.get("review_reason") or ""),
        "status": str(record.get("status") or ""),
        "source_ids": dict(record.get("source_ids") or {}),
        "evidence": dict(record.get("evidence") or {}),
        "attributes": dict(record.get("attributes") or {}),
        "relationships": list(record.get("relationships") or []),
        "matched_terms": matched_terms,
        "match_reasons": match_reasons,
    }


def is_overlap_association_preferred_record(record: dict[str, Any]) -> bool:
    attributes = dict(record.get("attributes") or {})
    record_type = str(record.get("record_type") or "")
    entity_family = str(record.get("entity_family") or "")
    relationship = normalize_text(str(attributes.get("relationship") or ""))
    search_blob = record_search_blob(record)
    return (
        record_type == "association_record"
        or entity_family == "association"
        or "possible vehicle misclassification" in relationship
        or "possible_vehicle_misclassification" in search_blob
        or "object_overlaps_vehicle" in search_blob
        or "object overlaps vehicle" in search_blob
    )


def is_overlap_object_fallback_record(record: dict[str, Any]) -> bool:
    record_type = str(record.get("record_type") or "")
    if record_type != "object_attribute_record":
        return False
    status = str(record.get("status") or "")
    return status == "possible_vehicle_misclassification" and bool(record.get("needs_review"))


def build_enriched_search_query_outputs(
    run_dir: Path,
    queries: list[str],
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    index_path = run_dir / "13_enriched_search_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required Step 14 input: {index_path}")
    index_payload = read_json(index_path)
    _report_payload = read_optional_json(run_dir / "13_enriched_search_index_report.json")
    _smoke_payload = read_optional_json(run_dir / "13_enriched_search_smoke_test.json")
    records = list(index_payload.get("records") or [])

    warnings: list[str] = []
    recommendations: list[str] = []
    debug_payload: dict[str, Any] = {"queries": []}
    all_query_results: list[dict[str, Any]] = []
    results_by_strength = {"strong": 0, "possible": 0, "review": 0}
    results_by_record_type: Counter[str] = Counter()
    results_by_entity_family: Counter[str] = Counter()
    queries_with_no_results: list[str] = []
    total_raw_matches = 0
    total_grouped_results = 0

    for query in queries:
        parsed_query = parse_query(query)
        raw_matches: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        intent_terms_in_index = False
        if (
            bool(parsed_query.get("wants_vehicle_like"))
            or bool(parsed_query.get("wants_misclassification"))
            or bool(parsed_query.get("wants_overlapping_vehicle"))
        ):
            intent_terms = []
            if bool(parsed_query.get("wants_vehicle_like")):
                intent_terms.extend(["vehicle_like_object", "vehicle like object", "uncertain_object", "uncertain object"])
            if bool(parsed_query.get("wants_misclassification")):
                intent_terms.extend(["possible_vehicle_misclassification", "possible vehicle misclassification", "false_suitcase_possible", "false suitcase possible"])
            if bool(parsed_query.get("wants_overlapping_vehicle")):
                intent_terms.extend(["object_overlaps_vehicle", "object overlaps vehicle", "object_overlaps_vehicle_detection", "object overlaps vehicle detection", "object_vehicle", "object vehicle"])
            for record in records:
                search_blob = record_search_blob(record)
                if any(term in search_blob for term in intent_terms):
                    intent_terms_in_index = True
                    break
        for record in records:
            match_strength, reasons = classify_record(record, parsed_query)
            if match_strength is None:
                if debug_full:
                    rejected_rows.append(
                        {
                            "search_id": record.get("search_id"),
                            "record_type": record.get("record_type"),
                            "entity_family": record.get("entity_family"),
                            "entity_type": record.get("entity_type"),
                            "reason": "no_match",
                        }
                    )
                continue
            matched_terms = [
                term
                for term in [parsed_query.get("entity"), *(list(parsed_query.get("colors") or [])), parsed_query.get("plate_token")]
                if str(term or "").strip()
            ]
            raw_matches.append(
                {
                    "record": record,
                    "match_strength": match_strength,
                    "match_reasons": reasons,
                    "matched_terms": matched_terms,
                }
            )

        if bool(parsed_query.get("wants_overlapping_vehicle")) and raw_matches:
            preferred_matches = [
                item
                for item in raw_matches
                if is_overlap_association_preferred_record(item["record"])
            ]
            if preferred_matches:
                raw_matches = preferred_matches
            else:
                raw_matches = [
                    item
                    for item in raw_matches
                    if is_overlap_object_fallback_record(item["record"])
                ]

        grouped: dict[tuple[str, ...], dict[str, Any]] = {}
        grouped_debug: list[dict[str, Any]] = []
        for item in raw_matches:
            record = item["record"]
            key = group_key_for_record(record)
            if key not in grouped:
                grouped[key] = item
                grouped_debug.append({"group_key": key, "kept": record.get("search_id"), "reason": "first_match"})
                continue
            current = grouped[key]
            current_priority = record_priority(current["record"], current["match_strength"])
            candidate_priority = record_priority(record, item["match_strength"])
            if candidate_priority < current_priority:
                grouped[key] = item
                grouped_debug.append({"group_key": key, "kept": record.get("search_id"), "dropped": current["record"].get("search_id"), "reason": "higher_priority"})
            else:
                grouped_debug.append({"group_key": key, "kept": current["record"].get("search_id"), "dropped": record.get("search_id"), "reason": "lower_priority"})

        strong_results: list[dict[str, Any]] = []
        possible_results: list[dict[str, Any]] = []
        review_results: list[dict[str, Any]] = []
        for item in grouped.values():
            result_record = build_result_record(
                item["record"],
                item["match_strength"],
                item["match_reasons"],
                item["matched_terms"],
            )
            if item["match_strength"] == "strong":
                strong_results.append(result_record)
            elif item["match_strength"] == "possible":
                possible_results.append(result_record)
            else:
                review_results.append(result_record)
            results_by_strength[item["match_strength"]] += 1
            results_by_record_type[str(result_record["record_type"])] += 1
            results_by_entity_family[str(result_record["entity_family"])] += 1

        for bucket in [strong_results, possible_results, review_results]:
            bucket.sort(
                key=lambda item: (
                    {"strong": 0, "possible": 1, "review": 2}.get(str(item.get("match_strength") or ""), 9),
                    -as_float(item.get("confidence"), 0.0),
                    as_float(item.get("representative_timestamp"), 0.0),
                )
            )

        grouped_total = len(strong_results) + len(possible_results) + len(review_results)
        total_raw_matches += len(raw_matches)
        total_grouped_results += grouped_total
        if grouped_total == 0:
            queries_with_no_results.append(query)
            if intent_terms_in_index:
                warning_text = "Intent terms exist in index but matcher did not return them."
                warnings.append(f"{query}: {warning_text}")

        query_result = {
            "query": query,
            "parsed_query": parsed_query,
            "total_raw_matches": len(raw_matches),
            "total_grouped_results": grouped_total,
            "strong_results": strong_results,
            "possible_results": possible_results,
            "review_results": review_results,
            "no_match_reason": "" if grouped_total else "No enriched records matched the requested query.",
            "notes": [warning_text] if grouped_total == 0 and intent_terms_in_index else [],
        }
        if query == "find confirmed suitcase" and grouped_total == 0:
            query_result["notes"].append("No confirmed suitcase exists; only review/possible suitcase evidence was found.")
        all_query_results.append(query_result)

        debug_entry = {
            "query": query,
            "parsed_query": parsed_query,
            "raw_candidate_matches": [
                {
                    "search_id": item["record"].get("search_id"),
                    "record_type": item["record"].get("record_type"),
                    "entity_family": item["record"].get("entity_family"),
                    "entity_type": item["record"].get("entity_type"),
                    "match_strength": item["match_strength"],
                    "match_reasons": item["match_reasons"],
                }
                for item in raw_matches[: (len(raw_matches) if debug_full else 80)]
            ],
            "grouped_decisions": grouped_debug[: (len(grouped_debug) if debug_full else 80)],
            "rejected_candidates": rejected_rows[: (len(rejected_rows) if debug_full else 0)],
        }
        debug_payload["queries"].append(debug_entry)

    report_payload = {
        "overall_status": "completed",
        "index_records_loaded": len(records),
        "queries_run": len(queries),
        "total_raw_matches": total_raw_matches,
        "total_grouped_results": total_grouped_results,
        "results_by_strength": dict(results_by_strength),
        "results_by_record_type": dict(sorted(results_by_record_type.items())),
        "results_by_entity_family": dict(sorted(results_by_entity_family.items())),
        "queries_with_no_results": queries_with_no_results,
        "confirmed_suitcase_results": sum(
            len(item["strong_results"])
            for item in all_query_results
            if normalize_text(item["query"]) == "find confirmed suitcase"
        ),
        "review_suitcase_results": sum(
            len(item["review_results"])
            for item in all_query_results
            if normalize_text(item["query"]) == "find suitcase"
        ),
        "vehicle_like_object_results": sum(
            item["total_grouped_results"]
            for item in all_query_results
            if normalize_text(item["query"]) == "find vehicle like object"
        ),
        "possible_vehicle_misclassification_results": sum(
            item["total_grouped_results"]
            for item in all_query_results
            if normalize_text(item["query"]) == "find possible vehicle misclassification"
        ),
        "plate_query_results": sum(
            item["total_grouped_results"]
            for item in all_query_results
            if "plate" in normalize_text(item["query"])
        ),
        "warnings": warnings,
        "recommendations": recommendations,
        "created_at": current_timestamp(),
    }
    results_payload = {
        "created_at": current_timestamp(),
        "index_path": str(index_path),
        "queries": all_query_results,
    }
    return {
        "results_payload": results_payload,
        "report_payload": report_payload,
        "debug_payload": debug_payload,
    }


def update_run_manifest_for_enriched_search_query_engine(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "14_enriched_search_query_engine" not in completed_steps:
        completed_steps.append("14_enriched_search_query_engine")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "15_evidence_ranking"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
