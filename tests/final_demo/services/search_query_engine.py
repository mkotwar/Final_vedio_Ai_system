from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


DEFAULT_DEMO_QUERIES = [
    "find white car",
    "find silver car",
    "find all cars",
    "find vehicle with plate HR38AE1442",
    "find all persons",
    "find person wearing red shirt",
    "find suitcase",
    "find all objects",
    "find vehicles needing review",
    "find objects needing review",
]

SINGULAR_NORMALIZATION = {
    "persons": "person",
    "people": "person",
    "humans": "person",
    "objects": "object",
    "items": "object",
    "vehicles": "vehicle",
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "motorcycles": "motorcycle",
    "bikes": "bicycle",
    "suitcases": "suitcase",
    "backpacks": "backpack",
    "bags": "bag",
}

ENTITY_TERMS = {
    "vehicle",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "object",
    "suitcase",
    "backpack",
    "laptop",
    "bag",
    "handbag",
}
COLOR_TERMS = {"white", "silver", "gray", "grey", "black", "brown", "red", "blue", "green", "yellow"}
FILLER_TERMS = {"find", "with", "all", "wearing", "shirt", "top", "pants", "bottom", "need", "needs", "needing", "review", "plate", "unreadable"}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())).strip()


def normalize_plate_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def parse_query(query: str) -> dict[str, Any]:
    normalized = normalize_text(query)
    raw_tokens = [token for token in normalized.split(" ") if token]
    tokens = [SINGULAR_NORMALIZATION.get(token, token) for token in raw_tokens]
    plate_tokens = [
        normalize_plate_token(token)
        for token in tokens
        if re.fullmatch(r"[a-z0-9]{6,}", token)
        and any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    ]
    plate_text = plate_tokens[0] if plate_tokens else ""
    entity = ""
    for candidate in ["car", "truck", "bus", "motorcycle", "bicycle", "person", "object", "vehicle", "suitcase", "backpack", "laptop", "handbag", "bag"]:
        if candidate in tokens:
            entity = candidate
            break
    color = ""
    for candidate in ["white", "silver", "gray", "grey", "black", "brown", "red", "blue", "green", "yellow"]:
        if candidate in tokens:
            color = candidate
            break
    asks_review = (
        "review" in tokens
        or "needing" in tokens
        or "need" in tokens
        or "uncertain" in tokens
        or "unreadable" in tokens
    )
    asks_all = "all" in tokens
    wants_person_clothing = "shirt" in tokens or "top" in tokens or "wearing" in tokens
    wants_object_hint = entity in {"suitcase", "backpack", "laptop", "handbag", "bag"}
    known_terms = set(FILLER_TERMS) | ENTITY_TERMS | COLOR_TERMS | set([plate_text] if plate_text else [])
    unknown_terms = [token for token in tokens if token not in known_terms]
    return {
        "normalized_query": normalized,
        "normalized_tokens": tokens,
        "entity": entity,
        "color": color,
        "plate_text": plate_text,
        "asks_review": asks_review,
        "asks_all": asks_all,
        "wants_person_clothing": wants_person_clothing,
        "wants_object_hint": wants_object_hint,
        "is_broad_all_query": normalized in {"find all", "show all"},
        "unknown_terms": unknown_terms,
    }


def record_sort_weight(record: dict[str, Any]) -> tuple[int, float, int, int]:
    record_type = str(record.get("record_type") or "")
    type_rank = {"event_record": 0, "track_record": 1, "detection_record": 2}.get(record_type, 9)
    confidence = float(record.get("confidence") or 0.0)
    has_best_image = 0 if dict(record.get("evidence") or {}).get("best_image_path") else 1
    has_crop = 0 if dict(record.get("evidence") or {}).get("crop_path") else 1
    return (type_rank, -confidence, has_best_image, has_crop)


def classify_match(record: dict[str, Any], parsed_query: dict[str, Any]) -> tuple[str | None, list[str]]:
    entity = str(parsed_query.get("entity") or "")
    color = str(parsed_query.get("color") or "")
    plate_text = str(parsed_query.get("plate_text") or "")
    asks_review = bool(parsed_query.get("asks_review"))
    wants_person_clothing = bool(parsed_query.get("wants_person_clothing"))
    attributes = dict(record.get("attributes") or {})
    facets = dict(record.get("match_facets") or {})
    reasons: list[str] = []

    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "")
    class_name = str(record.get("class_name") or "")
    safe_class_name = str(record.get("safe_class_name") or "")
    needs_review = bool(record.get("needs_review"))
    review_status = [str(item).lower() for item in list(facets.get("review_status") or [])]
    broad_all_query = bool(parsed_query.get("is_broad_all_query"))
    missing_person_clothing = not any(
        str(attributes.get(field) or "").strip()
        for field in ["clothing_top_color", "upper_clothing_color", "shirt_color", "top_color"]
    )
    object_missing_attributes = str(attributes.get("attribute_status") or "") == "not_extracted" or bool(attributes.get("object_attributes_need_review"))

    if not broad_all_query:
        if entity == "vehicle":
            if entity_family == "vehicle" or entity_type in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}:
                reasons.append("entity_family matched vehicle")
            else:
                return None, []
        if entity == "person":
            if entity_family == "person" or entity_type == "person" or class_name == "person":
                reasons.append("entity_family matched person")
            else:
                return None, []
        if entity == "object":
            if entity_family == "object":
                reasons.append("entity_family matched object")
            else:
                return None, []
    if entity in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        class_facet = [str(item).lower() for item in list(facets.get("class") or [])]
        if entity_type == entity or class_name == entity or safe_class_name == entity or entity in class_facet:
            reasons.append(f"entity_type matched {entity}")
        elif entity_family == "vehicle" and bool(attributes.get("vehicle_subtype_needs_review")) and entity in list(attributes.get("possible_vehicle_classes") or []):
            reasons.append(f"possible vehicle subtype includes {entity}")
            return "review", reasons
        else:
            return None, []
    if entity in {"suitcase", "backpack", "laptop", "handbag", "bag"}:
        object_type = str(attributes.get("object_type") or entity_type or "").lower()
        keywords = set([str(item).lower() for item in list(record.get("search_keywords") or [])])
        object_facet = [str(item).lower() for item in list(facets.get("object_type") or [])]
        if entity == object_type or entity == entity_type or entity in keywords or entity in object_facet:
            reasons.append(f"entity_type matched {entity}")
        else:
            return None, []
    if asks_review:
        if entity == "object":
            if not (needs_review or "needs_review" in review_status or str(attributes.get("attribute_status") or "") == "not_extracted"):
                return None, []
        else:
            if not (needs_review or "needs_review" in review_status):
                return None, []
        reasons.append("review status matched")
    if plate_text:
        plate_candidates = {
            normalize_plate_token(attributes.get("plate_text_normalized") or ""),
            normalize_plate_token(attributes.get("plate_text") or ""),
            normalize_plate_token(attributes.get("candidate_plate_text") or ""),
        }
        plate_candidates.update([normalize_plate_token(item) for item in list(facets.get("plate") or [])])
        plate_candidates.update([normalize_plate_token(item) for item in list(record.get("search_keywords") or [])])
        if plate_text in plate_candidates:
            reasons.append(f"plate_text matched {plate_text.upper()}")
            if str(attributes.get("plate_format_status") or ""):
                reasons.append(f"plate_format_status is {attributes.get('plate_format_status')}")
            if str(attributes.get("plate_ocr_status") or ""):
                reasons.append(f"OCR status is {attributes.get('plate_ocr_status')}")
        else:
            return None, []
    if color:
        normalized_color = str(attributes.get("normalized_color") or "").lower()
        color_family = [str(item).lower() for item in list(attributes.get("color_family") or [])]
        if color in {"gray", "grey"}:
            if normalized_color == "gray":
                reasons.append("normalized_color matched gray")
            elif any(item in color_family for item in ["gray", "grey", "silver"]):
                reasons.append("color_family contains gray/grey/silver")
                return "possible", reasons
            else:
                return None, []
        elif color == "silver":
            if normalized_color == "silver":
                reasons.append("normalized_color matched silver")
            elif "silver" in color_family:
                reasons.append("color_family contains silver")
                return "possible", reasons
            else:
                return None, []
        elif color == "white":
            if normalized_color == "white":
                reasons.append("normalized_color matched white")
            elif "white_possible" in color_family or "light" in color_family:
                reasons.append("color_family contains white_possible/light")
                if bool(attributes.get("vehicle_subtype_needs_review")):
                    reasons.append("returned as review match, not confirmed white")
                    return "review", reasons
                reasons.append("returned as possible match, not confirmed white")
                return "possible", reasons
            else:
                return None, []
        else:
            if normalized_color == color:
                reasons.append(f"normalized_color matched {color}")
            else:
                return None, []
    if wants_person_clothing and entity_family == "person":
        top_color = str(
            attributes.get("clothing_top_color")
            or attributes.get("upper_clothing_color")
            or attributes.get("shirt_color")
            or attributes.get("top_color")
            or ""
        ).lower()
        if color and top_color == color:
            reasons.append(f"top clothing color matched {color}")
        elif missing_person_clothing:
            reasons.append("person record exists")
            reasons.append("person clothing attributes are missing")
            reasons.append("returned as review match, not confirmed red shirt")
            return "review", reasons
        else:
            return None, []
    if entity == "person" and (str(attributes.get("attribute_status") or "") == "not_extracted" or missing_person_clothing):
        reasons.append("person attribute extraction not available")
        return "review", reasons
    if entity == "object" and object_missing_attributes:
        reasons.append("object attribute extraction not available")
        return "review", reasons
    if broad_all_query:
        reasons.append("broad all query returned all records")
    if asks_review:
        reasons.append("returned as strong review-state match")
    return "strong", reasons


def compatible_entity_match(record: dict[str, Any], parsed_query: dict[str, Any]) -> bool:
    entity = str(parsed_query.get("entity") or "")
    entity_family = str(record.get("entity_family") or "")
    entity_type = str(record.get("entity_type") or "")
    class_name = str(record.get("class_name") or "")
    if not entity:
        return True
    if entity == "vehicle":
        return entity_family == "vehicle"
    if entity == "person":
        return entity_family == "person"
    if entity == "object":
        return entity_family == "object"
    if entity in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return entity_type == entity or class_name == entity or entity_family == "vehicle"
    if entity in {"suitcase", "backpack", "laptop", "handbag", "bag"}:
        return entity_type == entity or class_name == entity or entity_family == "object"
    return True


def is_broad_query(parsed_query: dict[str, Any]) -> bool:
    entity = str(parsed_query.get("entity") or "")
    asks_all = bool(parsed_query.get("asks_all"))
    asks_review = bool(parsed_query.get("asks_review"))
    wants_person_clothing = bool(parsed_query.get("wants_person_clothing"))
    plate_text = str(parsed_query.get("plate_text") or "")
    color = str(parsed_query.get("color") or "")
    if plate_text or wants_person_clothing:
        return False
    return bool(entity) and (asks_all or asks_review or not color)


def suppress_detection_fallback_matches(
    raw_matches: list[dict[str, Any]],
    parsed_query: dict[str, Any],
) -> list[dict[str, Any]]:
    if not is_broad_query(parsed_query):
        return raw_matches
    stronger_records = [
        item["record"]
        for item in raw_matches
        if str(item["record"].get("record_type") or "") != "detection_record"
        and compatible_entity_match(item["record"], parsed_query)
    ]
    if not stronger_records:
        return raw_matches

    filtered: list[dict[str, Any]] = []
    for item in raw_matches:
        record = item["record"]
        if str(record.get("record_type") or "") != "detection_record":
            filtered.append(item)
            continue
        record_family = str(record.get("entity_family") or "")
        record_type = str(record.get("entity_type") or "")
        record_time = float(record.get("representative_timestamp") or 0.0)
        suppress = False
        for stronger in stronger_records:
            stronger_family = str(stronger.get("entity_family") or "")
            stronger_type = str(stronger.get("entity_type") or "")
            stronger_start = float(stronger.get("start_time") or stronger.get("representative_timestamp") or 0.0)
            stronger_end = float(stronger.get("end_time") or stronger.get("representative_timestamp") or stronger_start)
            if stronger_family != record_family:
                continue
            if record_family == "vehicle":
                if str(parsed_query.get("entity") or "") not in {"vehicle", ""} and stronger_type != record_type:
                    continue
            elif record_family == "object":
                if str(parsed_query.get("entity") or "") in {"suitcase", "backpack", "laptop", "handbag", "bag"} and stronger_type != record_type:
                    continue
            elif record_family == "person" and stronger_type != record_type:
                continue
            if stronger_start - 1.0 <= record_time <= stronger_end + 1.0:
                suppress = True
                break
        if not suppress:
            filtered.append(item)
    return filtered


def group_key_for_record(record: dict[str, Any]) -> str:
    for key in ["source_event_candidate_id", "attribute_track_id", "source_track_id", "source_detection_id"]:
        value = str(record.get(key) or "")
        if value:
            return f"{key}:{value}"
    return (
        f"{record.get('entity_family')}:"
        f"{record.get('entity_type')}:"
        f"{round(float(record.get('representative_timestamp') or 0.0), 1):.1f}"
    )


def build_grouped_result(
    *,
    group_records: list[dict[str, Any]],
    match_strength: str,
    query: str,
    result_index: int,
    match_reasons: list[str],
) -> dict[str, Any]:
    representative = sorted(group_records, key=record_sort_weight)[0]
    attributes = dict(representative.get("attributes") or {})
    evidence = dict(representative.get("evidence") or {})
    display_class_name = str(representative.get("safe_class_name") or representative.get("entity_type") or representative.get("class_name") or "unknown")
    color = str(attributes.get("normalized_color") or attributes.get("vehicle_color") or attributes.get("clothing_top_color") or "").strip().lower()
    display_label = f"{color} {display_class_name}".strip() if color else display_class_name
    plate_text = str(attributes.get("plate_text") or attributes.get("candidate_plate_text") or "").strip()
    review_reason = str(attributes.get("vehicle_subtype_review_reason") or "")
    normalized_query = normalize_text(query)
    query_is_review = any(term in normalized_query for term in ["review", "unreadable", "uncertain"])
    summary = f"{display_label} matched query '{query}'."
    if match_strength == "possible":
        summary = f"{display_label} may match query '{query}'."
    if match_strength == "review":
        summary = f"{display_label} requires review for query '{query}'."
    if match_strength == "strong" and query_is_review:
        summary = f"{display_label} matched because needs_review is true."
    if plate_text:
        summary = f"{summary} Plate: {plate_text}."
    title = f"{match_strength.capitalize()} {display_label}".strip()
    if query_is_review and match_strength == "strong":
        title = f"{display_class_name.capitalize()} needing review"
    return {
        "result_id": f"result_{result_index:06d}",
        "match_strength": match_strength,
        "entity_family": representative.get("entity_family"),
        "entity_type": representative.get("entity_type"),
        "display_class_name": display_class_name,
        "display_label": display_label,
        "title": title,
        "summary": summary,
        "start_time": representative.get("start_time"),
        "end_time": representative.get("end_time"),
        "representative_timestamp": representative.get("representative_timestamp"),
        "confidence": representative.get("confidence"),
        "needs_review": representative.get("needs_review"),
        "review_reason": review_reason,
        "source_track_id": representative.get("source_track_id"),
        "attribute_track_id": representative.get("attribute_track_id"),
        "source_event_candidate_id": representative.get("source_event_candidate_id"),
        "best_frame_id": evidence.get("best_frame_id"),
        "best_image_path": evidence.get("best_image_path"),
        "crop_path": evidence.get("crop_path"),
        "plate_crop_path": evidence.get("plate_crop_path"),
        "plate_text": plate_text,
        "attributes": attributes,
        "matched_record_ids": [str(item.get("search_id")) for item in group_records],
        "match_reasons": match_reasons,
        "raw_evidence": {
            "record_types": [str(item.get("record_type")) for item in group_records],
            "source_detection_ids": [str(item.get("source_detection_id") or "") for item in group_records if item.get("source_detection_id")],
            "supporting_timestamps": [item.get("representative_timestamp") for item in group_records],
        },
    }


def result_sort_key(result: dict[str, Any], plate_query: bool, asks_review: bool) -> tuple[int, int, int, float, float]:
    strength_rank = {"strong": 0, "possible": 1, "review": 2}.get(str(result.get("match_strength") or ""), 9)
    exact_plate = 0 if (plate_query and normalize_plate_token(result.get("plate_text") or "") == plate_query) else 1
    record_rank = 9
    for record_type in list(dict(result.get("raw_evidence") or {}).get("record_types") or []):
        record_rank = min(record_rank, {"event_record": 0, "track_record": 1, "detection_record": 2}.get(str(record_type), 9))
    review_rank = 0 if asks_review else (1 if result.get("needs_review") else 0)
    return (
        strength_rank,
        exact_plate,
        record_rank,
        review_rank,
        float(result.get("representative_timestamp") or 0.0),
    )


def update_run_manifest_for_search_query_engine(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "09_search_query_engine" not in completed_steps:
        completed_steps.append("09_search_query_engine")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "10_person_attribute_extraction"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_search_query_outputs(run_dir: Path, queries: list[str], *, debug_full: bool = False) -> dict[str, Any]:
    index_path = run_dir / "08_attribute_search_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required Step 9 input: {index_path}")

    index_payload = read_json(index_path)
    smoke_payload = read_optional_json(run_dir / "08_attribute_search_smoke_test.json")
    records = list(index_payload.get("records") or [])
    warnings: list[str] = []
    recommendations: list[str] = []
    results_by_strength = {"strong": 0, "possible": 0, "review": 0}
    results_by_query: dict[str, int] = {}
    debug_queries: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []

    if not any(str(dict(record.get("attributes") or {}).get("clothing_top_color") or "") for record in records):
        warnings.append("No person clothing attributes exist; person clothing search is review-only.")

    total_raw_matches = 0
    total_grouped_results = 0
    result_index = 1
    index_records_with_persons = sum(1 for record in records if str(record.get("entity_family") or "") == "person")
    index_records_with_objects = sum(1 for record in records if str(record.get("entity_family") or "") == "object")
    regression_checks = {
        "plate_HR38AE1442_single_match": False,
        "all_persons_nonzero_if_index_has_persons": True,
        "all_objects_nonzero_if_index_has_objects": True,
        "all_cars_not_equal_total_index_records": True,
        "person_red_shirt_fallback_returns_person_review_results": False,
        "all_persons_returns_8_review_results": False,
        "all_objects_returns_3_review_results": False,
    }
    for query in queries:
        parsed_query = parse_query(query)
        raw_matches: list[dict[str, Any]] = []
        debug_rows: list[dict[str, Any]] = []
        for record in records:
            match_strength, reasons = classify_match(record, parsed_query)
            if match_strength is not None or debug_full:
                debug_rows.append(
                    {
                        "search_id": record.get("search_id"),
                        "entity_family": record.get("entity_family"),
                        "entity_type": record.get("entity_type"),
                        "match_strength": match_strength,
                        "reasons": reasons,
                    }
                )
            if match_strength is None:
                continue
            raw_matches.append(
                {
                    "record": record,
                    "match_strength": match_strength,
                    "reasons": reasons,
                }
            )

        raw_matches = suppress_detection_fallback_matches(raw_matches, parsed_query)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        grouped_reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
        for item in raw_matches:
            key = (group_key_for_record(item["record"]), item["match_strength"])
            grouped[key].append(item["record"])
            for reason in item["reasons"]:
                if reason not in grouped_reasons[key]:
                    grouped_reasons[key].append(reason)

        strong_results: list[dict[str, Any]] = []
        possible_results: list[dict[str, Any]] = []
        review_results: list[dict[str, Any]] = []
        for (group_key, strength), group_records in grouped.items():
            result = build_grouped_result(
                group_records=group_records,
                match_strength=strength,
                query=query,
                result_index=result_index,
                match_reasons=grouped_reasons[(group_key, strength)],
            )
            result_index += 1
            if strength == "strong":
                strong_results.append(result)
            elif strength == "possible":
                possible_results.append(result)
            else:
                review_results.append(result)
            results_by_strength[strength] += 1

        plate_query = str(parsed_query.get("plate_text") or "")
        asks_review = bool(parsed_query.get("asks_review"))
        strong_results.sort(key=lambda item: result_sort_key(item, plate_query, asks_review))
        possible_results.sort(key=lambda item: result_sort_key(item, plate_query, asks_review))
        review_results.sort(key=lambda item: result_sort_key(item, plate_query, asks_review))
        total_matches = len(raw_matches)
        grouped_total = len(strong_results) + len(possible_results) + len(review_results)
        total_raw_matches += total_matches
        total_grouped_results += grouped_total
        results_by_query[query] = grouped_total
        no_match_reason = ""
        notes: list[str] = []
        if grouped_total == 0:
            no_match_reason = "No records matched the requested query terms."
        if parsed_query.get("unknown_terms"):
            notes.append(f"Unsupported terms: {parsed_query['unknown_terms']}")
        if parsed_query.get("wants_person_clothing") and not any(result for result in strong_results + possible_results + review_results):
            notes.append("person attribute extraction not available")
        if len([item for item in raw_matches if str(item['record'].get('record_type') or '') == 'detection_record']) >= 5:
            recommendations.append("Many detection fallback records were returned; add track/person/object attribute extraction.")
        if grouped_total > 25 and is_broad_query(parsed_query):
            warnings.append("Broad query returned many detection-level records; consider stricter grouping or UI pagination.")
        normalized_query = normalize_text(query)
        if normalized_query == "find vehicle with plate hr38ae1442":
            regression_checks["plate_HR38AE1442_single_match"] = grouped_total == 1 and len(strong_results) == 1
        if normalized_query in {"find all persons", "find all person"} and index_records_with_persons > 0:
            regression_checks["all_persons_nonzero_if_index_has_persons"] = grouped_total > 0
            regression_checks["all_persons_returns_8_review_results"] = grouped_total == 8 and len(review_results) == 8
        if normalized_query in {"find all objects", "find all object"} and index_records_with_objects > 0:
            regression_checks["all_objects_nonzero_if_index_has_objects"] = grouped_total > 0
            regression_checks["all_objects_returns_3_review_results"] = grouped_total == 3 and len(review_results) == 3
        if normalized_query in {"find all cars", "find all car"}:
            regression_checks["all_cars_not_equal_total_index_records"] = grouped_total < len(records)
        if normalized_query == "find person wearing red shirt":
            regression_checks["person_red_shirt_fallback_returns_person_review_results"] = grouped_total == 8 and len(review_results) == 8 and len(strong_results) == 0
        query_results.append(
            {
                "query": query,
                "parsed_query": parsed_query,
                "total_raw_matches": total_matches,
                "total_grouped_results": grouped_total,
                "strong_results": strong_results,
                "possible_results": possible_results,
                "review_results": review_results,
                "no_match_reason": no_match_reason,
                "notes": notes,
            }
        )
        debug_queries.append(
            {
                "query": query,
                "parsed_query": parsed_query,
                "matches": debug_rows,
            }
        )

    results_payload = {
        "created_at": current_timestamp(),
        "source": {"search_index": "08_attribute_search_index.json"},
        "queries": query_results,
    }
    report_payload = {
        "created_at": current_timestamp(),
        "overall_status": "completed",
        "index_records_loaded": len(records),
        "queries_run": len(queries),
        "total_raw_matches": total_raw_matches,
        "total_grouped_results": total_grouped_results,
        "results_by_query": results_by_query,
        "results_by_strength": results_by_strength,
        "regression_checks": regression_checks,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recommendations)),
    }
    debug_payload = {
        "created_at": current_timestamp(),
        "source": {"search_index": "08_attribute_search_index.json", "smoke_test": "08_attribute_search_smoke_test.json" if smoke_payload else None},
        "queries": debug_queries,
    }
    return {
        "results_payload": results_payload,
        "report_payload": report_payload,
        "debug_payload": debug_payload,
    }
