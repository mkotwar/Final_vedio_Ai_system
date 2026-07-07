from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from stage_checks import read_json, write_json
from step_08_query_search_validation import (
    INVALID_QUERY_TOKENS,
    normalize_plate,
    normalize_text,
    parse_time_query as _parse_time_query,
    search_vehicle_index,
)
from step_09_search_result_packaging import _build_card, _build_flat_card_row, write_json_any


ALLOWED_SEARCH_MODES = {
    "auto",
    "exact_plate",
    "color_class",
    "class_only",
    "color_only",
    "timestamp",
    "weak_ocr",
    "combined",
}
COLOR_TERMS = {"black", "white", "grey", "gray", "red", "blue", "yellow", "silver", "green", "brown"}
CLASS_TERMS = {"car", "motorcycle", "truck", "bus", "vehicle", "van", "auto", "bike"}
DEFAULT_DEMO_QUERIES = [
    "DL12CL4316",
    "DL4SAE0084",
    "HR47F3216",
    "white car",
    "red truck",
    "yellow bus",
    "black motorcycle 00:37",
    "vehicle at 03:11",
    "CITYDL1FT",
    "VICTORY",
]


def parse_timestamp_query(query: str) -> tuple[float | None, str | None]:
    """Parse supported timestamp query shapes."""

    return _parse_time_query(query)


def _is_plate_like(query: str) -> bool:
    """Return whether a query looks like a compact licence plate."""

    normalized_text = normalize_text(query)
    if " " in normalized_text or ":" in normalized_text:
        return False
    normalized = normalize_plate(query)
    return len(normalized) >= 6 and any(char.isalpha() for char in normalized) and any(char.isdigit() for char in normalized)


def detect_query_mode(query: str, requested_mode: str, allow_weak_ocr: bool) -> str:
    """Resolve the effective search mode for a query."""

    if requested_mode != "auto":
        return requested_mode

    normalized = normalize_text(query)
    tokens = normalized.split()
    color_tokens = [token for token in tokens if token in COLOR_TERMS]
    class_tokens = [token for token in tokens if token in CLASS_TERMS]
    has_timestamp = parse_timestamp_query(query)[0] is not None

    if _is_plate_like(query):
        return "exact_plate"
    if has_timestamp and (color_tokens or class_tokens):
        return "combined"
    if has_timestamp and not color_tokens and not class_tokens:
        return "timestamp"
    if color_tokens and class_tokens and len(tokens) <= 3:
        return "color_class"
    if class_tokens and len(tokens) == 1:
        return "class_only"
    if color_tokens and len(tokens) == 1:
        return "color_only"
    if allow_weak_ocr and _is_plate_like(query):
        return "weak_ocr"
    return "combined"


def _build_live_card(
    *,
    run_dir: Path,
    record: dict[str, Any],
    match: dict[str, Any],
    rank: int,
    schema_version: str,
    save_debug: bool,
    require_image_paths: bool,
) -> dict[str, Any]:
    """Build a Step 10 live result card using the Step 09 schema."""

    card = _build_card(
        card_id=f"live_result_card_{rank:06d}",
        run_dir=run_dir,
        record=record,
        match={**match, "rank": rank},
        schema_version=schema_version,
        include_weak_ocr=True,
        include_invalid_debug_fields=False,
        include_debug_paths=save_debug,
        validate_path_strings=True,
    )
    if "weak_possible_ocr" in list(match.get("match_reason", [])) and not card["license_plate"]["has_verified_plate"]:
        card["quality"]["result_badge"] = "weak_ocr"
        card["ui"]["show_weak_ocr_badge"] = True
        card["subtitle"] = f"{card['time']['display_time']} • weak OCR only • {card['quality']['confidence_badge']} confidence"
    card["debug"]["runner"] = "step_10_search_demo_runner"
    card["debug"]["source_index_file"] = "07_vehicle_search_index.json"
    if not save_debug:
        card["debug"] = {
            "source_index_file": "07_vehicle_search_index.json",
            "runner": "step_10_search_demo_runner",
        }
    if require_image_paths and (not card["media"]["crop_available"] or not card["media"]["full_frame_available"]):
        card["debug"]["image_path_filter_excluded"] = True
    return card


def _flatten_live_card(card: dict[str, Any], resolved_mode: str) -> dict[str, Any]:
    """Create flat output row for a returned live card."""

    flat = _build_flat_card_row(card)
    return {
        "query": card["query"],
        "resolved_mode": resolved_mode,
        "rank": flat["rank"],
        "track_id": flat["track_id"],
        "title": flat["title"],
        "vehicle_class": flat["vehicle_class"],
        "vehicle_color": flat["vehicle_color"],
        "verified_license_plate": flat["verified_license_plate"],
        "timestamp_text": flat["best_timestamp_text"],
        "crop_path": flat["crop_path"],
        "full_frame_path": flat["full_frame_path"],
        "score": card["score"],
        "confidence_badge": flat["confidence_badge"],
        "result_badge": flat["result_badge"],
    }


def _filter_records(records: list[dict[str, Any]], include_fallback: bool) -> list[dict[str, Any]]:
    """Optionally remove fallback records from live search."""

    if include_fallback:
        return records
    return [record for record in records if str(record.get("selection_group", "")) != "fallback"]


def run_search(
    records: list[dict[str, Any]],
    query: str,
    mode: str,
    top_k: int,
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], bool]:
    """Run deterministic search and return resolved mode, matches, and weak OCR fallback flag."""

    resolved_mode = detect_query_mode(query, mode, bool(config["allow_weak_ocr"]))
    fallback_to_weak_ocr = False
    matches = search_vehicle_index(
        records,
        query,
        mode=resolved_mode,
        top_k=top_k,
        allow_weak_ocr_search=bool(config["allow_weak_ocr"]),
        time_tolerance_seconds=float(config["time_tolerance_seconds"]),
    )
    if mode == "auto" and resolved_mode == "exact_plate" and not matches and bool(config["allow_weak_ocr"]):
        weak_matches = search_vehicle_index(
            records,
            query,
            mode="weak_ocr",
            top_k=top_k,
            allow_weak_ocr_search=True,
            time_tolerance_seconds=float(config["time_tolerance_seconds"]),
        )
        if weak_matches:
            fallback_to_weak_ocr = True
            resolved_mode = "weak_ocr"
            matches = weak_matches
    return resolved_mode, matches, fallback_to_weak_ocr


def _build_single_response(
    *,
    run_dir: Path,
    query: str,
    requested_mode: str,
    records: list[dict[str, Any]],
    record_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    schema_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Build one API-style query response."""

    normalized = normalize_text(query)
    blocked_terms = set(INVALID_QUERY_TOKENS) | {"unknown", "null", "none"}
    if normalized in blocked_terms or normalize_plate(query).lower() in blocked_terms:
        response = {
            "status": "success",
            "schema_version": schema_version,
            "run_dir": str(run_dir),
            "query": query,
            "requested_mode": requested_mode,
            "resolved_mode": requested_mode if requested_mode != "auto" else "auto",
            "query_status": "blocked_invalid_ocr",
            "total_matches": 0,
            "cards_returned": 0,
            "top_k": int(config["top_k"]),
            "fallback_to_weak_ocr": False,
            "message": "This query matches invalid OCR text and is blocked from normal search.",
            "cards": [],
        }
        return response, [], {
            "timestamp_generated": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "mode": response["resolved_mode"],
            "total_matches": 0,
            "cards_returned": 0,
            "top_result_track_id": None,
            "top_result_plate": None,
            "top_result_timestamp": None,
        }

    resolved_mode, matches, fallback_to_weak_ocr = run_search(records, query, requested_mode, int(config["top_k"]), config)
    cards: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        record = record_by_id.get(str(match.get("search_record_id", "")))
        if record is None:
            continue
        card = _build_live_card(
            run_dir=run_dir,
            record=record,
            match=match,
            rank=rank,
            schema_version=schema_version,
            save_debug=bool(config["save_debug"]),
            require_image_paths=bool(config["require_image_paths"]),
        )
        if bool(config["require_image_paths"]) and (not card["media"]["crop_available"] or not card["media"]["full_frame_available"]):
            continue
        cards.append(card)
        flat_rows.append(_flatten_live_card(card, resolved_mode))

    query_status = "matched" if cards else "no_results"
    message = f"Found {len(cards)} result." if len(cards) == 1 else f"Found {len(cards)} results."
    if not cards:
        message = "No matching vehicle found."
    response = {
        "status": "success",
        "schema_version": schema_version,
        "run_dir": str(run_dir),
        "query": query,
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "query_status": query_status,
        "total_matches": len(matches),
        "cards_returned": len(cards),
        "top_k": int(config["top_k"]),
        "fallback_to_weak_ocr": fallback_to_weak_ocr,
        "message": message,
        "cards": cards,
    }
    top_card = cards[0] if cards else None
    log_entry = {
        "timestamp_generated": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "mode": resolved_mode,
        "total_matches": len(matches),
        "cards_returned": len(cards),
        "top_result_track_id": top_card["track_id"] if top_card else None,
        "top_result_plate": top_card["license_plate"]["verified"] if top_card else None,
        "top_result_timestamp": top_card["time"]["best_timestamp_text"] if top_card else None,
    }
    return response, flat_rows, log_entry


def run_search_demo(
    *,
    run_dir: Path,
    query_inputs: list[str],
    search_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run the Step 10 live search demo and write API-style outputs."""

    index_payload = read_json(run_dir / "07_vehicle_search_index.json")
    step09_schema_path = run_dir / "09_search_result_card_schema.json"
    schema_payload = read_json(step09_schema_path) if step09_schema_path.exists() else {"schema_version": "v1"}
    records = _filter_records(list(index_payload.get("records", [])), bool(search_config["include_fallback"]))
    if not records:
        raise FileNotFoundError("Step 07 search index is missing records, so Step 10 cannot run search.")

    record_by_id = {str(record.get("search_record_id", "")): record for record in records}
    schema_version = str(schema_payload.get("schema_version", "v1") or "v1")

    query_list = [query for query in query_inputs if str(query or "").strip()]
    if not query_list:
        query_list = list(DEFAULT_DEMO_QUERIES)

    responses: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []
    query_log: list[dict[str, Any]] = []
    for query in query_list:
        response, flat_rows, log_entry = _build_single_response(
            run_dir=run_dir,
            query=query,
            requested_mode=str(search_config["mode"]),
            records=records,
            record_by_id=record_by_id,
            config=search_config,
            schema_version=schema_version,
        )
        responses.append(response)
        flat_results.extend(flat_rows)
        query_log.append(log_entry)

    if len(query_list) == 1:
        response_payload = responses[0]
    else:
        response_payload = {
            "status": "success",
            "schema_version": schema_version,
            "run_dir": str(run_dir),
            "query_count": len(query_list),
            "responses": responses,
        }

    total_cards_returned = len(flat_results)
    queries_with_results = sum(1 for item in responses if item["cards_returned"] > 0)
    queries_without_results = sum(1 for item in responses if item["cards_returned"] == 0 and item["query_status"] == "no_results")
    queries_blocked_invalid_ocr = sum(1 for item in responses if item["query_status"] == "blocked_invalid_ocr")
    cards_with_crop = sum(1 for item in flat_results if item.get("crop_path"))
    cards_with_full_frame = sum(1 for item in flat_results if item.get("full_frame_path"))

    report_payload = {
        "status": "success",
        "source_index_file": "07_vehicle_search_index.json",
        "source_schema_file": "09_search_result_card_schema.json" if step09_schema_path.exists() else "built_in_card_schema_v1",
        "records_loaded": len(records),
        "queries_run": len(query_list),
        "total_cards_returned": total_cards_returned,
        "queries_with_results": queries_with_results,
        "queries_without_results": queries_without_results,
        "queries_blocked_invalid_ocr": queries_blocked_invalid_ocr,
        "path_availability": {
            "cards_with_crop": cards_with_crop,
            "cards_missing_crop": total_cards_returned - cards_with_crop,
            "cards_with_full_frame": cards_with_full_frame,
            "cards_missing_full_frame": total_cards_returned - cards_with_full_frame,
        },
        "example_queries": query_list[:4],
        "recommendation": "Step 10 is ready for Streamlit/API integration.",
    }

    write_json(run_dir / "10_search_demo_response.json", response_payload)
    write_json_any(run_dir / "10_search_demo_results_flat.json", flat_results)
    write_json(run_dir / "10_search_demo_report.json", report_payload)
    write_json_any(run_dir / "10_search_demo_query_log.json", query_log)
    return response_payload, flat_results, report_payload, query_log
