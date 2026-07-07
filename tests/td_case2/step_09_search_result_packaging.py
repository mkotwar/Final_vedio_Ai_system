from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stage_checks import read_json, write_json


INVALID_DISPLAY_TOKENS = {
    "victory",
    "unanswerable",
    "answerable",
    "unknown",
    "null",
    "nil",
    "stop",
    "dev",
    "lancerail",
    "airambulance",
    "lancerailairambulance",
}


def write_json_any(output_path: Path, payload: Any) -> Path:
    """Write any JSON-serializable payload with stable formatting."""

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _normalize_relative_path(path_value: str | None) -> str | None:
    """Keep run-relative strings stable for UI use."""

    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized or None


def _path_available(run_dir: Path, path_value: str | None, validate_path_strings: bool) -> bool:
    """Return whether a run-relative media path is available."""

    normalized = _normalize_relative_path(path_value)
    if not normalized:
        return False
    if not validate_path_strings:
        return True
    return (run_dir / normalized).exists()


def _title_case_text(text: str) -> str:
    """Convert a small label into UI title case."""

    normalized = str(text or "").strip().replace("_", " ")
    return " ".join(word.capitalize() for word in normalized.split())


def _confidence_badge(search_confidence: float) -> str:
    """Map numeric search confidence into a UI badge."""

    if search_confidence >= 0.85:
        return "high"
    if search_confidence >= 0.55:
        return "medium"
    return "low"


def _result_badge(record: dict[str, Any], possible_candidates: list[dict[str, Any]]) -> str:
    """Choose the main UI result badge for a card."""

    if bool(record.get("verified_license_plate_valid")):
        return "verified_plate"
    if possible_candidates:
        return "weak_ocr"
    if str(record.get("vehicle_color", "unknown")) != "unknown":
        return "color_class"
    return "basic_track"


def _subtitle_status(record: dict[str, Any], result_badge: str) -> str:
    """Generate the middle status phrase for the card subtitle."""

    if result_badge == "verified_plate":
        return "verified plate"
    if result_badge == "weak_ocr":
        return "weak OCR only"
    if str(record.get("selection_group", "")) == "fallback":
        return "fallback track"
    if str(record.get("vehicle_color", "unknown")) != "unknown" or str(record.get("vehicle_class", "")):
        return "color/class match"
    return "basic track"


def _clean_possible_candidates(record: dict[str, Any], include_weak_ocr: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Return weak OCR candidates that are safe to show in the UI."""

    if not include_weak_ocr:
        return [], []

    possible_candidates: list[dict[str, Any]] = []
    weak_ocr_text: list[str] = []
    for candidate in list(record.get("possible_license_plate_candidates", [])):
        text_value = str(candidate.get("text", "") or "").strip()
        if not text_value:
            continue
        lower_value = text_value.lower()
        if lower_value in INVALID_DISPLAY_TOKENS:
            continue
        possible_candidates.append(
            {
                "text": text_value,
                "source": str(candidate.get("source", "") or "unknown"),
                "reason": str(candidate.get("reason", "") or "possible_ocr"),
                "plate_format_confidence": str(candidate.get("plate_format_confidence", "") or "unknown"),
                "label": "possible / weak OCR",
            }
        )
        weak_ocr_text.append(text_value)
    return possible_candidates, weak_ocr_text


def _display_title(record: dict[str, Any]) -> str:
    """Build the UI card title."""

    vehicle_color = str(record.get("vehicle_color", "unknown") or "unknown")
    vehicle_class = str(record.get("vehicle_class", "vehicle") or "vehicle")
    verified_plate = str(record.get("verified_license_plate", "not_visible") or "not_visible")

    color_part = "" if vehicle_color == "unknown" else _title_case_text(vehicle_color)
    class_part = _title_case_text(vehicle_class)

    if bool(record.get("verified_license_plate_valid")):
        if color_part:
            return f"{color_part} {class_part} - {verified_plate}"
        return f"{class_part} - {verified_plate}"
    if color_part:
        return f"{color_part} {class_part}"
    return class_part


def _description_text(
    record: dict[str, Any],
    possible_candidates: list[dict[str, Any]],
) -> str:
    """Build a safe human-readable description."""

    vehicle_color = str(record.get("vehicle_color", "unknown") or "unknown")
    vehicle_class = str(record.get("vehicle_class", "vehicle") or "vehicle")
    display_vehicle = f"{vehicle_color} {vehicle_class}".strip() if vehicle_color != "unknown" else vehicle_class
    display_vehicle = display_vehicle.lower()
    timestamp_text = str(record.get("best_timestamp_text", "") or "")
    verified_plate = str(record.get("verified_license_plate", "not_visible") or "not_visible")

    if bool(record.get("verified_license_plate_valid")):
        return f"{_title_case_text(display_vehicle)} detected at {timestamp_text} with verified licence plate {verified_plate}."
    if possible_candidates:
        weak_text = possible_candidates[0]["text"]
        return (
            f"{_title_case_text(display_vehicle)} detected at {timestamp_text}. OCR text {weak_text} is available as "
            "weak/possible text, not a verified licence plate."
        )
    return f"{_title_case_text(display_vehicle)} detected at {timestamp_text}. No verified licence plate is available."


def _query_type(query_item: dict[str, Any]) -> str:
    """Infer a stable query type label from the Step 08 query item."""

    mode = str(query_item.get("mode", "auto") or "auto")
    query = str(query_item.get("query", "") or "")
    if mode in {"exact_plate", "color_class", "timestamp", "combined", "weak_ocr"}:
        return mode
    if query == "step07_index_sample":
        return "fallback_sample"
    return "auto"


def _build_card(
    *,
    card_id: str,
    run_dir: Path,
    record: dict[str, Any],
    match: dict[str, Any],
    schema_version: str,
    include_weak_ocr: bool,
    include_invalid_debug_fields: bool,
    include_debug_paths: bool,
    validate_path_strings: bool,
) -> dict[str, Any]:
    """Build one UI-ready result card from a Step 07 record and Step 08 match."""

    possible_candidates, weak_ocr_text = _clean_possible_candidates(record, include_weak_ocr)
    search_confidence = float(record.get("search_confidence", 0.0) or 0.0)
    confidence_badge = _confidence_badge(search_confidence)
    result_badge = _result_badge(record, possible_candidates)
    subtitle_status = _subtitle_status(record, result_badge)
    crop_path = _normalize_relative_path(str(record.get("best_crop_path", "") or ""))
    full_frame_path = _normalize_relative_path(record.get("best_full_frame_path"))
    plate_crop_path = _normalize_relative_path(record.get("best_plate_crop_path"))
    contact_sheet_path = _normalize_relative_path(record.get("contact_sheet_path"))
    debug_image_path = _normalize_relative_path(record.get("debug_image_path"))

    media = {
        "crop_path": crop_path,
        "full_frame_path": full_frame_path,
        "plate_crop_path": plate_crop_path,
        "contact_sheet_path": contact_sheet_path,
        "debug_image_path": debug_image_path if include_debug_paths else None,
        "crop_available": _path_available(run_dir, crop_path, validate_path_strings),
        "full_frame_available": _path_available(run_dir, full_frame_path, validate_path_strings),
        "plate_crop_available": _path_available(run_dir, plate_crop_path, validate_path_strings),
        "contact_sheet_available": _path_available(run_dir, contact_sheet_path, validate_path_strings),
    }
    if not include_debug_paths:
        media.pop("debug_image_path")

    title = _display_title(record)
    subtitle = (
        f"{record.get('best_timestamp_text', '')} • {subtitle_status} • {confidence_badge} confidence"
    )
    card = {
        "card_id": card_id,
        "result_type": "vehicle_search_result",
        "schema_version": schema_version,
        "query": str(match.get("query", "") or ""),
        "query_mode": str(match.get("mode", "auto") or "auto"),
        "rank": int(match.get("rank", 0) or 0),
        "score": round(float(match.get("score", 0.0) or 0.0), 6),
        "match_reason": list(match.get("match_reason", [])),
        "track_id": str(record.get("track_id", "") or ""),
        "search_record_id": str(record.get("search_record_id", "") or ""),
        "title": title,
        "subtitle": subtitle,
        "description": _description_text(record, possible_candidates),
        "vehicle": {
            "class": str(record.get("vehicle_class", "vehicle") or "vehicle"),
            "color": str(record.get("vehicle_color", "unknown") or "unknown"),
            "color_source": str(record.get("vehicle_color_source", "unknown") or "unknown"),
        },
        "license_plate": {
            "verified": (
                str(record.get("verified_license_plate", "not_visible") or "not_visible")
                if bool(record.get("verified_license_plate_valid"))
                else "not_visible"
            ),
            "has_verified_plate": bool(record.get("verified_license_plate_valid")),
            "source": (
                str(record.get("verified_license_plate_source", "none") or "none")
                if bool(record.get("verified_license_plate_valid"))
                else "none"
            ),
            "confidence_level": (
                str(record.get("verified_license_plate_confidence_level", "none") or "none")
                if bool(record.get("verified_license_plate_valid"))
                else "none"
            ),
            "exact_plate_search_enabled": bool(record.get("verified_license_plate_valid")),
            "possible_candidates": possible_candidates,
            "weak_ocr_text": weak_ocr_text,
        },
        "time": {
            "best_timestamp_seconds": float(record.get("best_timestamp_seconds", 0.0) or 0.0),
            "best_timestamp_text": str(record.get("best_timestamp_text", "") or ""),
            "start_timestamp_seconds": float(record.get("start_timestamp_seconds", 0.0) or 0.0),
            "end_timestamp_seconds": float(record.get("end_timestamp_seconds", 0.0) or 0.0),
            "display_time": str(record.get("best_timestamp_text", "") or ""),
        },
        "media": media,
        "quality": {
            "selection_group": str(record.get("selection_group", "") or ""),
            "quality_label": str(record.get("quality_label", "") or ""),
            "search_confidence": search_confidence,
            "confidence_badge": confidence_badge,
            "result_badge": result_badge,
        },
        "ui": {
            "primary_image": "full_frame" if media["full_frame_available"] else "crop",
            "secondary_image": "crop",
            "show_verified_plate_badge": bool(record.get("verified_license_plate_valid")),
            "show_weak_ocr_badge": bool(possible_candidates) and not bool(record.get("verified_license_plate_valid")),
            "show_fallback_badge": str(record.get("selection_group", "")) == "fallback",
            "display_order": int(match.get("rank", 0) or 0),
        },
        "debug": {
            "source_index_file": "07_vehicle_search_index.json",
            "source_match_file": "08_query_validation_matches.json",
        },
    }

    if include_invalid_debug_fields:
        card["debug"]["invalid_ocr_text"] = list(record.get("invalid_ocr_text", []))
    if include_debug_paths:
        card["debug"]["best_crop_path"] = crop_path
        card["debug"]["best_full_frame_path"] = full_frame_path
        card["debug"]["best_plate_crop_path"] = plate_crop_path
        card["debug"]["contact_sheet_path"] = contact_sheet_path
        if debug_image_path is not None:
            card["debug"]["debug_image_path"] = debug_image_path
    return card


def _build_flat_card_row(card: dict[str, Any]) -> dict[str, Any]:
    """Create a flat row for API/table-friendly export."""

    return {
        "card_id": card["card_id"],
        "query": card["query"],
        "mode": card["query_mode"],
        "rank": card["rank"],
        "track_id": card["track_id"],
        "vehicle_class": card["vehicle"]["class"],
        "vehicle_color": card["vehicle"]["color"],
        "verified_license_plate": card["license_plate"]["verified"],
        "has_verified_plate": card["license_plate"]["has_verified_plate"],
        "best_timestamp_text": card["time"]["best_timestamp_text"],
        "title": card["title"],
        "subtitle": card["subtitle"],
        "crop_path": card["media"]["crop_path"],
        "full_frame_path": card["media"]["full_frame_path"],
        "contact_sheet_path": card["media"]["contact_sheet_path"],
        "search_confidence": card["quality"]["search_confidence"],
        "confidence_badge": card["quality"]["confidence_badge"],
        "result_badge": card["quality"]["result_badge"],
    }


def _validate_cards(cards: list[dict[str, Any]]) -> list[str]:
    """Run the required packaging validation checks."""

    validation_errors: list[str] = []
    if not any(card["quality"]["result_badge"] == "verified_plate" for card in cards):
        validation_errors.append("At least 1 exact plate card is required.")
    if not any(card["quality"]["result_badge"] == "color_class" for card in cards):
        validation_errors.append("At least 1 color/class card is required.")
    if not any(card["media"]["full_frame_available"] for card in cards):
        validation_errors.append("At least 1 card with full_frame_path is required.")

    for card in cards:
        for required_field in ("card_id", "query", "track_id", "title", "media", "quality"):
            if required_field not in card:
                validation_errors.append(f"Missing required field {required_field} in {card.get('card_id', 'unknown')}.")
                continue
            field_value = card[required_field]
            if field_value is None or field_value == "":
                validation_errors.append(f"Missing required field {required_field} in {card.get('card_id', 'unknown')}.")
        display_blob = " ".join([card["title"], card["subtitle"], card["description"]]).lower()
        for token in INVALID_DISPLAY_TOKENS:
            if token in display_blob:
                validation_errors.append(f"Invalid OCR token {token!r} appeared in display text for {card['card_id']}.")
                break
        if card["quality"]["result_badge"] == "weak_ocr" and "weak OCR" not in card["subtitle"]:
            validation_errors.append(f"Weak OCR card is not labelled clearly in subtitle for {card['card_id']}.")
        if card["quality"]["result_badge"] == "weak_ocr" and card["license_plate"]["has_verified_plate"]:
            validation_errors.append(f"Weak OCR card incorrectly marked as verified for {card['card_id']}.")
    return validation_errors


def _build_schema_payload(schema_version: str, example_card: dict[str, Any]) -> dict[str, Any]:
    """Create a compact schema/reference file for UI integration."""

    return {
        "schema_name": "td_case2_search_result_card",
        "schema_version": schema_version,
        "field_definitions": {
            "card_id": {"meaning": "Stable UI card id.", "required": True, "ui_use": "Frontend key"},
            "result_type": {"meaning": "High-level result category.", "required": True, "ui_use": "Filtering"},
            "query": {"meaning": "Original search query.", "required": True, "ui_use": "Result header"},
            "query_mode": {"meaning": "Search mode used for the result.", "required": True, "ui_use": "Debug/filter"},
            "rank": {"meaning": "Rank within query results.", "required": True, "ui_use": "Sort/display"},
            "score": {"meaning": "Deterministic search score.", "required": True, "ui_use": "Confidence context"},
            "match_reason": {"meaning": "Why the record matched.", "required": True, "ui_use": "Explainability"},
            "track_id": {"meaning": "Underlying track id.", "required": True, "ui_use": "Deep-link/debug"},
            "search_record_id": {"meaning": "Underlying Step 07 search index id.", "required": True, "ui_use": "API/debug"},
            "title": {"meaning": "Primary display title.", "required": True, "ui_use": "Card heading"},
            "subtitle": {"meaning": "Timestamp/status/confidence line.", "required": True, "ui_use": "Card summary"},
            "description": {"meaning": "Human-readable plain-language summary.", "required": True, "ui_use": "Card body"},
            "vehicle": {"meaning": "Vehicle class/color metadata.", "required": True, "ui_use": "Filter chips"},
            "license_plate": {"meaning": "Verified plate and weak OCR metadata.", "required": True, "ui_use": "Plate badge/detail"},
            "time": {"meaning": "Best and range timestamps.", "required": True, "ui_use": "Timeline/jump-to-video"},
            "media": {"meaning": "Preview image paths and availability flags.", "required": True, "ui_use": "Preview rendering"},
            "quality": {"meaning": "Selection and confidence labels.", "required": True, "ui_use": "Badges"},
            "ui": {"meaning": "Frontend display hints.", "required": True, "ui_use": "Presentation defaults"},
            "debug": {"meaning": "Source trace/debug-only metadata.", "required": False, "ui_use": "Debug panel"},
        },
        "example_card": example_card,
    }


def run_search_result_packaging(
    *,
    run_dir: Path,
    packaging_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Package Step 07 and Step 08 outputs into UI/demo-ready search result cards."""

    index_payload = read_json(run_dir / "07_vehicle_search_index.json")
    flat_index_path = run_dir / "07_vehicle_search_index_flat.json"
    validation_results_payload = read_json(run_dir / "08_query_validation_results.json") if (run_dir / "08_query_validation_results.json").exists() else {}
    validation_report_payload = read_json(run_dir / "08_query_validation_report.json") if (run_dir / "08_query_validation_report.json").exists() else {}
    stage_gate_payload = read_json(run_dir / "00_stage_gate_report.json") if (run_dir / "00_stage_gate_report.json").exists() else {}

    flat_index_records = json.loads(flat_index_path.read_text(encoding="utf-8")) if flat_index_path.exists() else []
    step07_records = list(index_payload.get("records", []))
    if not step07_records:
        raise FileNotFoundError("Step 07 search index is missing records, so Step 09 cannot package results.")

    matches_path = run_dir / "08_query_validation_matches.json"
    matches_missing = not matches_path.exists()
    matches_payload = read_json(matches_path) if matches_path.exists() else {"status": "missing", "queries": []}
    query_items = list(matches_payload.get("queries", []))
    status = "success"
    warnings: list[str] = []
    if matches_missing:
        status = "partial_success"
        warnings.append("Step 08 matches file is missing. Built fallback sample cards directly from Step 07 index.")

    record_by_search_id = {str(record.get("search_record_id", "")): record for record in step07_records}
    record_by_track_id = {str(record.get("track_id", "")): record for record in step07_records}

    if matches_missing or not bool(packaging_config["build_demo_queries"]):
        fallback_matches: list[dict[str, Any]] = []
        for rank, record in enumerate(step07_records[: int(packaging_config["top_k"])], start=1):
            fallback_matches.append(
                {
                    "query": "step07_index_sample",
                    "mode": "color_class",
                    "rank": rank,
                    "score": round(float(record.get("search_confidence", 0.0) or 0.0) * 100.0, 6),
                    "match_reason": ["step07_fallback_sample"],
                    "search_record_id": str(record.get("search_record_id", "")),
                    "track_id": str(record.get("track_id", "")),
                }
            )
        query_items = [{"query": "step07_index_sample", "mode": "color_class", "matches": fallback_matches}]

    deduped_cards: list[dict[str, Any]] = []
    dedupe_map: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    query_packages: list[dict[str, Any]] = []

    for query_index, query_item in enumerate(query_items, start=1):
        query = str(query_item.get("query", "") or "")
        mode = str(query_item.get("mode", "auto") or "auto")
        package_cards: list[dict[str, Any]] = []
        for match in list(query_item.get("matches", []))[: int(packaging_config["top_k"])]:
            record = record_by_search_id.get(str(match.get("search_record_id", ""))) or record_by_track_id.get(str(match.get("track_id", "")))
            if record is None:
                continue
            dedupe_key = (
                query,
                mode,
                str(record.get("track_id", "")),
                int(match.get("rank", 0) or 0),
            )
            existing_card = dedupe_map.get(dedupe_key)
            if existing_card is None:
                card_id = f"result_card_{len(deduped_cards) + 1:06d}"
                existing_card = _build_card(
                    card_id=card_id,
                    run_dir=run_dir,
                    record=record,
                    match=match,
                    schema_version=str(packaging_config["result_card_version"]),
                    include_weak_ocr=bool(packaging_config["include_weak_ocr"]),
                    include_invalid_debug_fields=bool(packaging_config["include_invalid_debug_fields"]),
                    include_debug_paths=bool(packaging_config["include_debug_paths"]),
                    validate_path_strings=bool(packaging_config["validate_path_strings"]),
                )
                dedupe_map[dedupe_key] = existing_card
                deduped_cards.append(existing_card)
            package_cards.append(existing_card)

        query_packages.append(
            {
                "query_id": f"demo_query_{query_index:06d}",
                "query": query,
                "mode": mode,
                "query_type": _query_type(query_item),
                "total_matches": len(list(query_item.get("matches", []))),
                "cards_returned": len(package_cards),
                "top_k": int(packaging_config["top_k"]),
                "cards": package_cards,
            }
        )

    flat_cards = [_build_flat_card_row(card) for card in deduped_cards]
    validation_errors = _validate_cards(deduped_cards)
    if validation_errors:
        status = "partial_success" if status != "failed" else "failed"
        warnings.extend(validation_errors)

    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    result_badge_counts = {"verified_plate": 0, "weak_ocr": 0, "color_class": 0, "basic_track": 0}
    for card in deduped_cards:
        confidence_counts[card["quality"]["confidence_badge"]] += 1
        result_badge_counts[card["quality"]["result_badge"]] += 1

    cards_with_crop = sum(1 for card in deduped_cards if card["media"]["crop_available"])
    cards_with_full_frame = sum(1 for card in deduped_cards if card["media"]["full_frame_available"])
    cards_with_contact_sheet = sum(1 for card in deduped_cards if card["media"]["contact_sheet_available"])
    cards_with_verified_plate = sum(1 for card in deduped_cards if card["license_plate"]["has_verified_plate"])
    cards_with_weak_ocr = sum(
        1 for card in deduped_cards if card["ui"]["show_weak_ocr_badge"]
    )

    generic_cards_payload = {
        "status": status,
        "schema_version": str(packaging_config["result_card_version"]),
        "source_index_file": "07_vehicle_search_index.json",
        "source_validation_file": "08_query_validation_matches.json",
        "summary": {
            "total_query_packages": len(query_packages),
            "total_cards_created": len(deduped_cards),
            "cards_with_verified_plate": cards_with_verified_plate,
            "cards_with_weak_ocr": cards_with_weak_ocr,
            "cards_with_full_frame": cards_with_full_frame,
            "cards_with_crop": cards_with_crop,
            "high_confidence_cards": confidence_counts["high"],
            "medium_confidence_cards": confidence_counts["medium"],
            "low_confidence_cards": confidence_counts["low"],
        },
        "cards": deduped_cards,
    }
    query_packages_payload = {
        "status": status,
        "schema_version": str(packaging_config["result_card_version"]),
        "top_k": int(packaging_config["top_k"]),
        "query_packages": query_packages,
    }
    schema_payload = _build_schema_payload(
        str(packaging_config["result_card_version"]),
        deduped_cards[0] if deduped_cards else {},
    )
    report_payload = {
        "status": status,
        "source_records_loaded": len(step07_records),
        "source_flat_records_loaded": len(flat_index_records),
        "source_validation_queries_loaded": len(query_items),
        "source_validation_status": str(validation_results_payload.get("status", matches_payload.get("status", "missing"))),
        "step08_report_status": str(validation_report_payload.get("status", "missing")),
        "stage_gate_overall_status": str(stage_gate_payload.get("overall_status", "unknown")),
        "total_query_packages": len(query_packages),
        "total_cards_created": len(deduped_cards),
        "unique_tracks_packaged": len({card["track_id"] for card in deduped_cards}),
        "cards_with_verified_plate": cards_with_verified_plate,
        "cards_with_weak_ocr": cards_with_weak_ocr,
        "cards_with_full_frame": cards_with_full_frame,
        "cards_with_crop": cards_with_crop,
        "confidence_badge_counts": confidence_counts,
        "result_badge_counts": result_badge_counts,
        "path_availability": {
            "cards_with_crop": cards_with_crop,
            "cards_missing_crop": len(deduped_cards) - cards_with_crop,
            "cards_with_full_frame": cards_with_full_frame,
            "cards_missing_full_frame": len(deduped_cards) - cards_with_full_frame,
            "cards_with_contact_sheet": cards_with_contact_sheet,
            "cards_missing_contact_sheet": len(deduped_cards) - cards_with_contact_sheet,
        },
        "example_cards": flat_cards[:5],
        "warnings": warnings,
        "recommendation": "Proceed to Step 10 Search Demo Runner / API-style query runner.",
    }

    write_json(run_dir / "09_search_result_cards.json", generic_cards_payload)
    write_json_any(run_dir / "09_search_result_cards_flat.json", flat_cards)
    write_json(run_dir / "09_demo_query_result_packages.json", query_packages_payload)
    write_json(run_dir / "09_search_result_card_schema.json", schema_payload)
    write_json(run_dir / "09_search_result_packaging_report.json", report_payload)
    return generic_cards_payload, flat_cards, query_packages_payload, schema_payload, report_payload
