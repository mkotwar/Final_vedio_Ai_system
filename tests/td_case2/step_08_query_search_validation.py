from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stage_checks import format_seconds_text, read_json, write_json


INVALID_QUERY_TOKENS = {
    "victory",
    "unanswerable",
    "lancerailairambulance",
    "dev",
    "stop",
}
CRITICAL_GROUPS = {"exact_plate", "color_class", "invalid_ocr_blocking", "path_availability"}
NON_CRITICAL_GROUPS = {"timestamp", "combined", "weak_ocr"}


def normalize_text(text: str) -> str:
    """Normalize generic text for matching."""

    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_plate(text: str) -> str:
    """Normalize a plate-like text into compact uppercase."""

    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def parse_time_query(query: str) -> tuple[float | None, str | None]:
    """Parse a time-like query into seconds and canonical text."""

    normalized = normalize_text(query)
    if not normalized:
        return None, None

    hhmmss_match = re.search(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b", normalized)
    if hhmmss_match:
        hours = int(hhmmss_match.group(1) or 0)
        minutes = int(hhmmss_match.group(2) or 0)
        seconds = int(hhmmss_match.group(3) or 0)
        total_seconds = float(hours * 3600 + minutes * 60 + seconds)
        return total_seconds, format_seconds_text(total_seconds)

    mmss_match = re.search(r"\b(\d{1,2}):(\d{2})\b", normalized)
    if mmss_match:
        minutes = int(mmss_match.group(1) or 0)
        seconds = int(mmss_match.group(2) or 0)
        total_seconds = float(minutes * 60 + seconds)
        return total_seconds, format_seconds_text(total_seconds)

    seconds_match = re.search(r"\b(\d+)\s*seconds?\b", normalized)
    if seconds_match:
        total_seconds = float(int(seconds_match.group(1)))
        return total_seconds, format_seconds_text(total_seconds)

    at_match = re.search(r"\bat\s+(\d+)\b", normalized)
    if at_match:
        total_seconds = float(int(at_match.group(1)))
        return total_seconds, format_seconds_text(total_seconds)

    return None, None


def _has_invalid_query_token(query: str) -> bool:
    """Return whether the query is an invalid OCR token that should be blocked."""

    normalized = normalize_text(query)
    normalized_compact = normalize_plate(normalized).lower()
    return normalized in INVALID_QUERY_TOKENS or normalized_compact in INVALID_QUERY_TOKENS


def _record_path_status(record: dict[str, Any], run_dir: Path) -> dict[str, bool]:
    """Check whether important preview paths exist."""

    def path_exists(path_value: str | None) -> bool:
        if not path_value:
            return False
        path = Path(path_value)
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        return path.exists()

    return {
        "best_crop_path_exists": path_exists(record.get("best_crop_path")),
        "best_full_frame_path_exists": path_exists(record.get("best_full_frame_path")),
        "contact_sheet_path_exists": path_exists(record.get("contact_sheet_path")),
    }


def _match_exact_plate(records: list[dict[str, Any]], plate_query: str) -> list[dict[str, Any]]:
    """Search by exact verified plate only."""

    normalized_plate = normalize_plate(plate_query)
    matches: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("exact_plate_search_enabled")):
            continue
        if normalize_plate(str(record.get("verified_license_plate", ""))) == normalized_plate:
            matches.append(
                {
                    "record": record,
                    "score": 100.0,
                    "match_reason": ["exact_verified_plate"],
                }
            )
    return matches


def _match_color_class(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Search by color/class fields only."""

    query_tokens = set(normalize_text(query).split())
    matches: list[dict[str, Any]] = []
    for record in records:
        score = 0.0
        reasons: list[str] = []
        vehicle_class = normalize_text(str(record.get("vehicle_class", "")))
        dominant_class = normalize_text(str(record.get("dominant_class_name", "")))
        vehicle_color = normalize_text(str(record.get("vehicle_color", "")))

        if vehicle_class in query_tokens or dominant_class in query_tokens:
            score += 20.0
            reasons.append("vehicle_class")
        if vehicle_color != "unknown" and vehicle_color in query_tokens:
            score += 20.0
            reasons.append("vehicle_color")
        if score > 0:
            matches.append({"record": record, "score": score, "match_reason": reasons})
    return matches


def _match_class_only(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Search by class only."""

    query_tokens = set(normalize_text(query).split())
    matches: list[dict[str, Any]] = []
    for record in records:
        vehicle_class = normalize_text(str(record.get("vehicle_class", "")))
        dominant_class = normalize_text(str(record.get("dominant_class_name", "")))
        if vehicle_class in query_tokens or dominant_class in query_tokens:
            matches.append({"record": record, "score": 20.0, "match_reason": ["vehicle_class"]})
    return matches


def _match_color_only(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Search by color only."""

    query_tokens = set(normalize_text(query).split())
    matches: list[dict[str, Any]] = []
    for record in records:
        vehicle_color = normalize_text(str(record.get("vehicle_color", "")))
        if vehicle_color != "unknown" and vehicle_color in query_tokens:
            matches.append({"record": record, "score": 20.0, "match_reason": ["vehicle_color"]})
    return matches


def _match_timestamp(records: list[dict[str, Any]], query: str, time_tolerance_seconds: float) -> list[dict[str, Any]]:
    """Search by timestamp proximity."""

    query_seconds, query_text = parse_time_query(query)
    if query_seconds is None:
        return []
    matches: list[dict[str, Any]] = []
    for record in records:
        timestamp_seconds = float(record.get("best_timestamp_seconds", 0.0) or 0.0)
        if abs(timestamp_seconds - query_seconds) <= time_tolerance_seconds:
            score = max(0.0, 15.0 - abs(timestamp_seconds - query_seconds))
            matches.append(
                {
                    "record": record,
                    "score": score,
                    "match_reason": [f"timestamp_within_{query_text or format_seconds_text(query_seconds)}"],
                }
            )
    return matches


def _match_weak_ocr(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Search weak OCR candidates only."""

    normalized_query = normalize_plate(query)
    if not normalized_query or _has_invalid_query_token(query):
        return []
    matches: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("weak_ocr_search_enabled")):
            continue
        matched_possible = False
        for candidate in list(record.get("possible_license_plate_candidates", [])):
            if normalize_plate(str(candidate.get("text", ""))) == normalized_query:
                matched_possible = True
                break
        if matched_possible:
            matches.append({"record": record, "score": 10.0, "match_reason": ["weak_possible_ocr"]})
    return matches


def _match_combined(
    records: list[dict[str, Any]],
    query: str,
    *,
    allow_weak_ocr_search: bool,
    time_tolerance_seconds: float,
) -> list[dict[str, Any]]:
    """Run combined multi-field search scoring."""

    query_tokens = set(normalize_text(query).split())
    query_plate = normalize_plate(query)
    query_seconds, _query_time_text = parse_time_query(query)
    matches: list[dict[str, Any]] = []

    for record in records:
        score = 0.0
        reasons: list[str] = []
        if bool(record.get("exact_plate_search_enabled")) and normalize_plate(str(record.get("verified_license_plate", ""))) == query_plate:
            score += 100.0
            reasons.append("exact_verified_plate")

        vehicle_class = normalize_text(str(record.get("vehicle_class", "")))
        dominant_class = normalize_text(str(record.get("dominant_class_name", "")))
        if vehicle_class in query_tokens or dominant_class in query_tokens:
            score += 20.0
            reasons.append("vehicle_class")

        vehicle_color = normalize_text(str(record.get("vehicle_color", "")))
        if vehicle_color != "unknown" and vehicle_color in query_tokens:
            score += 20.0
            reasons.append("vehicle_color")

        if query_seconds is not None:
            record_seconds = float(record.get("best_timestamp_seconds", 0.0) or 0.0)
            if abs(record_seconds - query_seconds) <= time_tolerance_seconds:
                score += 15.0
                reasons.append("timestamp")

        if allow_weak_ocr_search and query_plate:
            for candidate in list(record.get("possible_license_plate_candidates", [])):
                if normalize_plate(str(candidate.get("text", ""))) == query_plate:
                    score += 10.0
                    reasons.append("weak_possible_ocr")
                    break

        if str(record.get("selection_group", "")) == "primary":
            score += 5.0
            reasons.append("primary_bonus")

        score += float(record.get("search_confidence", 0.0) or 0.0) * 10.0
        if reasons:
            matches.append({"record": record, "score": round(score, 6), "match_reason": reasons})
    return matches


def search_vehicle_index(
    records: list[dict[str, Any]],
    query: str,
    *,
    mode: str = "auto",
    top_k: int = 20,
    allow_weak_ocr_search: bool = True,
    time_tolerance_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """Search the vehicle index using a deterministic mode."""

    if _has_invalid_query_token(query):
        return []

    normalized_query = normalize_text(query)
    query_plate = normalize_plate(query)
    effective_mode = mode
    if mode == "auto":
        if query_plate and len(query_plate) >= 6 and any(char.isdigit() for char in query_plate) and any(char.isalpha() for char in query_plate):
            effective_mode = "exact_plate"
        elif parse_time_query(query)[0] is not None:
            effective_mode = "timestamp"
        else:
            tokens = normalized_query.split()
            if len(tokens) >= 2:
                effective_mode = "combined"
            else:
                effective_mode = "combined"

    if effective_mode == "exact_plate":
        raw_matches = _match_exact_plate(records, query)
    elif effective_mode == "color_class":
        raw_matches = _match_color_class(records, query)
    elif effective_mode == "class_only":
        raw_matches = _match_class_only(records, query)
    elif effective_mode == "color_only":
        raw_matches = _match_color_only(records, query)
    elif effective_mode == "timestamp":
        raw_matches = _match_timestamp(records, query, time_tolerance_seconds)
    elif effective_mode == "weak_ocr":
        raw_matches = _match_weak_ocr(records, query) if allow_weak_ocr_search else []
    elif effective_mode == "combined":
        raw_matches = _match_combined(
            records,
            query,
            allow_weak_ocr_search=allow_weak_ocr_search,
            time_tolerance_seconds=time_tolerance_seconds,
        )
    else:
        raise ValueError(f"Unsupported search mode: {mode}")

    raw_matches.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["record"].get("search_confidence", 0.0) or 0.0),
            str(item["record"].get("track_id", "")),
        )
    )

    structured_matches: list[dict[str, Any]] = []
    for rank, match_item in enumerate(raw_matches[:top_k], start=1):
        record = match_item["record"]
        structured_matches.append(
            {
                "query": query,
                "mode": effective_mode,
                "rank": rank,
                "score": round(float(match_item["score"]), 6),
                "match_reason": match_item["match_reason"],
                "search_record_id": str(record.get("search_record_id", "")),
                "track_id": str(record.get("track_id", "")),
                "vehicle_class": str(record.get("vehicle_class", "")),
                "vehicle_color": str(record.get("vehicle_color", "")),
                "verified_license_plate": str(record.get("verified_license_plate", "not_visible")),
                "has_verified_plate": bool(record.get("verified_license_plate_valid")),
                "best_timestamp_seconds": float(record.get("best_timestamp_seconds", 0.0) or 0.0),
                "best_timestamp_text": str(record.get("best_timestamp_text", "")),
                "best_crop_path": str(record.get("best_crop_path", "")),
                "best_full_frame_path": record.get("best_full_frame_path"),
                "best_plate_crop_path": record.get("best_plate_crop_path"),
                "contact_sheet_path": record.get("contact_sheet_path"),
                "search_confidence": float(record.get("search_confidence", 0.0) or 0.0),
            }
        )
    return structured_matches


def _test_cases() -> dict[str, list[dict[str, Any]]]:
    """Return the deterministic validation test suite."""

    return {
        "exact_plate": [
            {"test_id": "exact_plate_001", "query": "DL12CL4316", "mode": "exact_plate", "expected_track_ids": ["vehicle_track_0038"], "min_matches": 1},
            {"test_id": "exact_plate_002", "query": "DL4SAE0084", "mode": "exact_plate", "expected_track_ids": ["vehicle_track_0101"], "min_matches": 1},
            {"test_id": "exact_plate_003", "query": "HR47F3216", "mode": "exact_plate", "expected_track_ids": ["vehicle_track_0103"], "min_matches": 1},
            {"test_id": "exact_plate_004", "query": "HR38AH0181", "mode": "exact_plate", "expected_track_ids": ["vehicle_track_0112"], "min_matches": 1},
            {"test_id": "exact_plate_005", "query": "DL1LR9174", "mode": "exact_plate", "expected_track_ids": ["vehicle_track_0120"], "min_matches": 1},
        ],
        "color_class": [
            {"test_id": "color_class_001", "query": "black car", "mode": "color_class", "min_matches": 1},
            {"test_id": "color_class_002", "query": "grey car", "mode": "color_class", "min_matches": 1},
            {"test_id": "color_class_003", "query": "white truck", "mode": "color_class", "min_matches": 1},
            {"test_id": "color_class_004", "query": "red motorcycle", "mode": "color_class", "min_matches": 1},
            {"test_id": "color_class_005", "query": "blue bus", "mode": "color_class", "min_matches": 1},
        ],
        "timestamp": [
            {"test_id": "timestamp_001", "query": "00:37", "mode": "timestamp", "expected_track_ids": ["vehicle_track_0021"], "min_matches": 1},
            {"test_id": "timestamp_002", "query": "01:12", "mode": "timestamp", "expected_track_ids": ["vehicle_track_0038"], "min_matches": 1},
            {"test_id": "timestamp_003", "query": "02:45", "mode": "timestamp", "expected_track_ids": ["vehicle_track_0101"], "min_matches": 1},
        ],
        "combined": [
            {"test_id": "combined_001", "query": "grey car DL12CL4316", "mode": "combined", "expected_top_track_id": "vehicle_track_0038"},
            {"test_id": "combined_002", "query": "white car HR47F3216", "mode": "combined", "expected_top_track_id": "vehicle_track_0103"},
            {"test_id": "combined_003", "query": "black motorcycle 00:37", "mode": "combined", "expected_top_track_id": "vehicle_track_0021"},
        ],
        "weak_ocr": [
            {"test_id": "weak_ocr_001", "query": "CITYDL1FT", "mode": "exact_plate", "expected_match_count": 0},
            {"test_id": "weak_ocr_002", "query": "CITYDL1FT", "mode": "weak_ocr", "expected_track_ids": ["vehicle_track_0059"], "min_matches": 1},
        ],
        "invalid_ocr_blocking": [
            {"test_id": "invalid_ocr_001", "query": "VICTORY", "mode": "auto", "expected_match_count": 0},
            {"test_id": "invalid_ocr_002", "query": "UNANSWERABLE", "mode": "auto", "expected_match_count": 0},
            {"test_id": "invalid_ocr_003", "query": "LANCERAILAIRAMBULANCE", "mode": "auto", "expected_match_count": 0},
            {"test_id": "invalid_ocr_004", "query": "DEV", "mode": "auto", "expected_match_count": 0},
        ],
    }


def _evaluate_test_case(test_case: dict[str, Any], matches: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Evaluate one validation test result."""

    expected_track_ids = list(test_case.get("expected_track_ids", []))
    expected_top_track_id = test_case.get("expected_top_track_id")
    expected_match_count = test_case.get("expected_match_count")
    min_matches = int(test_case.get("min_matches", 0) or 0)
    matched_track_ids = [item["track_id"] for item in matches]

    status = "passed"
    error_message = None
    warnings: list[str] = []

    if expected_match_count is not None and len(matches) != int(expected_match_count):
        status = "failed"
        error_message = f"Expected {expected_match_count} matches but found {len(matches)}."
    elif min_matches > 0 and len(matches) < min_matches:
        status = "failed"
        error_message = f"Expected at least {min_matches} matches but found {len(matches)}."
    elif expected_track_ids:
        if not any(track_id in matched_track_ids for track_id in expected_track_ids):
            status = "failed"
            error_message = f"Expected one of {expected_track_ids} in matches, got {matched_track_ids[:10]}."
    if status == "passed" and expected_top_track_id:
        top_match_track_id = matches[0]["track_id"] if matches else None
        if top_match_track_id != expected_top_track_id:
            status = "failed"
            error_message = f"Expected top track {expected_top_track_id}, got {top_match_track_id}."

    for match in matches:
        path_status = _record_path_status(match, run_dir)
        if not path_status["best_crop_path_exists"]:
            status = "failed"
            error_message = f"best_crop_path missing for {match['track_id']}."
            break
        if not path_status["best_full_frame_path_exists"]:
            status = "failed"
            error_message = f"best_full_frame_path missing for {match['track_id']}."
            break
        if match.get("contact_sheet_path") and not path_status["contact_sheet_path_exists"]:
            warnings.append(f"contact_sheet_path missing for {match['track_id']}")

    return {
        "test_id": str(test_case["test_id"]),
        "query": str(test_case["query"]),
        "mode": str(test_case["mode"]),
        "status": status,
        "expected_track_ids": expected_track_ids,
        "matched_track_ids": matched_track_ids,
        "match_count": len(matches),
        "top_match": matches[0] if matches else None,
        "warnings": warnings,
        "error_message": error_message,
    }


def run_query_search_validation(
    *,
    run_dir: Path,
    validation_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the Step 07 search index with deterministic query tests."""

    index_payload = read_json(run_dir / "07_vehicle_search_index.json")
    flat_index_path = run_dir / "07_vehicle_search_index_flat.json"
    flat_records = json.loads(flat_index_path.read_text(encoding="utf-8")) if flat_index_path.exists() else []
    report_payload = read_json(run_dir / "07_vehicle_search_index_report.json")

    records = list(index_payload.get("records", []))
    total_records_loaded = len(records)

    if not records:
        empty_results = {
            "status": "no_records",
            "source_index_file": "07_vehicle_search_index.json",
            "test_config": validation_config,
            "validation_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warning_tests": 0,
                "pass_rate": 0.0,
            },
            "test_results": [],
        }
        empty_matches = {"status": "no_records", "queries": []}
        empty_report = {
            "status": "no_records",
            "total_records_loaded": 0,
            "records_with_verified_plate": 0,
            "unique_verified_plate_count": 0,
            "records_with_color": 0,
            "records_with_full_frame": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "pass_rate": 0.0,
            "exact_plate_tests": {"passed": 0, "failed": 0},
            "color_class_tests": {"passed": 0, "failed": 0},
            "timestamp_tests": {"passed": 0, "failed": 0},
            "combined_tests": {"passed": 0, "failed": 0},
            "weak_ocr_tests": {"passed": 0, "failed": 0},
            "invalid_ocr_blocking_tests": {"passed": 0, "failed": 0},
            "path_availability_tests": {"passed": 0, "failed": 0},
            "example_passed_queries": [],
            "failed_tests": [],
            "recommendation": "No Step 07 records were available for validation.",
        }
        write_json(run_dir / "08_query_validation_results.json", empty_results)
        write_json(run_dir / "08_query_validation_matches.json", empty_matches)
        write_json(run_dir / "08_query_validation_report.json", empty_report)
        return empty_results, empty_matches, empty_report

    test_cases_by_group = _test_cases()
    test_results: list[dict[str, Any]] = []
    query_matches_output: list[dict[str, Any]] = []
    group_counts: dict[str, dict[str, int]] = {
        group_name: {"passed": 0, "failed": 0}
        for group_name in list(test_cases_by_group.keys()) + ["path_availability"]
    }

    for group_name, test_cases in test_cases_by_group.items():
        for test_case in test_cases:
            matches = search_vehicle_index(
                records,
                str(test_case["query"]),
                mode=str(test_case["mode"]),
                top_k=int(validation_config["top_k"]),
                allow_weak_ocr_search=bool(validation_config["allow_weak_ocr_search"]),
                time_tolerance_seconds=float(validation_config["time_tolerance_seconds"]),
            )
            query_matches_output.append(
                {
                    "query": str(test_case["query"]),
                    "mode": str(test_case["mode"]),
                    "matches": matches,
                }
            )
            test_result = _evaluate_test_case(test_case, matches, run_dir)
            test_results.append(test_result)
            group_counts[group_name]["passed" if test_result["status"] == "passed" else "failed"] += 1

    path_availability_failed = 0
    for record in records:
        path_status = _record_path_status(record, run_dir)
        if not path_status["best_crop_path_exists"] or not path_status["best_full_frame_path_exists"]:
            path_availability_failed += 1
    if path_availability_failed == 0:
        group_counts["path_availability"]["passed"] = 1
    else:
        group_counts["path_availability"]["failed"] = 1

    total_tests = len(test_results)
    passed_tests = sum(1 for item in test_results if item["status"] == "passed")
    failed_tests = total_tests - passed_tests
    warning_tests = sum(1 for item in test_results if item.get("warnings"))
    pass_rate = round((passed_tests / total_tests) * 100.0, 2) if total_tests > 0 else 0.0

    critical_failures = sum(group_counts[group]["failed"] for group in CRITICAL_GROUPS)
    non_critical_failures = sum(group_counts[group]["failed"] for group in NON_CRITICAL_GROUPS)
    overall_status = "success"
    if critical_failures > 0:
        overall_status = "failed" if bool(validation_config["fail_on_critical_test_failure"]) else "partial_success"
    elif non_critical_failures > 0:
        overall_status = "partial_success"

    results_output = {
        "status": overall_status,
        "source_index_file": "07_vehicle_search_index.json",
        "test_config": validation_config,
        "validation_summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "warning_tests": warning_tests,
            "pass_rate": pass_rate,
        },
        "test_results": test_results,
    }
    matches_output = {
        "status": overall_status,
        "queries": query_matches_output,
    }
    report_output = {
        "status": overall_status,
        "total_records_loaded": total_records_loaded,
        "records_with_verified_plate": int(report_payload.get("records_with_verified_plate", 0) or 0),
        "unique_verified_plate_count": int(report_payload.get("unique_verified_plate_count", 0) or 0),
        "records_with_color": int(report_payload.get("records_with_color", 0) or 0),
        "records_with_full_frame": int(report_payload.get("records_with_full_frame", 0) or 0),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "pass_rate": pass_rate,
        "exact_plate_tests": group_counts["exact_plate"],
        "color_class_tests": group_counts["color_class"],
        "timestamp_tests": group_counts["timestamp"],
        "combined_tests": group_counts["combined"],
        "weak_ocr_tests": group_counts["weak_ocr"],
        "invalid_ocr_blocking_tests": group_counts["invalid_ocr_blocking"],
        "path_availability_tests": group_counts["path_availability"],
        "flat_index_record_count": len(flat_records),
        "example_passed_queries": [
            {"query": item["query"], "mode": item["mode"], "top_track_id": item["top_match"]["track_id"] if item["top_match"] else None}
            for item in test_results
            if item["status"] == "passed"
        ][:10],
        "failed_tests": [
            {
                "test_id": item["test_id"],
                "query": item["query"],
                "mode": item["mode"],
                "error_message": item["error_message"],
            }
            for item in test_results
            if item["status"] != "passed"
        ],
        "recommendation": "Proceed to Step 09 UI/search result packaging if all critical tests pass.",
    }

    write_json(run_dir / "08_query_validation_results.json", results_output)
    write_json(run_dir / "08_query_validation_matches.json", matches_output)
    write_json(run_dir / "08_query_validation_report.json", report_output)
    return results_output, matches_output, report_output
