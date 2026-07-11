from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, read_json, update_stage_gate_report, write_json
from traffic_search_common import INVALID_OCR_TERMS, path_exists, run_traffic_search


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"


@dataclass(frozen=True)
class Step08BConfig:
    run_dir: Path


def read_config() -> Step08BConfig:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 08B.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")
    required_inputs = [
        run_dir / "07B_traffic_object_search_index.json",
        run_dir / "07B_traffic_object_search_index_flat.json",
        run_dir / "07B_traffic_object_search_index_report.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 08B input is missing: {required_path}")
    return Step08BConfig(run_dir=run_dir.resolve())


def _test_result(category: str, query: str, passed: bool, matches: int, details: str) -> dict[str, Any]:
    return {
        "category": category,
        "query": query,
        "passed": passed,
        "matches_found": matches,
        "details": details,
    }


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    write_json(
        run_dir / "08B_dynamic_search_validation_results.json",
        {
            "status": "failed",
            "validation_summary": {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "pass_rate": 0.0},
            "test_results": [],
            "error_message": error_message,
        },
    )
    write_json(
        run_dir / "08B_dynamic_search_validation_matches.json",
        {"status": "failed", "queries": [], "error_message": error_message},
    )
    write_json(
        run_dir / "08B_dynamic_search_validation_report.json",
        {
            "status": "failed",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "pass_rate": 0.0,
            "category_counts": {},
            "recommendation": error_message,
            "error_message": error_message,
        },
    )


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        payload = read_json(config.run_dir / "07B_traffic_object_search_index.json")
        records = list(payload.get("records", []))
        test_results: list[dict[str, Any]] = []
        query_matches: list[dict[str, Any]] = []

        class_counts = Counter(str(item.get("class_name", "") or "unknown") for item in records)
        color_counts = Counter(
            str(item.get("verified_vehicle_color", "") or "").strip()
            for item in records
            if item.get("verified_vehicle_color")
        )
        possible_color_counts = Counter(
            str(item.get("possible_vehicle_color", "") or "").strip()
            for item in records
            if item.get("possible_vehicle_color")
        )
        combo_counts = Counter(
            f"{item.get('verified_vehicle_color')} {item.get('class_name')}"
            for item in records
            if item.get("verified_vehicle_color") and item.get("class_name")
        )
        possible_combo_counts = Counter(
            f"{item.get('possible_vehicle_color')} {item.get('class_name')}"
            for item in records
            if item.get("possible_vehicle_color") and item.get("class_name")
        )
        plates = []
        timestamps = []
        for item in records:
            plate = str(item.get("verified_license_plate", "") or "").strip()
            if plate and plate not in plates:
                plates.append(plate)
            timestamp = str(item.get("timestamp_text", "") or "").strip()
            if timestamp and timestamp not in timestamps:
                timestamps.append(timestamp)

        verified_color_pair_tests = 0
        possible_color_pair_tests = 0
        verified_plate_exact_tests = 0

        def run_and_record(
            category: str,
            query: str,
            expect_results: bool = True,
            *,
            include_uncertain_colors: bool = False,
            include_possible_plates: bool = False,
        ) -> None:
            result = run_traffic_search(
                records,
                query,
                top_k=20,
                time_tolerance_seconds=5.0,
                require_full_frame=True,
                run_dir=config.run_dir,
                include_uncertain_colors=include_uncertain_colors,
                include_possible_plates=include_possible_plates,
            )
            matches = list(result.get("matches", []))
            passed = (len(matches) > 0) if expect_results else bool(result.get("blocked"))
            details = result.get("reason", "ok")
            if expect_results and matches:
                bad_paths = [
                    match for match in matches
                    if not path_exists(config.run_dir, match["record"].get("full_frame_path"))
                ]
                if bad_paths:
                    passed = False
                    details = "returned_result_missing_full_frame_path_on_disk"
                if category == "class_color_search":
                    for match in matches:
                        record = match["record"]
                        if record.get("verified_vehicle_color") is None:
                            passed = False
                            details = "structured_color_search_returned_unverified_color"
                            break
                if category == "possible_color_search":
                    for match in matches:
                        record = match["record"]
                        if not (record.get("verified_vehicle_color") or record.get("possible_vehicle_color")):
                            passed = False
                            details = "possible_color_query_returned_record_without_color"
                            break
                if category == "possible_class_color_search":
                    for match in matches:
                        record = match["record"]
                        if not (record.get("verified_vehicle_color") or record.get("possible_vehicle_color")):
                            passed = False
                            details = "possible_class_color_query_returned_record_without_color"
                            break
                if category == "verified_plate_search":
                    for match in matches:
                        record = match["record"]
                        if record.get("verified_plate_status") != "verified":
                            passed = False
                            details = "verified_plate_query_returned_non_verified_plate"
                            break
            test_results.append(_test_result(category, query, passed, len(matches), details))
            query_matches.append(
                {
                    "category": category,
                    "query": query,
                    "blocked": bool(result.get("blocked")),
                    "reason": result.get("reason"),
                    "matches": [
                        {
                            "object_record_id": match["record"].get("object_record_id"),
                            "class_name": match["record"].get("class_name"),
                            "timestamp_text": match["record"].get("timestamp_text"),
                            "full_frame_path": match["record"].get("full_frame_path"),
                            "crop_path": match["record"].get("crop_path"),
                            "score": match.get("score"),
                            "verified_vehicle_color": match["record"].get("verified_vehicle_color"),
                            "possible_vehicle_color": match["record"].get("possible_vehicle_color"),
                            "verified_license_plate": match["record"].get("verified_license_plate"),
                            "possible_plate_text": match["record"].get("possible_plate_text"),
                            "match_explanation": match.get("match_explanation"),
                        }
                        for match in matches
                    ],
                }
            )

        for class_name in sorted(class_counts):
            run_and_record("class_search", class_name)
        for color_name, _count in color_counts.most_common(10):
            run_and_record("color_search", color_name)
        for color_name, _count in possible_color_counts.most_common(10):
            run_and_record("possible_color_search", color_name, include_uncertain_colors=True)
        for combo, _count in combo_counts.most_common(10):
            verified_color_pair_tests += 1
            run_and_record("class_color_search", combo)
        for combo, _count in possible_combo_counts.most_common(10):
            possible_color_pair_tests += 1
            run_and_record("possible_class_color_search", combo, include_uncertain_colors=True)
        for plate in plates[:10]:
            verified_plate_exact_tests += 1
            run_and_record("verified_plate_search", plate)
        for timestamp in timestamps[:10]:
            run_and_record("timestamp_search", timestamp)
        for invalid_term in sorted(INVALID_OCR_TERMS):
            run_and_record("invalid_ocr_blocking", invalid_term, expect_results=False)

        if any(
            item.get("class_name") == "car"
            and (item.get("verified_vehicle_color") == "red" or item.get("possible_vehicle_color") == "red")
            for item in records
        ):
            run_and_record("special_red_car_search", "red car", expect_results=True, include_uncertain_colors=True)
        else:
            red_car_result = run_traffic_search(
                records,
                "red car",
                top_k=20,
                time_tolerance_seconds=5.0,
                require_full_frame=True,
                run_dir=config.run_dir,
                include_uncertain_colors=True,
            )
            red_car_matches = list(red_car_result.get("matches", []))
            passed = len(red_car_matches) == 0
            test_results.append(
                _test_result(
                    "special_red_car_search",
                    "red car",
                    passed,
                    len(red_car_matches),
                    "expected_zero_without_verified_red_car" if passed else "unexpected_red_car_match",
                )
            )

        total_tests = len(test_results)
        passed_tests = sum(1 for item in test_results if item["passed"])
        failed_tests = total_tests - passed_tests
        category_counts = Counter(item["category"] for item in test_results)
        results_payload = {
            "status": "success",
            "source_index_file": "07B_traffic_object_search_index.json",
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": round((passed_tests / total_tests) * 100.0, 3) if total_tests else 0.0,
            },
            "test_results": test_results,
        }
        matches_payload = {"status": "success", "queries": query_matches}
        report_payload = {
            "status": "success",
            "total_records_loaded": len(records),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": results_payload["validation_summary"]["pass_rate"],
            "category_counts": dict(category_counts),
            "class_search_tests": int(category_counts.get("class_search", 0)),
            "color_search_tests": int(category_counts.get("color_search", 0)),
            "class_color_search_tests": int(category_counts.get("class_color_search", 0)),
            "verified_plate_search_tests": int(category_counts.get("verified_plate_search", 0)),
            "possible_color_search_tests": int(category_counts.get("possible_color_search", 0)),
            "possible_class_color_search_tests": int(category_counts.get("possible_class_color_search", 0)),
            "timestamp_search_tests": int(category_counts.get("timestamp_search", 0)),
            "invalid_ocr_blocking_tests": int(category_counts.get("invalid_ocr_blocking", 0)),
            "records_with_verified_color": sum(1 for item in records if item.get("verified_vehicle_color")),
            "records_with_possible_color": sum(1 for item in records if item.get("possible_vehicle_color")),
            "records_with_unknown_color": sum(1 for item in records if item.get("color_status") == "unknown"),
            "records_with_verified_plate": sum(1 for item in records if item.get("verified_plate_status") == "verified"),
            "records_with_possible_plate": sum(1 for item in records if item.get("verified_plate_status") == "possible"),
            "rejected_plate_count": sum(1 for item in records if item.get("verified_plate_status") == "rejected"),
            "tail_light_confusion_warning_count": sum(
                1 for item in records if item.get("color_warning") == "possible_tail_light_color_confusion"
            ),
            "class_color_query_pass_rate": round(
                (
                    sum(1 for item in test_results if item["category"] == "class_color_search" and item["passed"])
                    / max(1, verified_color_pair_tests)
                )
                * 100.0,
                3,
            ),
            "possible_color_query_pass_rate": round(
                (
                    sum(1 for item in test_results if item["category"] == "possible_class_color_search" and item["passed"])
                    / max(1, possible_color_pair_tests)
                )
                * 100.0,
                3,
            ),
            "exact_plate_query_pass_rate": round(
                (
                    sum(1 for item in test_results if item["category"] == "verified_plate_search" and item["passed"])
                    / max(1, verified_plate_exact_tests)
                )
                * 100.0,
                3,
            ),
            "image_paths_exist_for_returned_results": all(
                item["passed"] for item in test_results if item["category"] != "invalid_ocr_blocking"
            ),
            "recommendation": "Universal traffic object search index is ready." if failed_tests == 0 else "Review failed validation categories before demo use.",
        }
        write_json(config.run_dir / "08B_dynamic_search_validation_results.json", results_payload)
        write_json(config.run_dir / "08B_dynamic_search_validation_matches.json", matches_payload)
        write_json(config.run_dir / "08B_dynamic_search_validation_report.json", report_payload)
        update_stage_gate_report(
            config.run_dir,
            "08B_dynamic_search_validation",
            {
                "status": "success" if failed_tests == 0 else "failed",
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": report_payload["pass_rate"],
                "class_color_query_pass_rate": report_payload["class_color_query_pass_rate"],
                "possible_color_query_pass_rate": report_payload["possible_color_query_pass_rate"],
                "exact_plate_query_pass_rate": report_payload["exact_plate_query_pass_rate"],
                "image_paths_exist_for_returned_results": report_payload["image_paths_exist_for_returned_results"],
                "ready_for_step09b_universal_search_cards": failed_tests == 0,
            },
        )
        log(f"Records loaded: {len(records)}")
        log(f"Total tests: {total_tests}")
        log(f"Passed tests: {passed_tests}")
        log(f"Failed tests: {failed_tests}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "08B_dynamic_search_validation", build_failure_payload(exc))
        log(f"Step 08B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
