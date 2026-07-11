from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, read_json, update_stage_gate_report, write_json
from traffic_search_common import make_demo_queries, run_traffic_search


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"


@dataclass(frozen=True)
class Step10BConfig:
    run_dir: Path


def read_config() -> Step10BConfig:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 10B.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")
    if not (run_dir / "07B_traffic_object_search_index.json").exists():
        raise FileNotFoundError(f"Required Step 10B input is missing: {run_dir / '07B_traffic_object_search_index.json'}")
    return Step10BConfig(run_dir=run_dir.resolve())


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    write_json(run_dir / "10B_universal_search_demo_response.json", {"status": "failed", "queries": [], "error_message": error_message})
    write_json(run_dir / "10B_universal_search_demo_report.json", {"status": "failed", "error_message": error_message})


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        payload = read_json(config.run_dir / "07B_traffic_object_search_index.json")
        index_report = read_json(config.run_dir / "07B_traffic_object_search_index_report.json")
        records = list(payload.get("records", []))
        queries = make_demo_queries(records)
        responses = []
        total_cards = 0
        queries_with_results = 0
        for query in queries:
            result = run_traffic_search(
                records,
                query,
                top_k=10,
                time_tolerance_seconds=5.0,
                require_full_frame=True,
                run_dir=config.run_dir,
                include_uncertain_colors=True,
                include_possible_plates=False,
            )
            matches = [
                {
                    "object_record_id": match["record"].get("object_record_id"),
                    "class_name": match["record"].get("class_name"),
                    "timestamp_text": match["record"].get("timestamp_text"),
                    "full_frame_path": match["record"].get("full_frame_path"),
                    "crop_path": match["record"].get("crop_path"),
                    "verified_vehicle_color": match["record"].get("verified_vehicle_color"),
                    "possible_vehicle_color": match["record"].get("possible_vehicle_color"),
                    "verified_license_plate": match["record"].get("verified_license_plate"),
                    "possible_plate_text": match["record"].get("possible_plate_text"),
                    "verified_plate_status": match["record"].get("verified_plate_status"),
                    "match_explanation": match.get("match_explanation"),
                    "score": match.get("score"),
                }
                for match in list(result.get("matches", []))
            ]
            if matches:
                queries_with_results += 1
            total_cards += len(matches)
            responses.append({"query": query, "blocked": bool(result.get("blocked")), "reason": result.get("reason"), "matches": matches})
        response_payload = {"status": "success", "queries": responses}
        report_payload = {
            "status": "success",
            "records_loaded": len(records),
            "queries_run": len(queries),
            "queries_with_results": queries_with_results,
            "total_cards_returned": total_cards,
            "default_queries": queries,
            "records_with_verified_color": index_report.get("records_with_verified_color", 0),
            "records_with_possible_color": index_report.get("records_with_possible_color", 0),
            "records_with_unknown_color": index_report.get("records_with_unknown_color", 0),
            "records_with_verified_plate": index_report.get("records_with_verified_plate", 0),
            "records_with_possible_plate": index_report.get("records_with_possible_plate", 0),
            "rejected_plate_count": index_report.get("rejected_plate_count", 0),
            "color_conflict_count": index_report.get("color_conflict_count", 0),
            "tail_light_confusion_warning_count": index_report.get("tail_light_confusion_warning_count", 0),
            "image_path_readiness": index_report.get("image_path_ready", False),
            "recommendation": "Universal traffic search demo queries are ready.",
        }
        write_json(config.run_dir / "10B_universal_search_demo_response.json", response_payload)
        write_json(config.run_dir / "10B_universal_search_demo_report.json", report_payload)
        update_stage_gate_report(
            config.run_dir,
            "10B_universal_search_demo",
            {
                "status": "success",
                "records_loaded": len(records),
                "queries_run": len(queries),
                "queries_with_results": queries_with_results,
                "total_cards_returned": total_cards,
                "records_with_verified_plate": report_payload["records_with_verified_plate"],
                "records_with_verified_color": report_payload["records_with_verified_color"],
            },
        )
        log(f"Queries run: {len(queries)}")
        log(f"Total cards returned: {total_cards}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "10B_universal_search_demo", build_failure_payload(exc))
        log(f"Step 10B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
