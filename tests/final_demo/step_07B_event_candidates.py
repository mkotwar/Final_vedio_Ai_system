from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _add_project_root_to_sys_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


_add_project_root_to_sys_path()

from tests.final_demo.services.event_candidate_generator import (  # noqa: E402
    build_event_candidate_outputs,
    update_run_manifest_for_event_candidates,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_07B_event_candidates(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 7B: event candidate generation")

    event_result = build_event_candidate_outputs(run_dir)
    events_path = run_dir / "07B_event_candidates.json"
    report_path = run_dir / "07B_event_candidate_report.json"
    audit_path = run_dir / "07B_ocr_event_audit.json"
    audit_report_path = run_dir / "07B_ocr_event_audit_report.json"

    write_json(events_path, event_result["events_payload"])
    write_json(report_path, event_result["report_payload"])
    write_json(audit_path, event_result["audit_payload"])
    write_json(audit_report_path, event_result["audit_report_payload"])
    update_run_manifest_for_event_candidates(run_dir / "run_manifest.json")

    report_payload = event_result["report_payload"]
    audit_report_payload = event_result["audit_report_payload"]
    print(f"[final-demo] Tracks loaded: {report_payload['total_tracks_input']}")
    print(f"[final-demo] Attributes loaded: {report_payload['total_attributes_input']}")
    print(f"[final-demo] OCR results loaded: {report_payload['total_ocr_results_input']}")
    print(f"[final-demo] Frame-scan OCR results: {report_payload.get('frame_scan_plate_ocr_results', 0)}")
    print(f"[final-demo] Event candidates generated: {report_payload['total_event_candidates']}")
    print(f"[final-demo] OCR audit rows loaded: {audit_report_payload['ocr_results_total_loaded']}")
    print(f"[final-demo] OCR audit frame-scan by candidate_source: {audit_report_payload['results_with_candidate_source_frame_scan']}")
    print(f"[final-demo] OCR audit frame-scan by null-track logic: {audit_report_payload['results_with_null_track_ids']}")
    print(f"[final-demo] OCR audit eligible plate events: {audit_report_payload['eligible_for_plate_event_count']}")
    print(
        "[final-demo] OCR audit target HR38AE1442: "
        f"found={audit_report_payload['target_plate_found']} "
        f"eligible={audit_report_payload['target_plate_eligible']} "
        f"skip_reason={audit_report_payload['target_plate_skip_reason']}"
    )
    print(f"[final-demo] Events by type: {report_payload['events_by_type']}")
    print(f"[final-demo] Events needing review: {report_payload['events_needing_review']}")
    print(f"[final-demo] Events path: {events_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] OCR audit path: {audit_path}")
    print(f"[final-demo] OCR audit report path: {audit_report_path}")

    return {
        "run_dir": run_dir,
        "events_path": events_path,
        "report_path": report_path,
        "audit_path": audit_path,
        "audit_report_path": audit_report_path,
        "event_result": event_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 7B event candidate generation.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 7B directly."
        )
    run_step_07B_event_candidates(Path(run_dir_value))


if __name__ == "__main__":
    main()
