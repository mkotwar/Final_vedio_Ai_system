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

from tests.final_demo.services.plate_candidate_detector import (  # noqa: E402
    build_plate_candidate_outputs,
    update_run_manifest_for_plate_candidates,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_06A_detect_plate_candidates(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 6A: licence plate candidate detection")

    step_result = build_plate_candidate_outputs(run_dir)
    candidates_path = run_dir / "06A_plate_candidates.json"
    report_path = run_dir / "06A_plate_candidate_report.json"
    class_audit_path = run_dir / "06A_vehicle_class_correction_audit.json"
    class_audit_report_path = run_dir / "06A_vehicle_class_correction_audit_report.json"

    write_json(candidates_path, step_result["candidates_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(class_audit_path, step_result["class_correction_audit_payload"])
    write_json(class_audit_report_path, step_result["class_correction_audit_report"])
    update_run_manifest_for_plate_candidates(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Plate scan mode: {report_payload['plate_scan_mode']}")
    print(f"[final-demo] Vehicle tracks processed: {report_payload['tracks_processed']}")
    print(f"[final-demo] Frames inspected: {report_payload['frames_inspected']}")
    print(f"[final-demo] Frame-scan frames checked: {report_payload['total_frame_scan_frames_checked']}")
    print(f"[final-demo] Frame-scan vehicle crops checked: {report_payload['total_frame_scan_vehicle_crops_checked']}")
    print(f"[final-demo] Plate candidates found: {report_payload['total_plate_candidates']}")
    print(
        f"[final-demo] Candidates by source: "
        f"{report_payload['candidates_by_source']}"
    )
    print(f"[final-demo] Candidate crops saved: {report_payload['candidate_crops_saved']}")
    print(f"[final-demo] Detector mode: {report_payload['plate_detector_mode']}")
    print(
        f"[final-demo] Vehicle class corrections: "
        f"{step_result['class_correction_audit_report']['candidates_with_class_correction']}"
    )
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Class correction audit path: {class_audit_path}")

    return {
        "run_dir": run_dir,
        "candidates_path": candidates_path,
        "report_path": report_path,
        "class_audit_path": class_audit_path,
        "class_audit_report_path": class_audit_report_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 6A plate candidate detection.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 6A directly."
        )
    run_step_06A_detect_plate_candidates(Path(run_dir_value))


if __name__ == "__main__":
    main()
