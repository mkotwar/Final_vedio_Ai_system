from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.plate_candidate_detector import (
    build_plate_candidate_outputs,
    update_run_manifest_for_plate_candidates,
)
from tests.final_demo.services.video_io import write_json


def run_step_06A_detect_plate_candidates(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 6A: licence plate candidate detection")

    step_result = build_plate_candidate_outputs(run_dir)
    candidates_path = run_dir / "06A_plate_candidates.json"
    report_path = run_dir / "06A_plate_candidate_report.json"

    write_json(candidates_path, step_result["candidates_payload"])
    write_json(report_path, step_result["report_payload"])
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
    print(f"[final-demo] Report path: {report_path}")

    return {
        "run_dir": run_dir,
        "candidates_path": candidates_path,
        "report_path": report_path,
        "step_result": step_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 6A from the final demo pipeline or call run_step_06A_detect_plate_candidates(run_dir)."
    )


if __name__ == "__main__":
    main()
