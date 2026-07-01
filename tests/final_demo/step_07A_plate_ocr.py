from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.plate_ocr import (
    build_plate_ocr_outputs,
    update_run_manifest_for_plate_ocr,
)
from tests.final_demo.services.video_io import write_json


def run_step_07A_plate_ocr(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 7A: plate / OCR candidate reading")

    ocr_result = build_plate_ocr_outputs(run_dir)
    results_path = run_dir / "07A_plate_ocr_results.json"
    report_path = run_dir / "07A_plate_ocr_report.json"

    write_json(results_path, ocr_result["results_payload"])
    write_json(report_path, ocr_result["report_payload"])
    update_run_manifest_for_plate_ocr(run_dir / "run_manifest.json")

    report_payload = ocr_result["report_payload"]
    print(f"[final-demo] OCR backend requested: {report_payload['ocr_backend_requested']}")
    print(f"[final-demo] OCR backend used: {report_payload['ocr_backend_used']}")
    print(f"[final-demo] OCR device requested: {report_payload['ocr_device_requested']}")
    print(f"[final-demo] OCR device used: {report_payload['ocr_device_used']}")
    print(f"[final-demo] CUDA available: {report_payload['cuda_available']}")
    print(f"[final-demo] CUDA device name: {report_payload['cuda_device_name']}")
    print(f"[final-demo] OCR GPU enabled: {report_payload['gpu_enabled_for_ocr']}")
    print(f"[final-demo] Vehicle tracks found: {report_payload['total_vehicle_tracks']}")
    print(
        f"[final-demo] Plate candidates found: "
        f"{report_payload['tracks_with_plate_candidates']}"
    )
    print(f"[final-demo] Frame-scan candidates available: {report_payload.get('frame_scan_candidates_available', 0)}")
    print(f"[final-demo] Frame-scan candidates after filter: {report_payload.get('frame_scan_candidates_after_filter', 0)}")
    print(f"[final-demo] Frame-scan OCR inputs: {report_payload.get('frame_scan_candidates_input', 0)}")
    print(f"[final-demo] Frame-scan OCR results: {report_payload.get('frame_scan_ocr_results', 0)}")
    print(f"[final-demo] Frame-scan skipped missing crop: {report_payload.get('frame_scan_candidates_skipped_missing_crop', 0)}")
    print(f"[final-demo] Frame-scan skipped low score: {report_payload.get('frame_scan_candidates_skipped_low_score', 0)}")
    print(f"[final-demo] Frame-scan skipped limit: {report_payload.get('frame_scan_candidates_skipped_limit', 0)}")
    print(f"[final-demo] OCR attempted count: {report_payload['ocr_attempted_count']}")
    print(f"[final-demo] OCR success count: {report_payload['ocr_success_count']}")
    print(f"[final-demo] Strong reads: {report_payload.get('ocr_strong_count', 0)}")
    print(f"[final-demo] Review reads: {report_payload.get('ocr_needs_review_count', 0)}")
    print(f"[final-demo] Weak reads: {report_payload.get('ocr_weak_count', 0)}")
    print(
        f"[final-demo] Unreadable count: "
        f"{report_payload.get('ocr_unreadable_count', report_payload.get('unreadable_count', 0))}"
    )
    print(f"[final-demo] Results path: {results_path}")
    print(f"[final-demo] Report path: {report_path}")

    return {
        "run_dir": run_dir,
        "results_path": results_path,
        "report_path": report_path,
        "ocr_result": ocr_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 7A from the final demo pipeline or call run_step_07A_plate_ocr(run_dir)."
    )


if __name__ == "__main__":
    main()
