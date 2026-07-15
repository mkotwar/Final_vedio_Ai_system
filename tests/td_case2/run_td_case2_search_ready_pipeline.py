from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import ENV_RUN_DIR
from device_manager import record_stage_device
from run_td_case2_step01_02 import log
from run_td_case2_step01_02_02a import main as step01_main
from run_td_case2_step03_yolo import main as step03_main
from run_td_case2_step04a_florence_audit import main as step04a_main
from run_td_case2_step04b_tracking import main as step04b_main
from run_td_case2_step05_best_frames import main as step05_main
from run_td_case2_step06_ocr_color import main as step06_main
from run_td_case2_step07b_traffic_object_search_index import main as step07b_main
from run_td_case2_step08b_dynamic_search_validation import main as step08b_main
from run_td_case2_step09b_universal_search_cards import main as step09b_main
from run_td_case2_step10b_universal_search_demo import main as step10b_main
from stage_checks import write_json


def _record_cpu_stage(run_dir: Path, stage_name: str, component_name: str, reason: str, notes: list[str] | None = None) -> None:
    record_stage_device(
        run_dir=run_dir,
        stage_name=stage_name,
        component_name=component_name,
        supports_gpu=False,
        selected_device="cpu",
        actual_device="cpu",
        preprocessing_device="cpu",
        inference_device="cpu",
        postprocess_device="cpu",
        reason=reason,
        notes=notes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the td_case2 search-ready pipeline.")
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume after completed Step 03 outputs in an existing run directory.",
    )
    return parser.parse_args()


def _validate_resume_run_dir(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    required_outputs = (
        resolved_run_dir / "03_yolo_detections.json",
        resolved_run_dir / "03_yolo_object_crops",
    )
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "Cannot resume after Step 03 because required outputs are missing: "
            + ", ".join(missing_outputs)
        )
    return resolved_run_dir


def _write_pipeline_status(run_dir: Path, latest_completed_step: str, object_search_status: str, error_message: str | None = None) -> None:
    existing = {}
    status_path = run_dir / "pipeline_status.json"
    if status_path.exists():
        try:
            import json

            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(
        {
            "object_search_status": object_search_status,
            "latest_completed_step": latest_completed_step,
            "error_message": error_message,
        }
    )
    write_json(status_path, existing)


def main(resume_run_dir: Path | None = None) -> None:
    if resume_run_dir is None:
        run_dir = step01_main().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Step 01 did not create a valid run directory: {run_dir}")
    else:
        run_dir = _validate_resume_run_dir(resume_run_dir)
        log(f"Resuming search-ready pipeline after Step 03: {run_dir}")

    # The newly created directory is authoritative. Do not reuse a stale shell or
    # .env value, which may refer to an earlier run or even the input video.
    os.environ[ENV_RUN_DIR] = str(run_dir)
    log(f"Search-ready run directory: {run_dir}")
    _write_pipeline_status(run_dir, "step03" if resume_run_dir is not None else "step01_02_02a", "running")
    try:
        if resume_run_dir is None:
            _record_cpu_stage(
                run_dir,
                "01_video_info",
                "opencv_video_metadata",
                "Video metadata extraction uses OpenCV VideoCapture and filesystem I/O on CPU.",
            )
            _record_cpu_stage(
                run_dir,
                "02_frame_sampling",
                "opencv_frame_sampling",
                "Frame decode plus JPEG writes are CPU-bound OpenCV/file I/O in this testcase.",
            )
            _record_cpu_stage(
                run_dir,
                "02A_adaptive_sampling",
                "opencv_motion_sampling",
                "Adaptive sampling uses CPU OpenCV differencing, thresholding, contour logic, and histogram comparison; no full cv2.cuda path is wired here.",
                notes=["OpenCV CUDA may exist on the machine, but this stage still relies on CPU-only control flow and image I/O."],
            )
            step03_main()
            _write_pipeline_status(run_dir, "step03", "running")
        step04a_main()
        _write_pipeline_status(run_dir, "step04a", "running")
        step04b_main()
        _record_cpu_stage(
            run_dir,
            "04B_tracking",
            "deterministic_tracking",
            "Track association is a CPU ranking/matching algorithm over JSON detections; this repo does not implement a GPU tracker for td_case2.",
        )
        _write_pipeline_status(run_dir, "step04b", "running")
        step05_main()
        _record_cpu_stage(
            run_dir,
            "05_best_track_frames",
            "track_frame_selection",
            "Best-frame ranking and crop selection are lightweight CPU scoring and filesystem operations.",
        )
        _write_pipeline_status(run_dir, "step05", "running")
        step06_main()
        _write_pipeline_status(run_dir, "step06", "running")
        step07b_main()
        _record_cpu_stage(
            run_dir,
            "07B_search_index",
            "search_index_enrichment",
            "Search-index enrichment is JSON normalization and indexing logic on CPU.",
        )
        _write_pipeline_status(run_dir, "step07b", "running")
        step08b_main()
        _record_cpu_stage(
            run_dir,
            "08B_search_validation",
            "query_validation",
            "Search validation is deterministic matching and report generation on CPU.",
        )
        _write_pipeline_status(run_dir, "step08b", "running")
        step09b_main()
        _record_cpu_stage(
            run_dir,
            "09B_search_cards",
            "result_packaging",
            "Result-card packaging is JSON formatting and path validation on CPU.",
        )
        _write_pipeline_status(run_dir, "step09b", "running")
        step10b_main()
        _record_cpu_stage(
            run_dir,
            "10B_search_demo",
            "search_demo",
            "Demo query serving is CPU-side filtering and response assembly.",
        )
        _write_pipeline_status(run_dir, "step10b", "ready")
        log(f"Search-ready pipeline complete. TD_CASE2_RUN_DIR={run_dir}")
    except Exception as exc:
        _write_pipeline_status(run_dir, "failed", "failed", error_message=str(exc))
        raise


if __name__ == "__main__":
    main(_parse_args().resume_run_dir)
