from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any
from traffic_search_common import build_traffic_index_payload, write_traffic_index_outputs


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"


@dataclass(frozen=True)
class Step07BConfig:
    run_dir: Path


def read_config() -> Step07BConfig:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 07B.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")
    required_inputs = [
        run_dir / "03_yolo_detections.json",
        run_dir / "04B_tracks.json",
        run_dir / "05_best_track_frames.json",
        run_dir / "01_video_info.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 07B input is missing: {required_path}")
    return Step07BConfig(run_dir=run_dir.resolve())


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    payload = {
        "status": "failed",
        "schema_version": "v2",
        "summary": {
            "total_object_records": 0,
            "track_records": 0,
            "detection_records": 0,
            "records_with_verified_color": 0,
            "records_with_verified_plate": 0,
            "image_path_ready": False,
        },
        "records": [],
        "error_message": error_message,
    }
    report = {
        "status": "failed",
        "total_object_records": 0,
        "track_records": 0,
        "detection_records": 0,
        "class_counts": {},
        "object_type_counts": {},
        "verified_color_counts": {},
        "quality_counts": {},
        "color_status_counts": {},
        "plate_status_counts": {},
        "records_with_verified_color": 0,
        "records_with_possible_color": 0,
        "records_with_unknown_color": 0,
        "color_conflict_count": 0,
        "tail_light_confusion_warning_count": 0,
        "records_with_verified_plate": 0,
        "records_with_possible_plate": 0,
        "rejected_plate_count": 0,
        "full_frame_counts": {},
        "crop_counts": {},
        "ignored_class_counts": {},
        "excluded_by_whitelist_detection_count": 0,
        "image_path_ready": False,
        "warnings_count": 0,
        "error_message": error_message,
    }
    write_json(run_dir / "07B_traffic_object_search_index.json", payload)
    write_json_any(run_dir / "07B_traffic_object_search_index_flat.json", [])
    write_json(run_dir / "07B_traffic_object_search_index_report.json", report)


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        payload, flat_records, report = build_traffic_index_payload(config.run_dir)
        write_traffic_index_outputs(config.run_dir, payload, flat_records, report)
        update_stage_gate_report(
            config.run_dir,
            "07B_traffic_object_search_index",
            {
                "status": payload["status"],
                "total_object_records": report["total_object_records"],
                "track_records": report["track_records"],
                "detection_records": report["detection_records"],
                "class_counts": report["class_counts"],
                "image_path_ready": report["image_path_ready"],
                "records_with_verified_color": report["records_with_verified_color"],
                "records_with_possible_color": report["records_with_possible_color"],
                "records_with_verified_plate": report["records_with_verified_plate"],
                "records_with_possible_plate": report["records_with_possible_plate"],
                "rejected_plate_count": report["rejected_plate_count"],
                "tail_light_confusion_warning_count": report["tail_light_confusion_warning_count"],
                "records_with_full_frame": report["full_frame_counts"]["records_with_full_frame"],
                "ready_for_step08b_dynamic_search_validation": report["total_object_records"] > 0,
            },
        )
        log(f"Object records created: {report['total_object_records']}")
        log(f"Track records: {report['track_records']}")
        log(f"Detection fallback records: {report['detection_records']}")
        log(f"Traffic classes: {report['class_counts']}")
        log(
            "Output paths: "
            f"{config.run_dir / '07B_traffic_object_search_index.json'} | "
            f"{config.run_dir / '07B_traffic_object_search_index_flat.json'} | "
            f"{config.run_dir / '07B_traffic_object_search_index_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "07B_traffic_object_search_index", build_failure_payload(exc))
        log(f"Step 07B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
