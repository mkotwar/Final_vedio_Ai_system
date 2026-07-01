from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def count_by_class(items: list[dict[str, Any]], field: str = "class_name") -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        class_name = str(item.get(field) or "").strip().lower()
        if class_name:
            counts[class_name] += 1
    return dict(sorted(counts.items()))


def count_by_source(items: list[dict[str, Any]], field: str = "candidate_source") -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        source = str(item.get(field) or "").strip().lower()
        if source:
            counts[source] += 1
    return dict(sorted(counts.items()))


def build_class_propagation_audit(run_dir: Path, *, mode: str) -> dict[str, Any]:
    detections_payload = read_optional_json(run_dir / "04_yolo_detections.json")
    tracks_payload = read_optional_json(run_dir / "05_tracks.json")
    clean_tracks_payload = read_optional_json(run_dir / "05B_clean_tracks.json")
    attributes_payload = read_optional_json(run_dir / "06_track_attributes.json")
    plate_candidates_payload = read_optional_json(run_dir / "06A_plate_candidates.json")
    ocr_results_payload = read_optional_json(run_dir / "07A_plate_ocr_results.json")
    event_payload = read_optional_json(run_dir / "07B_event_candidates.json")

    step4 = count_by_class(list((detections_payload or {}).get("detections") or []))
    step5 = count_by_class(list((tracks_payload or {}).get("tracks") or []))
    step5b = count_by_class(list((clean_tracks_payload or {}).get("clean_tracks") or []))
    step6 = count_by_class(list((attributes_payload or {}).get("attributes") or []))
    step6a = count_by_class(list((plate_candidates_payload or {}).get("candidates") or []))
    step7a = count_by_class(list((ocr_results_payload or {}).get("results") or []))
    step7b = count_by_class(list((event_payload or {}).get("events") or []))

    step6a_sources = count_by_source(list((plate_candidates_payload or {}).get("candidates") or []))
    step7a_sources = count_by_source(list((ocr_results_payload or {}).get("results") or []))
    step7b_sources = count_by_source(
        [dict(item.get("attributes") or {}) for item in list((event_payload or {}).get("events") or [])]
    )

    class_drop_warnings: list[str] = []
    possible_causes = [
        "Tracker metadata lost original YOLO class.",
        "Class defaulted during Step 5 track creation.",
        "ByteTrack returned no tracker_id and detections were not preserved.",
        "Cleanup merge changed class.",
        "Step 6 vehicle_type mapping defaulted incorrectly.",
    ]

    tracked_steps = [
        ("Step 5", step5),
        ("Step 5B", step5b),
    ]
    if mode == "final":
        tracked_steps.extend(
            [
                ("Step 6", step6),
                ("Step 6A", step6a),
                ("Step 7A", step7a),
                ("Step 7B", step7b),
            ]
        )

    for class_name, count in sorted(step4.items()):
        if count <= 0:
            continue
        for label, counts in tracked_steps:
            if counts.get(class_name, 0) == 0:
                class_drop_warnings.append(
                    f"YOLO detected {count} {class_name} detections but {label} has 0 {class_name} items."
                )
                break

    overall_status = "ok" if not class_drop_warnings else "needs_review"
    class_propagation_status = "preserved" if not class_drop_warnings else "class_drop_detected"
    report_payload = {
        "created_at": current_timestamp(),
        "audit_mode": mode,
        "overall_status": overall_status,
        "class_propagation_status": class_propagation_status,
        "warnings": class_drop_warnings,
        "recommendations": [
            "Check Step 5 synthetic fallback tracks for untracked detections.",
            "Review mixed-class tracks before enabling any cross-class cleanup merge.",
            "Use the final audit after reruns to confirm car/bus/motorcycle classes survive downstream.",
        ],
    }
    audit_payload = {
        "created_at": current_timestamp(),
        "audit_mode": mode,
        "step4_detections_by_class": step4,
        "step5_tracks_by_class": step5,
        "step5B_clean_tracks_by_class": step5b,
        "class_drop_warnings": class_drop_warnings,
        "possible_causes": possible_causes,
    }
    if mode == "final":
        audit_payload.update(
            {
                "step6_attributes_by_class": step6,
                "step6A_plate_candidates_by_class": step6a,
                "step6A_plate_candidates_by_source": step6a_sources,
                "step7A_ocr_results_by_class": step7a,
                "step7A_ocr_results_by_source": step7a_sources,
                "step7B_events_by_class": step7b,
                "step7B_events_by_source": step7b_sources,
            }
        )
    return {
        "audit_payload": audit_payload,
        "report_payload": report_payload,
    }


def write_class_propagation_audit(
    run_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode == "pre_attribute":
        audit_name = "05C_pre_attribute_class_propagation_audit.json"
        report_name = "05C_pre_attribute_class_propagation_audit_report.json"
        manifest_step = "05C_class_propagation_audit"
        next_step = "06_attribute_extraction"
    else:
        audit_name = "07C_class_propagation_audit.json"
        report_name = "07C_class_propagation_audit_report.json"
        manifest_step = "07C_class_propagation_audit"
        next_step = "08_event_grouping_or_ranking"

    audit_result = build_class_propagation_audit(run_dir, mode="final" if mode == "final" else "pre_attribute")
    audit_path = run_dir / audit_name
    report_path = run_dir / report_name
    write_json(audit_path, audit_result["audit_payload"])
    write_json(report_path, audit_result["report_payload"])
    update_run_manifest_for_class_audit(
        run_dir / "run_manifest.json",
        manifest_step=manifest_step,
        next_step=next_step,
    )
    return {
        "audit_path": audit_path,
        "report_path": report_path,
        "audit_result": audit_result,
    }


def update_run_manifest_for_class_audit(
    run_manifest_path: Path,
    *,
    manifest_step: str,
    next_step: str,
) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if manifest_step not in completed_steps:
        completed_steps.append(manifest_step)
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = next_step
    write_json(run_manifest_path, run_manifest)
    return run_manifest
