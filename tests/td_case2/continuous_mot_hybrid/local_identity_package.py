from __future__ import annotations

from typing import Any


def _motion_direction(local_object: dict[str, Any]) -> str:
    timeline = list(local_object.get("sanitized_valid_timeline", []))
    if len(timeline) < 2:
        return "unknown"
    if "bbox_xyxy" not in timeline[0] or "bbox_xyxy" not in timeline[-1]:
        return "unknown"
    start = timeline[0]["bbox_xyxy"]
    end = timeline[-1]["bbox_xyxy"]
    start_x = (float(start[0]) + float(start[2])) / 2.0
    end_x = (float(end[0]) + float(end[2])) / 2.0
    if end_x > start_x + 8.0:
        return "left_to_right"
    if end_x < start_x - 8.0:
        return "right_to_left"
    return "stationary_or_vertical"


def build_local_identity_packages(
    *,
    local_objects: list[dict[str, Any]],
    representative_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    representative_by_key = {str(item["local_object_key"]): item for item in representative_frames}
    packages: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for item in local_objects:
        key = str(item["local_object_key"])
        representative = representative_by_key.get(key, {})
        frame_payload = dict(representative.get("representative_frames", {}))
        warnings = sorted(set([*list(item.get("warnings", [])), *list(representative.get("warnings", []))]))
        detector_observations = len([row for row in list(item.get("sanitized_valid_timeline", [])) if str(row.get("bbox_source")) == "yolo"])
        duration_seconds = float(item.get("sanitized_duration_seconds", item.get("duration_seconds", 0.0)) or 0.0)
        status = "manual_review"
        if not frame_payload.get("primary"):
            status = "rejected"
        elif bool(item.get("confirmed")) and duration_seconds >= 0.5 and detector_observations >= 3:
            status = "ready"
        elif frame_payload.get("primary"):
            status = "fallback"
        if "boundary_stuck" in warnings or "frozen_bbox" in warnings:
            status = "manual_review" if status != "rejected" else status
        package = {
            "camera_id": item["camera_id"],
            "camera_group": item["camera_group"],
            "camera_timezone": item["camera_timezone"],
            "local_object_id": item["local_object_id"],
            "global_object_id": None,
            "object_family": item["object_family"],
            "final_class": item["final_class"],
            "class_votes": dict(item.get("class_votes", {})),
            "source_raw_track_ids": list(item.get("source_raw_track_ids", [])),
            "start_timestamp": round(float(item["start_timestamp_seconds"]), 6),
            "end_timestamp": round(float(item["end_timestamp_seconds"]), 6),
            "duration": round(float(item["duration_seconds"]), 6),
            "first_source_frame_index": int(item["first_source_frame_index"]),
            "last_source_frame_index": int(item["last_source_frame_index"]),
            "entry_boundary": item.get("entry_boundary"),
            "exit_boundary": item.get("exit_boundary"),
            "motion_direction": _motion_direction(item),
            "confirmed": bool(item.get("confirmed")),
            "track_quality": item.get("quality_level"),
            "integrity_status": item.get("track_integrity_status"),
            "representative_crops": [
                row["crop_path"]
                for row in [frame_payload.get("primary"), *list(frame_payload.get("alternatives", []))]
                if isinstance(row, dict) and row.get("crop_path")
            ],
            "representative_full_frames": [
                row["full_frame_path"]
                for row in [frame_payload.get("primary"), *list(frame_payload.get("alternatives", []))]
                if isinstance(row, dict) and row.get("full_frame_path")
            ],
            "plate_candidate_frames": [frame_payload["plate_candidate"]["crop_path"]] if isinstance(frame_payload.get("plate_candidate"), dict) else [],
            "verified_plate": None,
            "possible_plate_text": [],
            "vehicle_color": None,
            "person_attributes": None,
            "appearance_embeddings": [],
            "trajectory_summary": {
                "observation_count": len(list(item.get("sanitized_valid_timeline", []))),
                "detector_observations": detector_observations,
                "motion_direction": _motion_direction(item),
            },
            "quality_flags": list(item.get("quality_flags", [])),
            "warnings": warnings,
            "downstream_status": status,
        }
        packages.append(package)
        flat_rows.append(
            {
                "camera_id": package["camera_id"],
                "local_object_id": package["local_object_id"],
                "object_family": package["object_family"],
                "final_class": package["final_class"],
                "start_timestamp": package["start_timestamp"],
                "end_timestamp": package["end_timestamp"],
                "duration": package["duration"],
                "downstream_status": status,
            }
        )
    report = {
        "status": "success",
        "total_packages": len(packages),
        "ready_packages": len([item for item in packages if item["downstream_status"] == "ready"]),
        "fallback_packages": len([item for item in packages if item["downstream_status"] == "fallback"]),
        "manual_review_packages": len([item for item in packages if item["downstream_status"] == "manual_review"]),
        "rejected_packages": len([item for item in packages if item["downstream_status"] == "rejected"]),
        "packages_with_primary_crop": len([item for item in packages if item["representative_crops"]]),
        "packages_with_plate_candidate": len([item for item in packages if item["plate_candidate_frames"]]),
        "vehicle_packages": len([item for item in packages if item["object_family"] == "vehicle"]),
        "person_packages": len([item for item in packages if item["object_family"] == "person"]),
    }
    return packages, flat_rows, report
