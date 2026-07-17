from __future__ import annotations

from typing import Any


def _motion_direction(local_object: dict[str, Any]) -> str:
    trajectory = list(local_object.get("combined_trajectory", []))
    if len(trajectory) < 2:
        return "unknown"
    start_bbox = list(trajectory[0]["bbox_xyxy"])
    end_bbox = list(trajectory[-1]["bbox_xyxy"])
    start_center_y = (float(start_bbox[1]) + float(start_bbox[3])) / 2.0
    end_center_y = (float(end_bbox[1]) + float(end_bbox[3])) / 2.0
    if end_center_y > start_center_y + 5.0:
        return "top_to_bottom"
    if end_center_y < start_center_y - 5.0:
        return "bottom_to_top"
    return "stationary_or_lateral"


def build_local_identity_packages(
    *,
    local_objects: list[dict[str, Any]],
    representative_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    representatives_by_key = {str(item["local_object_key"]): item for item in representative_frames}
    packages: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for local_object in local_objects:
        local_object_key = str(local_object["local_object_key"])
        representative = representatives_by_key.get(local_object_key, {})
        representative_payload = dict(representative.get("representative_frames", {}))
        downstream_status = str(representative.get("downstream_status", "manual_review"))
        package = {
            "camera_id": local_object["camera_id"],
            "camera_group": local_object["camera_group"],
            "camera_timezone": local_object["camera_timezone"],
            "local_object_id": local_object["local_object_id"],
            "local_object_key": local_object_key,
            "global_object_id": None,
            "object_family": local_object["object_family"],
            "final_class": local_object["final_class"],
            "class_votes": dict(local_object.get("class_votes", {})),
            "source_raw_track_ids": list(local_object.get("source_raw_track_ids", [])),
            "start_timestamp_seconds": local_object["start_timestamp_seconds"],
            "end_timestamp_seconds": local_object["end_timestamp_seconds"],
            "duration_seconds": local_object["duration_seconds"],
            "first_source_frame_index": local_object["first_source_frame_index"],
            "last_source_frame_index": local_object["last_source_frame_index"],
            "entry_boundary": local_object.get("entry_boundary"),
            "exit_boundary": local_object.get("exit_boundary"),
            "motion_direction": _motion_direction(local_object),
            "confirmed": bool(local_object.get("confirmed", False)),
            "quality_level": local_object["quality_level"],
            "quality_score": local_object["quality_score"],
            "track_integrity_status": str(local_object.get("track_integrity_status", "unknown")),
            "timeline_correction_applied": bool(local_object.get("timeline_correction_applied", False)),
            "invalid_observation_count": int(local_object.get("invalid_observation_count", 0) or 0),
            "trimmed_kcf_duration_seconds": round(float(local_object.get("trimmed_kcf_duration_seconds", 0.0) or 0.0), 6),
            "frozen_kcf_detected": bool(local_object.get("frozen_kcf_detected", False)),
            "boundary_stuck_detected": bool(local_object.get("boundary_stuck_detected", False)),
            "representative_crops": [
                item["crop_path"]
                for item in [representative_payload.get("primary"), *list(representative_payload.get("alternatives", []))]
                if isinstance(item, dict) and item.get("crop_path")
            ],
            "representative_full_frames": [
                item["full_frame_path"]
                for item in [representative_payload.get("primary"), *list(representative_payload.get("alternatives", []))]
                if isinstance(item, dict) and item.get("full_frame_path")
            ],
            "plate_candidate_frames": [
                representative_payload["plate_candidate"]["crop_path"]
            ] if isinstance(representative_payload.get("plate_candidate"), dict) and representative_payload["plate_candidate"].get("crop_path") else [],
            "verified_plate": None,
            "possible_plate_text": [],
            "vehicle_color": None,
            "person_attributes": None,
            "appearance_embeddings": [],
            "representative_embedding": None,
            "embedding_model_name": None,
            "trajectory_summary": {
                "start_center_normalized": [],
                "end_center_normalized": [],
                "mean_velocity_normalized": [],
                "direction": _motion_direction(local_object),
            },
            "quality_flags": list(local_object.get("quality_flags", [])),
            "warnings": list(local_object.get("warnings", [])),
            "downstream_status": downstream_status,
            "valid_yolo_crop_count": len([item for item in [representative_payload.get("primary"), *list(representative_payload.get("alternatives", []))] if isinstance(item, dict) and str(item.get("bbox_source")) == "yolo" and bool(item.get("crop_content_valid", True))]),
            "valid_kcf_crop_count": len([item for item in [representative_payload.get("primary"), *list(representative_payload.get("alternatives", []))] if isinstance(item, dict) and str(item.get("bbox_source")) == "kcf" and bool(item.get("crop_content_valid", True))]),
            "invalid_crop_candidate_count": len(list(representative.get("invalid_crop_candidates", []))) if isinstance(representative, dict) else 0,
            "reconciliation_status": "merged" if len(list(local_object.get("source_raw_track_ids", []))) > 1 else "single_segment",
            "manual_review_reasons": list(local_object.get("warnings", [])) if downstream_status == "manual_review" else [],
        }
        packages.append(package)
        flat_rows.append(
            {
                "camera_id": package["camera_id"],
                "local_object_id": package["local_object_id"],
                "local_object_key": package["local_object_key"],
                "object_family": package["object_family"],
                "final_class": package["final_class"],
                "start_timestamp_seconds": package["start_timestamp_seconds"],
                "end_timestamp_seconds": package["end_timestamp_seconds"],
                "duration_seconds": package["duration_seconds"],
                "quality_level": package["quality_level"],
                "quality_score": package["quality_score"],
                "primary_crop_path": representative_payload.get("primary", {}).get("crop_path") if isinstance(representative_payload.get("primary"), dict) else None,
                "primary_full_frame_path": representative_payload.get("primary", {}).get("full_frame_path") if isinstance(representative_payload.get("primary"), dict) else None,
                "plate_candidate_crop_path": representative_payload.get("plate_candidate", {}).get("crop_path") if isinstance(representative_payload.get("plate_candidate"), dict) else None,
                "source_raw_track_ids": list(package["source_raw_track_ids"]),
                "confirmed": package["confirmed"],
                "entry_boundary": package["entry_boundary"],
                "exit_boundary": package["exit_boundary"],
                "motion_direction": package["motion_direction"],
                "downstream_status": downstream_status,
            }
        )
    report = {
        "status": "success",
        "total_packages": len(packages),
        "ready_packages": len([item for item in packages if item["downstream_status"] == "ready"]),
        "fallback_packages": len([item for item in packages if item["downstream_status"] == "fallback"]),
        "manual_review_packages": len([item for item in packages if item["downstream_status"] == "manual_review"]),
        "rejected_packages": len([item for item in packages if item["downstream_status"] == "rejected"]),
        "vehicle_packages": len([item for item in packages if item["object_family"] == "vehicle"]),
        "person_packages": len([item for item in packages if item["object_family"] == "person"]),
        "packages_with_primary_crop": len([item for item in packages if item["representative_crops"]]),
        "packages_with_plate_candidate": len([item for item in packages if item["plate_candidate_frames"]]),
        "packages_missing_crop": len([item for item in packages if not item["representative_crops"]]),
        "packages_by_quality_level": {
            "high": len([item for item in packages if item["quality_level"] == "high"]),
            "medium": len([item for item in packages if item["quality_level"] == "medium"]),
            "low": len([item for item in packages if item["quality_level"] == "low"]),
            "invalid": len([item for item in packages if item["quality_level"] == "invalid"]),
        },
    }
    return packages, flat_rows, report
