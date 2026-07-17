from __future__ import annotations

from collections import Counter
from typing import Any


def build_runtime_report(*, video_duration_seconds: float, processed_frames: int, tracking_runtime_seconds: float, cleanup_runtime_seconds: float, crop_runtime_seconds: float) -> dict[str, Any]:
    total_runtime_seconds = tracking_runtime_seconds + cleanup_runtime_seconds + crop_runtime_seconds
    return {
        "status": "success",
        "tracking_runtime_seconds": round(tracking_runtime_seconds, 6),
        "cleanup_runtime_seconds": round(cleanup_runtime_seconds, 6),
        "crop_runtime_seconds": round(crop_runtime_seconds, 6),
        "total_runtime_seconds": round(total_runtime_seconds, 6),
        "realtime_factor": round(total_runtime_seconds / max(video_duration_seconds, 1e-6), 6),
        "average_ms_per_processed_frame": round((total_runtime_seconds * 1000.0) / max(processed_frames, 1), 6),
    }


def build_detector_report(*, processed_frames: int, detector_calls: list[dict[str, Any]], source_duration_seconds: float) -> dict[str, Any]:
    state_counts = Counter(str(item["scheduler_state"]) for item in detector_calls)
    emergency_calls = len([item for item in detector_calls if str(item["scheduler_state"]) == "emergency"])
    return {
        "status": "success",
        "processed_frames": processed_frames,
        "total_yolo_calls": len(detector_calls),
        "scheduled_yolo_calls": len(detector_calls) - emergency_calls,
        "emergency_yolo_calls": emergency_calls,
        "dense_state_calls": int(state_counts.get("dense", 0)),
        "normal_state_calls": int(state_counts.get("normal", 0)),
        "sparse_state_calls": int(state_counts.get("sparse", 0)),
        "idle_state_calls": int(state_counts.get("idle", 0)),
        "yolo_calls_per_second": round(len(detector_calls) / max(source_duration_seconds, 1e-6), 6),
        "detector_reduction_vs_every_processed_frame_percent": round((1.0 - (len(detector_calls) / max(processed_frames, 1))) * 100.0, 6),
    }


def build_crop_report(*, representative_report: dict[str, Any], identity_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "success",
        "primary_crops": int(representative_report.get("valid_primary_crops", 0)),
        "three_frame_sets": len([1 for _ in range(int(representative_report.get("valid_primary_crops", 0)))]) if representative_report.get("valid_primary_crops") else 0,
        "plate_candidates": int(representative_report.get("objects_with_plate_candidate", 0)),
        "crop_failures": int(representative_report.get("crop_failures", 0)),
        "ready_packages": int(identity_report.get("ready_packages", 0)),
        "fallback_packages": int(identity_report.get("fallback_packages", 0)),
        "manual_review_packages": int(identity_report.get("manual_review_packages", 0)),
        "rejected_packages": int(identity_report.get("rejected_packages", 0)),
    }

