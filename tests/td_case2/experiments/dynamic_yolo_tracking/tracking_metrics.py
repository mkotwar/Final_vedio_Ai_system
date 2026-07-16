from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.step04b_bytetrack.tracking_metrics import (
    build_step05_compatible_tracks,
    validate_step05_compatibility,
)


def summarize_tracks(
    *,
    mode_name: str,
    total_video_frames: int,
    frames_processed: int,
    yolo_seconds: float,
    tracker_seconds: float,
    merge_seconds: float,
    yolo_payload: dict[str, Any],
    raw_tracks_payload: dict[str, Any],
    merged_tracks_payload: dict[str, Any],
    bytetrack_meta: dict[str, Any],
) -> dict[str, Any]:
    vehicle_tracks = [item for item in list(merged_tracks_payload.get("tracks", [])) if item.get("track_type") == "vehicle"]
    track_quality_counts = Counter(str(item.get("track_quality", "")) for item in vehicle_tracks)
    detections_per_track = [int(item.get("detection_count", 0) or 0) for item in vehicle_tracks]
    durations = [float(item.get("duration_seconds", 0.0) or 0.0) for item in vehicle_tracks]
    longest = max(vehicle_tracks, key=lambda item: item.get("detection_count", 0), default=None)
    processed_pct = round((frames_processed / total_video_frames) * 100.0, 3) if total_video_frames > 0 else 0.0
    return {
        "mode_name": mode_name,
        "total_video_frames": total_video_frames,
        "frames_processed_by_yolo": frames_processed,
        "processed_frame_percentage": processed_pct,
        "yolo_processing_seconds": round(yolo_seconds, 3),
        "tracker_processing_seconds": round(tracker_seconds, 3),
        "fragment_merging_seconds": round(merge_seconds, 3),
        "total_processing_seconds": round(yolo_seconds + tracker_seconds + merge_seconds, 3),
        "detections_by_class": dict(sorted(yolo_payload.get("class_counts", {}).items())),
        "raw_track_count": len([item for item in list(raw_tracks_payload.get("tracks", [])) if item.get("track_type") == "vehicle"]),
        "post_merge_track_count": len(vehicle_tracks),
        "single_frame_tracks": track_quality_counts.get("single_frame", 0),
        "fragmented_tracks": track_quality_counts.get("fragmented", 0),
        "good_tracks": track_quality_counts.get("good", 0),
        "average_track_duration_seconds": round(sum(durations) / len(durations), 6) if durations else 0.0,
        "average_detections_per_track": round(sum(detections_per_track) / len(detections_per_track), 6) if detections_per_track else 0.0,
        "longest_track_id": None if longest is None else longest.get("track_id"),
        "longest_track_detections": 0 if longest is None else int(longest.get("detection_count", 0)),
        "longest_track_duration_seconds": 0.0 if longest is None else float(longest.get("duration_seconds", 0.0)),
        "lost_track_recoveries": int(bytetrack_meta.get("vehicle_refind_count", 0)),
        "step05_compatibility": validate_step05_compatibility(merged_tracks_payload),
    }


def compare_mode_summaries(*summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "modes": list(summaries),
        "frame_savings_vs_fixed_5fps": _frame_savings_vs_reference(summaries, reference_mode="fixed_5fps", target_mode="dynamic"),
    }


def _frame_savings_vs_reference(summaries: tuple[dict[str, Any], ...], *, reference_mode: str, target_mode: str) -> dict[str, Any]:
    lookup = {str(item.get("mode_name")): item for item in summaries}
    reference = lookup.get(reference_mode)
    target = lookup.get(target_mode)
    if reference is None or target is None:
        return {"available": False}
    saved = int(reference["frames_processed_by_yolo"]) - int(target["frames_processed_by_yolo"])
    return {
        "available": True,
        "reference_mode": reference_mode,
        "target_mode": target_mode,
        "frames_saved": saved,
        "percent_saved": round((saved / max(1, int(reference["frames_processed_by_yolo"]))) * 100.0, 3),
    }


__all__ = [
    "build_step05_compatible_tracks",
    "validate_step05_compatibility",
    "summarize_tracks",
    "compare_mode_summaries",
]

