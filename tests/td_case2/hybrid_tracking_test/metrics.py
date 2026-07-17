from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from .data_models import HybridTrack


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_timing_report(
    *,
    video_duration_seconds: float,
    total_runtime_seconds: float,
    processed_frame_count: int,
    yolo_call_count: int,
    kcf_update_count: int,
    timing_lists: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "video_duration_seconds": round(float(video_duration_seconds), 6),
        "total_runtime_seconds": round(float(total_runtime_seconds), 6),
        "realtime_factor": round(float(total_runtime_seconds) / max(float(video_duration_seconds), 1e-6), 6),
        "effective_processed_fps": round(float(processed_frame_count) / max(float(total_runtime_seconds), 1e-6), 6),
        "processed_frame_count": int(processed_frame_count),
        "yolo_call_count": int(yolo_call_count),
        "kcf_update_count": int(kcf_update_count),
        "average_yolo_inference_ms": round(safe_mean(timing_lists.get("yolo_inference_ms", [])), 6),
        "average_kcf_update_ms_per_track": round(safe_mean(timing_lists.get("kcf_update_ms_per_track", [])), 6),
        "decode_time_ms_total": round(sum(timing_lists.get("decode_time_ms", [])), 6),
        "frame_preprocessing_time_ms_total": round(sum(timing_lists.get("preprocess_time_ms", [])), 6),
        "association_time_ms_total": round(sum(timing_lists.get("association_time_ms", [])), 6),
        "kcf_initialization_time_ms_total": round(sum(timing_lists.get("kcf_initialization_time_ms", [])), 6),
        "kcf_update_time_ms_total": round(sum(timing_lists.get("kcf_update_time_ms", [])), 6),
        "motion_trigger_time_ms_total": round(sum(timing_lists.get("motion_trigger_time_ms", [])), 6),
        "visualization_time_ms_total": round(sum(timing_lists.get("visualization_time_ms", [])), 6),
        "json_writing_time_ms_total": round(sum(timing_lists.get("json_writing_time_ms", [])), 6),
    }


def build_track_summary(tracks: list[HybridTrack]) -> dict[str, Any]:
    summaries = [track.to_summary_dict() for track in tracks]
    return {
        "status": "success",
        "track_count": len(summaries),
        "tracks": summaries,
    }


def build_main_report(
    *,
    config: dict[str, Any],
    video_metadata: dict[str, Any],
    frame_metrics: list[dict[str, Any]],
    track_manager,
    event_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    timing_report: dict[str, Any],
) -> dict[str, Any]:
    trigger_reason_counts: Counter[str] = Counter()
    simultaneous_tracks: list[int] = []
    detection_gaps: list[float] = []
    for frame_item in frame_metrics:
        simultaneous_tracks.append(int(frame_item.get("active_track_count", 0)))
        trigger_reason_counts.update(list(frame_item.get("yolo_trigger_reasons", [])))
        for track in list(frame_item.get("tracks", [])):
            detection_gaps.append(float(track.get("seconds_since_detection", 0.0) or 0.0))
    tracks = track_manager.all_tracks()
    durations = [float(track.last_update_timestamp_seconds - track.created_timestamp_seconds) for track in tracks]
    short_under_half = [track for track in tracks if float(track.last_update_timestamp_seconds - track.created_timestamp_seconds) < 0.5]
    short_under_one = [track for track in tracks if float(track.last_update_timestamp_seconds - track.created_timestamp_seconds) < 1.0]
    one_hit_tracks = [track for track in tracks if int(track.detection_hits) <= 1]
    missed_refresh_tracks = [track for track in tracks if str(track.termination_reason) == "missed_refresh_limit"]
    raw_tracks_by_class = Counter(str(track.class_name) for track in tracks)
    raw_confirmed_by_class = Counter(str(track.class_name) for track in tracks if bool(track.is_confirmed))
    short_car_fragments = len(
        [
            track
            for track in tracks
            if str(track.class_name) == "car" and float(track.last_update_timestamp_seconds - track.created_timestamp_seconds) < 1.0
        ]
    )
    return {
        "status": "success" if not failures else "completed_with_warnings",
        "video_metadata": video_metadata,
        "configuration": config,
        "processed_frames": len(frame_metrics),
        "raw_track_id_count": len(tracks),
        "confirmed_raw_track_count": len([track for track in tracks if bool(track.is_confirmed)]),
        "tentative_raw_track_count": len([track for track in tracks if not bool(track.is_confirmed)]),
        "reactivated_track_count": int(track_manager.counters.get("tracks_reactivated", 0)),
        "reconciled_track_count": None,
        "reconciled_confirmed_object_count": None,
        "reconciled_objects_by_class": None,
        "fragment_reduction_percent": None,
        "yolo_call_count": int(track_manager.counters.get("yolo_call_count", 0)),
        "scheduled_yolo_call_count": int(track_manager.counters.get("scheduled_yolo_call_count", 0)),
        "emergency_yolo_call_count": int(track_manager.counters.get("emergency_yolo_call_count", 0)),
        "motion_triggered_yolo_call_count": int(track_manager.counters.get("motion_triggered_yolo_call_count", 0)),
        "empty_scene_yolo_call_count": int(track_manager.counters.get("empty_scene_yolo_call_count", 0)),
        "kcf_failure_yolo_call_count": int(track_manager.counters.get("kcf_failure_yolo_call_count", 0)),
        "overlap_yolo_call_count": int(track_manager.counters.get("overlap_yolo_call_count", 0)),
        "box_validation_yolo_call_count": int(track_manager.counters.get("box_validation_yolo_call_count", 0)),
        "yolo_calls_by_trigger_reason": dict(sorted(trigger_reason_counts.items())),
        "kcf_update_count": int(track_manager.counters.get("kcf_update_count", 0)),
        "kcf_initialization_count": int(track_manager.counters.get("kcf_initialization_count", 0)),
        "kcf_reinitialization_count": int(track_manager.counters.get("kcf_reinitialization_count", 0)),
        "kcf_failure_count": int(track_manager.counters.get("kcf_failure_count", 0)),
        "invalid_box_count": int(track_manager.counters.get("invalid_box_count", 0)),
        "tracks_created": int(track_manager.counters.get("tracks_created", 0)),
        "tracks_confirmed": int(track_manager.counters.get("tracks_confirmed", 0)),
        "tracks_removed": int(track_manager.counters.get("tracks_removed", 0)),
        "maximum_simultaneous_tracks": max(simultaneous_tracks, default=0),
        "average_simultaneous_tracks": round(safe_mean([float(value) for value in simultaneous_tracks]), 6),
        "track_durations": {
            "min": round(min(durations), 6) if durations else 0.0,
            "mean": round(safe_mean(durations), 6),
            "median": round(float(median(durations)), 6) if durations else 0.0,
            "max": round(max(durations), 6) if durations else 0.0,
        },
        "average_detection_gap": round(safe_mean(detection_gaps), 6),
        "maximum_detection_gap": round(max(detection_gaps, default=0.0), 6),
        "processing_speed": timing_report,
        "yolo_execution_ratio": round(int(track_manager.counters.get("yolo_call_count", 0)) / max(len(frame_metrics), 1), 6),
        "yolo_reduction_percent": round((1.0 - (int(track_manager.counters.get("yolo_call_count", 0)) / max(len(frame_metrics), 1))) * 100.0, 6),
        "warnings": [item["message"] for item in failures if item.get("severity") != "error"],
        "errors": [item["message"] for item in failures if item.get("severity") == "error"],
        "event_count": len(event_records),
        "raw_tracks_by_class": dict(sorted(raw_tracks_by_class.items())),
        "confirmed_raw_tracks_by_class": dict(sorted(raw_confirmed_by_class.items())),
        "duplication_diagnostics": {
            "tracks_shorter_than_0_5_seconds": len(short_under_half),
            "tracks_shorter_than_1_0_second": len(short_under_one),
            "tracks_with_only_one_detector_hit": len(one_hit_tracks),
            "tracks_ending_because_of_missed_refresh_limit": len(missed_refresh_tracks),
            "tracks_reactivated": int(track_manager.counters.get("tracks_reactivated", 0)),
            "cars": {
                "raw_car_track_ids": int(raw_tracks_by_class.get("car", 0)),
                "confirmed_car_track_ids": int(raw_confirmed_by_class.get("car", 0)),
                "short_car_fragments_under_1_second": int(short_car_fragments),
                "reactivated_car_tracks": len([track for track in tracks if str(track.class_name) == "car" and int(track.reactivation_count) > 0]),
            },
        },
    }


def build_acceptance_assessment(
    *,
    report_payload: dict[str, Any],
    comparison_payload: dict[str, Any] | None,
) -> dict[str, str]:
    assessments = {
        "yolo_calls_reduced_by_at_least_40_percent": "requires_manual_review",
        "no_significant_loss_of_visible_object_coverage": "requires_manual_review",
        "new_object_detection_delay_below_0_5_seconds": "requires_manual_review",
        "track_fragmentation_not_significantly_worse": "requires_manual_review",
        "bounding_boxes_visually_stable": "requires_manual_review",
        "runtime_improves_meaningfully": "requires_manual_review",
        "downstream_crop_extraction_would_remain_usable": "requires_manual_review",
    }
    if report_payload.get("yolo_reduction_percent", 0.0) >= 40.0:
        assessments["yolo_calls_reduced_by_at_least_40_percent"] = "passed"
    else:
        assessments["yolo_calls_reduced_by_at_least_40_percent"] = "failed"
    if comparison_payload:
        if comparison_payload.get("heuristics", {}).get("hybrid_short_lived_tracks", 0) <= comparison_payload.get("heuristics", {}).get("baseline_short_lived_tracks", 0):
            assessments["track_fragmentation_not_significantly_worse"] = "passed"
        if comparison_payload.get("hybrid", {}).get("frame_coverage_ratio", 0.0) > 0.0:
            assessments["no_significant_loss_of_visible_object_coverage"] = "requires_manual_review"
    return assessments
