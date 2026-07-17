from __future__ import annotations

import math
from statistics import median
from typing import Any


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def normalize_per_minute(value: float | int | None, duration_seconds: float | int | None) -> float:
    if value is None or duration_seconds in {None, 0, 0.0}:
        return 0.0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(numeric_value * 60.0 / float(duration_seconds), 6)


def build_runtime_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    td_tracking = td_case2["timings"].get("tracking_runtime_seconds")
    td_post = (td_case2["timings"].get("step05_runtime_seconds") or 0.0) + (td_case2["timings"].get("step06_runtime_seconds") or 0.0) + (td_case2["timings"].get("step07_runtime_seconds") or 0.0)
    td_total = td_case2.get("adapter_total_runtime_seconds") or td_case2.get("known_stage_runtime_seconds")
    hy_tracking = float(hybrid["tracking_report"]["processing_speed"]["total_runtime_seconds"])
    hy_post = float(hybrid.get("adapter_post_runtime_seconds", 0.0) or 0.0)
    hy_total = float(hybrid.get("adapter_total_runtime_seconds") or (hy_tracking + hy_post))
    source_duration = float(td_case2.get("source_duration_seconds") or hybrid["tracking_report"]["video_metadata"]["duration_seconds"] or 0.0)
    return {
        "td_case2": {
            "tracking_runtime_seconds": td_tracking,
            "post_processing_runtime_seconds": td_post,
            "end_to_end_tracking_plus_crop_runtime_seconds": td_total,
            "realtime_factor": round(safe_divide(td_total, source_duration), 6),
            "average_ms_per_processed_frame": round(safe_divide(td_total * 1000.0, td_case2.get("processed_frames", 0)), 6),
            "average_ms_per_source_video_second": round(safe_divide(td_total * 1000.0, source_duration), 6),
        },
        "hybrid": {
            "tracking_runtime_seconds": hy_tracking,
            "post_processing_runtime_seconds": hy_post,
            "end_to_end_tracking_plus_crop_runtime_seconds": hy_total,
            "realtime_factor": round(safe_divide(hy_total, source_duration), 6),
            "average_ms_per_processed_frame": round(safe_divide(hy_total * 1000.0, len(hybrid["frame_metrics"].get("frames", []))), 6),
            "average_ms_per_source_video_second": round(safe_divide(hy_total * 1000.0, source_duration), 6),
        },
    }


def build_detector_usage_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    source_duration = float(td_case2.get("source_duration_seconds") or hybrid["tracking_report"]["video_metadata"]["duration_seconds"] or 0.0)
    hybrid_report = hybrid["tracking_report"]
    processed_frames_hybrid = len(hybrid["frame_metrics"].get("frames", []))
    yolo_calls_hybrid = int(hybrid_report.get("yolo_call_count", 0) or 0)
    yolo_calls_td = td_case2.get("yolo_calls")
    td_processed_frames = int(td_case2.get("processed_frames", 0) or 0)
    return {
        "td_case2": {
            "source_frame_count": td_case2.get("source_frame_count"),
            "processed_frame_count": td_processed_frames,
            "yolo_calls": yolo_calls_td,
            "yolo_calls_per_source_video_second": "not_available" if yolo_calls_td == "not_available" else round(safe_divide(float(yolo_calls_td), source_duration), 6),
            "percentage_of_processed_frames_using_yolo": "not_available" if yolo_calls_td == "not_available" else round(safe_divide(float(yolo_calls_td) * 100.0, td_processed_frames), 6),
            "detector_call_reduction_vs_yolo_every_processed_frame_percent": "not_available" if yolo_calls_td == "not_available" else round(100.0 - safe_divide(float(yolo_calls_td) * 100.0, td_processed_frames), 6),
            "emergency_detector_calls": "not_available",
            "tracker_only_updates_between_detector_calls": "not_available",
        },
        "hybrid": {
            "source_frame_count": hybrid_report["video_metadata"]["frame_count"],
            "processed_frame_count": processed_frames_hybrid,
            "yolo_calls": yolo_calls_hybrid,
            "yolo_calls_per_source_video_second": round(safe_divide(yolo_calls_hybrid, source_duration), 6),
            "percentage_of_processed_frames_using_yolo": round(safe_divide(yolo_calls_hybrid * 100.0, processed_frames_hybrid), 6),
            "detector_call_reduction_vs_yolo_every_processed_frame_percent": round(100.0 - safe_divide(yolo_calls_hybrid * 100.0, processed_frames_hybrid), 6),
            "emergency_detector_calls": int(hybrid_report.get("emergency_yolo_call_count", 0) or 0),
            "tracker_only_updates_between_detector_calls": int(hybrid_report.get("kcf_update_count", 0) or 0),
        },
    }


def build_tracking_count_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    hybrid_quality = hybrid["quality_report"]
    quality_breakdown = dict(hybrid_quality.get("quality_breakdown", {}))
    return {
        "td_case2": {
            "raw_track_ids": td_case2.get("raw_tracks"),
            "confirmed_raw_tracks": td_case2.get("confirmed_tracks"),
            "reconciled_local_objects": td_case2.get("reconciled_objects"),
            "person_objects": td_case2.get("persons"),
            "vehicle_objects": td_case2.get("vehicles"),
            "accepted_merges": 0,
            "possible_merges": 0,
            "tracks_shorter_than_0_5_seconds": td_case2.get("short_tracks_under_0_5_seconds"),
            "tracks_shorter_than_1_second": td_case2.get("short_tracks_under_1_0_second"),
            "average_duration_seconds": td_case2.get("track_duration_stats", {}).get("avg"),
            "median_duration_seconds": td_case2.get("track_duration_stats", {}).get("median"),
            "maximum_duration_seconds": td_case2.get("track_duration_stats", {}).get("max"),
            "frozen_or_stale_tracks": 0,
            "boundary_stuck_tracks": 0,
            "track_quality_distribution": td_case2.get("track_quality_counts", {}),
        },
        "hybrid": {
            "raw_track_ids": hybrid["tracking_report"].get("raw_track_id_count"),
            "confirmed_raw_tracks": hybrid["tracking_report"].get("confirmed_raw_track_count"),
            "reconciled_local_objects": len(hybrid["reconciled_tracks"].get("tracks", [])),
            "person_objects": hybrid["package_report"].get("person_packages"),
            "vehicle_objects": hybrid["package_report"].get("vehicle_packages"),
            "accepted_merges": len(hybrid["merge_events"].get("events", [])),
            "possible_merges": max(0, len(hybrid["candidates"].get("candidates", [])) - len(hybrid["merge_events"].get("events", []))),
            "tracks_shorter_than_0_5_seconds": _count_quality_short(hybrid_quality.get("tracks", []), 0.5),
            "tracks_shorter_than_1_second": _count_quality_short(hybrid_quality.get("tracks", []), 1.0),
            "average_duration_seconds": hybrid["tracking_report"].get("track_durations", {}).get("mean"),
            "median_duration_seconds": hybrid["tracking_report"].get("track_durations", {}).get("median"),
            "maximum_duration_seconds": hybrid["tracking_report"].get("track_durations", {}).get("max"),
            "frozen_or_stale_tracks": _count_boolean(hybrid["quality_report"].get("tracks", []), "frozen_kcf_detected"),
            "boundary_stuck_tracks": _count_boolean(hybrid["quality_report"].get("tracks", []), "boundary_stuck_detected"),
            "track_quality_distribution": quality_breakdown,
        },
    }


def build_crop_quality_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    hybrid_representative = hybrid["representative_report"]
    return {
        "td_case2": {
            "objects_with_primary_crop": td_case2.get("tracks_with_primary_crop"),
            "objects_with_three_representative_crops": td_case2.get("tracks_with_three_representative_crops"),
            "objects_with_full_scene_frame": td_case2.get("objects_with_full_scene_frame"),
            "fallback_crop_count": td_case2.get("fallback_crop_count"),
            "invalid_crop_candidates": td_case2.get("invalid_crop_candidates"),
            "crop_failures": td_case2.get("crop_failures"),
            "yolo_selected_crops": td_case2.get("yolo_selected_crops"),
            "kcf_selected_crops": td_case2.get("kcf_selected_crops"),
            "plate_candidate_count": td_case2.get("plate_candidates"),
        },
        "hybrid": {
            "objects_with_primary_crop": hybrid_representative.get("valid_primary_crops"),
            "objects_with_three_representative_crops": len([item for item in hybrid["packages"].get("packages", []) if len(list(item.get("representative_crops", []))) >= 3]),
            "objects_with_full_scene_frame": len([item for item in hybrid["packages"].get("packages", []) if list(item.get("representative_full_frames", []))]),
            "fallback_crop_count": hybrid_representative.get("fallback_crops"),
            "invalid_crop_candidates": len(hybrid["invalid_crop_candidates"].get("candidates", [])),
            "crop_failures": int(hybrid_representative.get("crop_failures", 0) or 0),
            "yolo_selected_crops": len([item for item in hybrid["packages"].get("packages", []) if int(item.get("valid_yolo_crop_count", 0) or 0) > 0]),
            "kcf_selected_crops": len([item for item in hybrid["packages"].get("packages", []) if int(item.get("valid_kcf_crop_count", 0) or 0) > 0]),
            "plate_candidate_count": hybrid_representative.get("objects_with_plate_candidate"),
        },
    }


def build_class_count_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    td_counts = dict(td_case2.get("class_counts", {}))
    hybrid_counts = {}
    for package in hybrid["packages"].get("packages", []):
        name = str(package.get("final_class", "unknown"))
        hybrid_counts[name] = int(hybrid_counts.get(name, 0)) + 1
    return {"td_case2": td_counts, "hybrid": hybrid_counts}


def build_failure_comparison(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    return {
        "td_case2": {
            "failures": td_case2.get("failures", []),
            "warnings": td_case2.get("warnings", []),
        },
        "hybrid": {
            "failures": hybrid.get("failures", {}),
            "warnings": hybrid["tracking_report"].get("warnings", []),
        },
    }


def build_manual_review_summary(hybrid: dict[str, Any]) -> dict[str, Any]:
    progress = hybrid.get("manual_review_progress", {})
    summary = hybrid.get("manual_review_summary", {})
    if not progress and not summary:
        return {"status": "not_found"}
    reviewed_count = int(progress.get("reviewed_objects", summary.get("reviewed_local_objects", 0)) or 0)
    total_objects = int(progress.get("total_objects", summary.get("total_local_objects", 0)) or 0)
    return {
        "status": "found",
        "reviewed_object_count": reviewed_count,
        "unreviewed_object_count": max(0, total_objects - reviewed_count),
        "reviewed_duplicates": int(summary.get("duplicate_tracks", 0) or 0),
        "reviewed_false_detections": int(summary.get("false_detections", 0) or 0),
        "reviewed_crop_preferences": {
            "good_primary_crops": int(summary.get("good_primary_crops", 0) or 0),
            "alternative_crops_preferred": int(summary.get("alternative_crops_preferred", 0) or 0),
            "all_crops_bad": int(summary.get("all_crops_bad", 0) or 0),
        },
    }


def build_config_differences(td_case2_artifacts: dict[str, Any], hybrid_artifacts: dict[str, Any], camera_id: str, camera_group: str, camera_timezone: str) -> dict[str, Any]:
    td_env = td_case2_artifacts["config_snapshot"]["env_overrides"]
    hy_env = hybrid_artifacts["config_snapshot"]["env_overrides"]
    differences: list[dict[str, Any]] = []
    if td_env.get("TD_CASE2_YOLO_MODEL_PATH") != hy_env.get("TD_CASE2_YOLO_MODEL_PATH"):
        differences.append({"setting": "yolo_model_path", "td_case2": td_env.get("TD_CASE2_YOLO_MODEL_PATH"), "hybrid": hy_env.get("TD_CASE2_YOLO_MODEL_PATH"), "impact": "potential detector-model mismatch"})
    differences.append({"setting": "tracker_algorithm", "td_case2": td_case2_artifacts["metrics"].get("tracker_name"), "hybrid": hybrid_artifacts["metrics"].get("tracking_report", {}).get("configuration", {}).get("visual_tracker", "KCF"), "impact": "tracking logic is not identical"})
    differences.append({"setting": "camera_group", "td_case2": "not_explicitly_propagated", "hybrid": camera_group, "impact": "metadata mismatch recorded only, not hidden"})
    differences.append({"setting": "camera_timezone", "td_case2": "not_explicitly_propagated", "hybrid": camera_timezone, "impact": "metadata mismatch recorded only, not hidden"})
    differences.append({"setting": "camera_id", "td_case2": "not_explicitly_propagated", "hybrid": camera_id, "impact": "metadata mismatch recorded only, not hidden"})
    differences.append({"setting": "frame_sampling", "td_case2": td_case2_artifacts["metrics"]["processed_frames"], "hybrid": len(hybrid_artifacts["metrics"]["frame_metrics"].get("frames", [])), "impact": "processed frame counts may differ"})
    return {
        "status": "success",
        "differences": differences,
        "fairness_warnings": [
            "The two pipelines use different tracker algorithms.",
            "The two pipelines may process different frame counts and therefore need normalized metrics.",
            "td_case2 does not expose emergency-detector-call metrics comparable to the hybrid tracker.",
            "Hybrid includes additional post-tracking cleanup stages that the baseline td_case2 pipeline does not.",
        ],
    }


def build_normalized_metrics(td_case2: dict[str, Any], hybrid: dict[str, Any], runtime_comparison: dict[str, Any]) -> dict[str, Any]:
    td_duration = float(td_case2.get("source_duration_seconds", 0.0) or 0.0)
    hy_duration = float(hybrid["tracking_report"]["video_metadata"]["duration_seconds"])
    hy_reconciled = len(hybrid["reconciled_tracks"].get("tracks", []))
    return {
        "td_case2": {
            "raw_tracks_per_minute": normalize_per_minute(td_case2.get("raw_tracks"), td_duration),
            "confirmed_tracks_per_minute": normalize_per_minute(td_case2.get("confirmed_tracks"), td_duration),
            "reconciled_objects_per_minute": normalize_per_minute(td_case2.get("reconciled_objects") or 0, td_duration),
            "short_tracks_per_minute": normalize_per_minute(td_case2.get("short_tracks_under_1_0_second"), td_duration),
            "yolo_calls_per_minute": 0.0 if td_case2.get("yolo_calls") == "not_available" else normalize_per_minute(td_case2.get("yolo_calls"), td_duration),
            "runtime_per_video_minute": round(safe_divide(float(runtime_comparison["td_case2"]["end_to_end_tracking_plus_crop_runtime_seconds"] or 0.0), td_duration / 60.0 if td_duration else 0.0), 6),
            "crops_per_reconciled_object": 0.0,
        },
        "hybrid": {
            "raw_tracks_per_minute": normalize_per_minute(hybrid["tracking_report"].get("raw_track_id_count"), hy_duration),
            "confirmed_tracks_per_minute": normalize_per_minute(hybrid["tracking_report"].get("confirmed_raw_track_count"), hy_duration),
            "reconciled_objects_per_minute": normalize_per_minute(hy_reconciled, hy_duration),
            "short_tracks_per_minute": normalize_per_minute(_count_quality_short(hybrid["quality_report"].get("tracks", []), 1.0), hy_duration),
            "yolo_calls_per_minute": normalize_per_minute(hybrid["tracking_report"].get("yolo_call_count"), hy_duration),
            "runtime_per_video_minute": round(safe_divide(float(runtime_comparison["hybrid"]["end_to_end_tracking_plus_crop_runtime_seconds"] or 0.0), hy_duration / 60.0 if hy_duration else 0.0), 6),
            "crops_per_reconciled_object": round(safe_divide(sum(len(list(item.get("representative_crops", []))) for item in hybrid["packages"].get("packages", [])), hy_reconciled), 6),
        },
    }


def build_approximate_cross_pipeline_matches(td_case2: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    td_tracks = list(td_case2.get("best_frames", {}).get("tracks", []))
    hybrid_tracks = list(hybrid.get("reconciled_tracks", {}).get("tracks", []))
    matches: list[dict[str, Any]] = []
    for td_track in td_tracks:
        td_class = str(td_track.get("dominant_class_name", ""))
        td_family = str(td_track.get("track_type", ""))
        td_start = float(td_track.get("selected_detections", [{}])[0].get("timestamp_seconds", td_track.get("duration_seconds", 0.0)) or 0.0)
        td_end = float(td_track.get("selected_detections", [{}])[-1].get("timestamp_seconds", td_start) or td_start)
        td_center = _track_center_from_td(td_track)
        best_match = None
        best_score = -1.0
        for hy_track in hybrid_tracks:
            hy_class = str(hy_track.get("final_class", ""))
            hy_family = str(hy_track.get("object_family", ""))
            if td_family and hy_family and td_family != hy_family:
                continue
            hy_start = float(hy_track.get("start_timestamp_seconds", 0.0) or 0.0)
            hy_end = float(hy_track.get("end_timestamp_seconds", 0.0) or 0.0)
            overlap = max(0.0, min(td_end, hy_end) - max(td_start, hy_start))
            union = max(td_end, hy_end) - min(td_start, hy_start)
            time_overlap = safe_divide(overlap, union) if union > 0 else 0.0
            hy_center = _track_center_from_hybrid(hy_track)
            spatial_similarity = max(0.0, 1.0 - _normalized_center_distance(td_center, hy_center))
            class_match = td_class == hy_class
            score = (0.45 * time_overlap) + (0.35 * spatial_similarity) + (0.20 * (1.0 if class_match else 0.0))
            if score > best_score:
                best_score = score
                best_match = {
                    "td_case2_local_object_id": td_track.get("track_id"),
                    "hybrid_local_object_id": hy_track.get("local_object_id"),
                    "time_overlap": round(time_overlap, 6),
                    "class_match": class_match,
                    "spatial_similarity": round(spatial_similarity, 6),
                    "match_confidence": round(score, 6),
                    "review_required": score < 0.6,
                }
        if best_match is not None:
            matches.append(best_match)
    return {"status": "success", "matches": matches}


def _track_center_from_td(track: dict[str, Any]) -> tuple[float, float]:
    detections = list(track.get("selected_detections", [])) or list(track.get("detections", []))
    if not detections:
        return (0.0, 0.0)
    bbox = detections[0].get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _track_center_from_hybrid(track: dict[str, Any]) -> tuple[float, float]:
    trajectory = list(track.get("combined_trajectory", []))
    if not trajectory:
        return (0.0, 0.0)
    bbox = trajectory[0].get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _normalized_center_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return min(1.0, math.sqrt((dx * dx) + (dy * dy)) / 1500.0)


def _count_boolean(rows: list[dict[str, Any]], key: str) -> int:
    return len([item for item in rows if bool(item.get(key))])


def _count_quality_short(rows: list[dict[str, Any]], threshold: float) -> int:
    count = 0
    for item in rows:
        duration = float(item.get("duration_seconds", 0.0) or 0.0)
        if duration < threshold:
            count += 1
    return count


def _metric_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def choose_better_smaller(td_value: float | int | None, hy_value: float | int | None) -> str:
    if td_value is None or hy_value is None:
        return "inconclusive"
    if math.isclose(float(td_value), float(hy_value), rel_tol=0.05, abs_tol=1e-6):
        return "approximately_equal"
    return "td_case2_better" if float(td_value) < float(hy_value) else "hybrid_better"


def choose_better_larger(td_value: float | int | None, hy_value: float | int | None) -> str:
    if td_value is None or hy_value is None:
        return "inconclusive"
    if math.isclose(float(td_value), float(hy_value), rel_tol=0.05, abs_tol=1e-6):
        return "approximately_equal"
    return "td_case2_better" if float(td_value) > float(hy_value) else "hybrid_better"


def build_decisions(runtime_comparison: dict[str, Any], detector_comparison: dict[str, Any], tracking_comparison: dict[str, Any], crop_comparison: dict[str, Any]) -> dict[str, Any]:
    decisions = {
        "speed_winner": {
            "decision": choose_better_smaller(
                runtime_comparison["td_case2"]["end_to_end_tracking_plus_crop_runtime_seconds"],
                runtime_comparison["hybrid"]["end_to_end_tracking_plus_crop_runtime_seconds"],
            ),
            "evidence": runtime_comparison,
        },
        "detector_efficiency_winner": {
            "decision": "inconclusive" if detector_comparison["td_case2"]["yolo_calls"] == "not_available" else choose_better_smaller(detector_comparison["td_case2"]["yolo_calls"], detector_comparison["hybrid"]["yolo_calls"]),
            "evidence": detector_comparison,
        },
        "track_continuity_winner": {
            "decision": choose_better_smaller(tracking_comparison["td_case2"]["tracks_shorter_than_1_second"], tracking_comparison["hybrid"]["tracks_shorter_than_1_second"]),
            "evidence": tracking_comparison,
        },
        "fragmentation_risk_winner": {
            "decision": choose_better_smaller(tracking_comparison["td_case2"]["raw_track_ids"], tracking_comparison["hybrid"]["raw_track_ids"]),
            "evidence": tracking_comparison,
        },
        "crop_quality_winner": {
            "decision": choose_better_larger(crop_comparison["td_case2"]["objects_with_primary_crop"], crop_comparison["hybrid"]["objects_with_primary_crop"]),
            "evidence": crop_comparison,
        },
        "operational_safety_winner": {
            "decision": choose_better_smaller(tracking_comparison["td_case2"]["frozen_or_stale_tracks"], tracking_comparison["hybrid"]["frozen_or_stale_tracks"]),
            "evidence": tracking_comparison,
        },
        "better_base_for_ocr_and_colour": {
            "decision": choose_better_larger(crop_comparison["td_case2"]["plate_candidate_count"], crop_comparison["hybrid"]["plate_candidate_count"]),
            "evidence": crop_comparison,
        },
        "better_base_for_future_multi_camera_tracking": {
            "decision": choose_better_smaller(tracking_comparison["td_case2"]["tracks_shorter_than_1_second"], tracking_comparison["hybrid"]["tracks_shorter_than_1_second"]),
            "evidence": tracking_comparison,
        },
    }
    return decisions
