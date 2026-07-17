from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SanitizationConfig:
    maximum_supported_kcf_gap_seconds: float = 0.3
    maximum_fallback_kcf_gap_seconds: float = 0.6


def _bbox_valid(bbox_xyxy: list[float]) -> bool:
    if len(bbox_xyxy) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return x2 > x1 and y2 > y1


def sanitize_track_timeline(
    track: dict[str, Any],
    drift_report: dict[str, Any],
    *,
    config: SanitizationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_timeline = list(track.get("rebuilt_timeline", track.get("trajectory", [])))
    segment_by_source_index: dict[int, dict[str, Any]] = {}
    for segment in list(drift_report.get("segments", [])):
        for source_frame_index in list(segment.get("observation_indexes", [])):
            segment_by_source_index[int(source_frame_index)] = segment
    sanitized_timeline: list[dict[str, Any]] = []
    sanitization_events: list[dict[str, Any]] = []
    valid_or_supported: list[dict[str, Any]] = []
    valid_crop_candidates = 0
    invalid_count = 0
    valid_yolo_count = 0
    valid_kcf_count = 0
    invalid_kcf_count = 0
    fallback_count = 0
    for observation in raw_timeline:
        bbox = list(observation.get("bbox_xyxy", []))
        bbox_valid = _bbox_valid(bbox)
        validation = dict(observation.get("validation", {}))
        validation_valid = bool(validation.get("valid", True))
        segment = segment_by_source_index.get(int(observation.get("source_frame_index", 0) or 0))
        segment_flags = set(segment.get("segment_flags", [])) if segment else set()
        seconds_since_yolo = float(observation.get("seconds_since_detection", 0.0) or 0.0)
        reasons: list[str] = []
        validity = "invalid"
        bbox_source = str(observation.get("bbox_source", ""))
        if not bbox_valid:
            reasons.append("invalid_bbox")
        if not validation_valid:
            reasons.append("timeline_inconsistency")
        if bbox_source == "yolo":
            if bbox_valid and validation_valid:
                validity = "valid"
                reasons.append("recent_yolo_confirmation")
        else:
            if not bool(observation.get("kcf_success", False)):
                reasons.append("kcf_failed")
            if "frozen_kcf_box" in segment_flags:
                reasons.append("frozen_kcf_box")
            if "boundary_stuck_box" in segment_flags:
                reasons.append("boundary_stuck_box")
            if "detector_kcf_disagreement" in segment_flags:
                reasons.append("detector_kcf_disagreement")
            if "stale_kcf_propagation" in segment_flags:
                reasons.append("stale_kcf_propagation")
            if "long_kcf_only_segment" in segment_flags:
                reasons.append("long_kcf_only_gap")
            if bbox_valid and validation_valid and bool(observation.get("kcf_success", False)) and not any(
                reason in reasons
                for reason in {
                    "frozen_kcf_box",
                    "boundary_stuck_box",
                    "detector_kcf_disagreement",
                    "stale_kcf_propagation",
                }
            ):
                if seconds_since_yolo <= config.maximum_supported_kcf_gap_seconds:
                    validity = "supported"
                    reasons.append("valid_short_kcf_bridge")
                elif seconds_since_yolo <= config.maximum_fallback_kcf_gap_seconds:
                    validity = "fallback"
                    reasons.append("fallback_kcf_bridge")
                else:
                    reasons.append("long_kcf_only_gap")
        annotated = {
            **dict(observation),
            "observation_validity": validity,
            "observation_validity_reasons": sorted(set(reasons)),
            "drift_segment_flags": sorted(segment_flags),
        }
        sanitized_timeline.append(annotated)
        if validity in {"valid", "supported"}:
            valid_or_supported.append(annotated)
            valid_crop_candidates += 1
            if bbox_source == "yolo":
                valid_yolo_count += 1
            else:
                valid_kcf_count += 1
        elif validity == "fallback":
            fallback_count += 1
        else:
            invalid_count += 1
            if bbox_source == "kcf":
                invalid_kcf_count += 1
    if valid_or_supported:
        sanitized_start = round(float(valid_or_supported[0]["timestamp_seconds"]), 6)
        sanitized_end = round(float(valid_or_supported[-1]["timestamp_seconds"]), 6)
        sanitized_duration = round(max(0.0, sanitized_end - sanitized_start), 6)
        sanitized_first_frame = int(valid_or_supported[0]["source_frame_index"])
        sanitized_last_frame = int(valid_or_supported[-1]["source_frame_index"])
    else:
        sanitized_start = None
        sanitized_end = None
        sanitized_duration = 0.0
        sanitized_first_frame = None
        sanitized_last_frame = None
    last_valid_yolo_timestamp = None
    for item in valid_or_supported:
        if str(item.get("bbox_source")) == "yolo":
            last_valid_yolo_timestamp = round(float(item["timestamp_seconds"]), 6)
    integrity_status = "usable"
    if not valid_or_supported:
        integrity_status = "invalid"
    elif valid_yolo_count <= 1:
        integrity_status = "weak_single_detection"
    elif valid_yolo_count == 0 and fallback_count > 0:
        integrity_status = "fallback_only"
    original_end = track.get("actual_end_timestamp_seconds")
    if sanitized_end is not None and original_end is not None and float(sanitized_end) < float(original_end):
        sanitization_events.append(
            {
                "event_type": "invalid_kcf_tail_trimmed",
                "track_id": int(track.get("track_id", 0)),
                "details": {
                    "original_end_timestamp": round(float(original_end), 6),
                    "sanitized_end_timestamp": round(float(sanitized_end), 6),
                    "trimmed_duration_seconds": round(float(original_end) - float(sanitized_end), 6),
                    "reason": "boundary_stuck_frozen_kcf",
                },
            }
        )
    if list(track.get("summary_mismatch_fields", [])):
        sanitization_events.append(
            {
                "event_type": "summary_metadata_corrected",
                "track_id": int(track.get("track_id", 0)),
                "details": {"flags": list(track.get("summary_mismatch_fields", []))},
            }
        )
    if "frozen_kcf_box" in list(drift_report.get("flags", [])):
        sanitization_events.append({"event_type": "frozen_segment_detected", "track_id": int(track.get("track_id", 0)), "details": {}})
    if "boundary_stuck_box" in list(drift_report.get("flags", [])):
        sanitization_events.append({"event_type": "boundary_stuck_segment_detected", "track_id": int(track.get("track_id", 0)), "details": {}})
    if integrity_status == "invalid":
        sanitization_events.append({"event_type": "track_invalidated", "track_id": int(track.get("track_id", 0)), "details": {}})
    elif integrity_status == "fallback_only":
        sanitization_events.append({"event_type": "track_fallback_only", "track_id": int(track.get("track_id", 0)), "details": {}})
    sanitized_track = {
        **dict(track),
        "sanitized_timeline": sanitized_timeline,
        "sanitized_valid_timeline": valid_or_supported,
        "sanitized_start_timestamp_seconds": sanitized_start,
        "sanitized_end_timestamp_seconds": sanitized_end,
        "sanitized_duration_seconds": sanitized_duration,
        "sanitized_first_source_frame_index": sanitized_first_frame,
        "sanitized_last_source_frame_index": sanitized_last_frame,
        "last_valid_yolo_timestamp": last_valid_yolo_timestamp,
        "valid_observation_count": len(valid_or_supported),
        "invalid_observation_count": invalid_count,
        "valid_yolo_count": valid_yolo_count,
        "valid_kcf_count": valid_kcf_count,
        "invalid_kcf_count": invalid_kcf_count,
        "fallback_observation_count": fallback_count,
        "track_integrity_status": integrity_status,
        "timeline_correction_applied": bool(track.get("summary_mismatch_fields")),
        "frozen_kcf_detected": "frozen_kcf_box" in list(drift_report.get("flags", [])),
        "boundary_stuck_detected": "boundary_stuck_box" in list(drift_report.get("flags", [])),
        "long_kcf_only_gap_detected": "long_kcf_only_segment" in list(drift_report.get("flags", [])),
        "drift_flags": list(drift_report.get("flags", [])),
        "valid_crop_candidate_count": valid_crop_candidates,
        "trimmed_kcf_duration_seconds": round(
            max(
                0.0,
                float(track.get("actual_end_timestamp_seconds", 0.0) or 0.0) - float(sanitized_end or 0.0),
            ),
            6,
        ) if sanitized_end is not None else 0.0,
    }
    return sanitized_track, sanitization_events


__all__ = ["SanitizationConfig", "sanitize_track_timeline"]
