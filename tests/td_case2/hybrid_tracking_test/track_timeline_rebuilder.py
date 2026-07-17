from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _normalize_bbox(raw_bbox: list[float] | tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in list(raw_bbox)]


def _median_positive(values: list[int]) -> int:
    positives = [int(value) for value in values if int(value) > 0]
    if not positives:
        return 1
    return max(1, int(round(median(positives))))


def _observation_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(observation.get("source_frame_index", 0) or 0),
        int(observation.get("processed_frame_index", 0) or 0),
        round(float(observation.get("timestamp_seconds", 0.0) or 0.0), 6),
        tuple(_normalize_bbox(list(observation.get("bbox_xyxy", [])))),
        str(observation.get("bbox_source", "")),
    )


def _segment_observations(observations: list[dict[str, Any]], source_step_hint: int) -> list[dict[str, Any]]:
    if not observations:
        return []
    segments: list[list[dict[str, Any]]] = [[observations[0]]]
    for previous, current in zip(observations, observations[1:]):
        source_gap = int(current["source_frame_index"]) - int(previous["source_frame_index"])
        processed_gap = int(current["processed_frame_index"]) - int(previous["processed_frame_index"])
        if source_gap > max(source_step_hint * 2, 1) or processed_gap > 1:
            segments.append([current])
        else:
            segments[-1].append(current)
    payloads: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        payloads.append(
            {
                "segment_index": index,
                "observation_count": len(segment),
                "start_timestamp_seconds": round(float(segment[0]["timestamp_seconds"]), 6),
                "end_timestamp_seconds": round(float(segment[-1]["timestamp_seconds"]), 6),
                "start_source_frame_index": int(segment[0]["source_frame_index"]),
                "end_source_frame_index": int(segment[-1]["source_frame_index"]),
                "bbox_sources": dict(sorted(Counter(str(item.get("bbox_source", "")) for item in segment).items())),
            }
        )
    return payloads


def rebuild_track_timelines(
    raw_tracks: list[dict[str, Any]],
    frame_metrics_payload: dict[str, Any],
    tracking_report_payload: dict[str, Any],
    *,
    timeline_timestamp_tolerance_seconds: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    frames = list(frame_metrics_payload.get("frames", []))
    source_fps = float(tracking_report_payload.get("video_metadata", {}).get("fps", 0.0) or 0.0)
    source_frame_indexes = [int(frame.get("source_frame_index", 0) or 0) for frame in frames]
    source_step_hint = _median_positive(
        [current - previous for previous, current in zip(source_frame_indexes, source_frame_indexes[1:])]
    )
    by_track_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        source_frame_index = int(frame.get("source_frame_index", 0) or 0)
        processed_frame_index = int(frame.get("processed_frame_index", 0) or 0)
        timestamp_seconds = round(float(frame.get("timestamp_seconds", 0.0) or 0.0), 6)
        for raw_track in list(frame.get("tracks", [])):
            observation = {
                "track_id": int(raw_track.get("track_id", 0) or 0),
                "class_id": raw_track.get("class_id"),
                "class_name": str(raw_track.get("class_name", "unknown")),
                "object_family": str(raw_track.get("object_family", "other")),
                "bbox_xyxy": _normalize_bbox(list(raw_track.get("bbox_xyxy", []))),
                "bbox_source": str(raw_track.get("bbox_source", "")),
                "status": raw_track.get("status"),
                "kcf_success": bool(raw_track.get("kcf_success", False)),
                "frames_since_detection": int(raw_track.get("frames_since_detection", 0) or 0),
                "seconds_since_detection": round(float(raw_track.get("seconds_since_detection", 0.0) or 0.0), 6),
                "last_detection_confidence": None if raw_track.get("last_detection_confidence") is None else round(float(raw_track.get("last_detection_confidence")), 6),
                "reactivation_count": int(raw_track.get("reactivation_count", 0) or 0),
                "validation": dict(raw_track.get("validation", {})),
                "source_frame_index": source_frame_index,
                "processed_frame_index": processed_frame_index,
                "timestamp_seconds": timestamp_seconds,
            }
            by_track_id[int(observation["track_id"])].append(observation)

    rebuilt_tracks: list[dict[str, Any]] = []
    corrected_track_count = 0
    mismatch_counter: Counter[str] = Counter()
    raw_tracks_by_id = {int(item.get("track_id", 0) or 0): item for item in raw_tracks}
    for track_id, raw_track in sorted(raw_tracks_by_id.items()):
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        duplicate_timestamp_conflict = False
        for observation in sorted(
            by_track_id.get(track_id, []),
            key=lambda item: (
                float(item.get("timestamp_seconds", 0.0) or 0.0),
                int(item.get("processed_frame_index", 0) or 0),
                int(item.get("source_frame_index", 0) or 0),
                str(item.get("bbox_source", "")),
            ),
        ):
            key = _observation_key(observation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if deduped:
                previous = deduped[-1]
                if (
                    math.isclose(float(previous["timestamp_seconds"]), float(observation["timestamp_seconds"]), abs_tol=1e-6)
                    and int(previous["source_frame_index"]) == int(observation["source_frame_index"])
                    and tuple(previous["bbox_xyxy"]) != tuple(observation["bbox_xyxy"])
                ):
                    duplicate_timestamp_conflict = True
            deduped.append(observation)
        original_summary = {
            "start_timestamp_seconds": round(float(raw_track.get("start_timestamp_seconds", 0.0) or 0.0), 6),
            "end_timestamp_seconds": round(float(raw_track.get("end_timestamp_seconds", 0.0) or 0.0), 6),
            "duration_seconds": round(float(raw_track.get("duration_seconds", 0.0) or 0.0), 6),
            "first_source_frame_index": int(raw_track.get("first_source_frame_index", 0) or 0),
            "last_source_frame_index": int(raw_track.get("last_source_frame_index", 0) or 0),
        }
        integrity_flags: list[str] = []
        mismatch_details: dict[str, Any] = {}
        if not deduped:
            rebuilt_tracks.append(
                {
                    **dict(raw_track),
                    "rebuilt_timeline": [],
                    "continuous_observation_segments": [],
                    "missing_frame_gaps": [],
                    "bbox_history": [],
                    "center_history": [],
                    "area_history": [],
                    "all_yolo_observations": [],
                    "all_kcf_observations": [],
                    "full_observation_count": 0,
                    "actual_start_timestamp_seconds": None,
                    "actual_end_timestamp_seconds": None,
                    "actual_duration_seconds": 0.0,
                    "actual_first_source_frame_index": None,
                    "actual_last_source_frame_index": None,
                    "actual_first_processed_frame_index": None,
                    "actual_last_processed_frame_index": None,
                    "original_summary": original_summary,
                    "integrity_flags": ["missing_frame_level_observations"],
                    "summary_mismatch_fields": [],
                }
            )
            mismatch_counter["missing_frame_level_observations"] += 1
            corrected_track_count += 1
            continue
        expected_errors: list[float] = []
        missing_gaps: list[dict[str, Any]] = []
        non_monotonic_timestamp = False
        non_monotonic_source_frame = False
        for previous, current in zip(deduped, deduped[1:]):
            if float(current["timestamp_seconds"]) < float(previous["timestamp_seconds"]):
                non_monotonic_timestamp = True
            if int(current["source_frame_index"]) < int(previous["source_frame_index"]):
                non_monotonic_source_frame = True
            source_gap = int(current["source_frame_index"]) - int(previous["source_frame_index"])
            if source_gap > max(source_step_hint * 2, 1):
                missing_gaps.append(
                    {
                        "start_source_frame_index": int(previous["source_frame_index"]),
                        "end_source_frame_index": int(current["source_frame_index"]),
                        "missing_source_frame_count_estimate": max(0, int(round(source_gap / max(source_step_hint, 1))) - 1),
                        "time_gap_seconds": round(float(current["timestamp_seconds"]) - float(previous["timestamp_seconds"]), 6),
                    }
                )
        if non_monotonic_timestamp:
            integrity_flags.append("non_monotonic_timestamp")
        if non_monotonic_source_frame:
            integrity_flags.append("non_monotonic_source_frame")
        if duplicate_timestamp_conflict:
            integrity_flags.append("duplicate_timestamp_conflict")
        for observation in deduped:
            if source_fps > 0.0:
                expected_timestamp = float(observation["source_frame_index"]) / source_fps
                timestamp_error = abs(expected_timestamp - float(observation["timestamp_seconds"]))
                observation["expected_timestamp_from_source_frame"] = round(expected_timestamp, 6)
                observation["timestamp_error_seconds"] = round(timestamp_error, 6)
                expected_errors.append(timestamp_error)
                if timestamp_error > timeline_timestamp_tolerance_seconds:
                    observation["timeline_flags"] = ["timestamp_frame_mismatch"]
                    integrity_flags.append("timestamp_frame_mismatch")
                else:
                    observation["timeline_flags"] = []
        actual_start = round(float(deduped[0]["timestamp_seconds"]), 6)
        actual_end = round(float(deduped[-1]["timestamp_seconds"]), 6)
        actual_first_source_frame = int(deduped[0]["source_frame_index"])
        actual_last_source_frame = int(deduped[-1]["source_frame_index"])
        actual_first_processed_frame = int(deduped[0]["processed_frame_index"])
        actual_last_processed_frame = int(deduped[-1]["processed_frame_index"])
        actual_duration = round(max(0.0, actual_end - actual_start), 6)
        if actual_end < actual_start:
            integrity_flags.append("negative_duration")
        comparisons = {
            "summary_start_mismatch": (original_summary["start_timestamp_seconds"], actual_start),
            "summary_end_mismatch": (original_summary["end_timestamp_seconds"], actual_end),
            "summary_duration_mismatch": (original_summary["duration_seconds"], actual_duration),
            "summary_first_frame_mismatch": (original_summary["first_source_frame_index"], actual_first_source_frame),
            "summary_last_frame_mismatch": (original_summary["last_source_frame_index"], actual_last_source_frame),
        }
        for flag, (left, right) in comparisons.items():
            if isinstance(left, float) or isinstance(right, float):
                mismatch = not math.isclose(float(left), float(right), abs_tol=0.151)
            else:
                mismatch = int(left) != int(right)
            if mismatch:
                integrity_flags.append(flag)
                mismatch_details[flag] = {"original": left, "rebuilt": right}
        summary_mismatch_fields = sorted(set(flag for flag in integrity_flags if flag.startswith("summary_")))
        if summary_mismatch_fields or any(flag in integrity_flags for flag in {"missing_frame_level_observations", "timestamp_frame_mismatch"}):
            corrected_track_count += 1
        for flag in set(integrity_flags):
            mismatch_counter[flag] += 1
        rebuilt_tracks.append(
            {
                **dict(raw_track),
                "rebuilt_timeline": deduped,
                "continuous_observation_segments": _segment_observations(deduped, source_step_hint),
                "missing_frame_gaps": missing_gaps,
                "bbox_history": [list(item["bbox_xyxy"]) for item in deduped],
                "center_history": [[round(value, 6) for value in _bbox_center(list(item["bbox_xyxy"]))] for item in deduped],
                "area_history": [round(_bbox_area(list(item["bbox_xyxy"])), 6) for item in deduped],
                "all_yolo_observations": [item for item in deduped if str(item.get("bbox_source")) == "yolo"],
                "all_kcf_observations": [item for item in deduped if str(item.get("bbox_source")) == "kcf"],
                "full_observation_count": len(deduped),
                "actual_start_timestamp_seconds": actual_start,
                "actual_end_timestamp_seconds": actual_end,
                "actual_duration_seconds": actual_duration,
                "actual_first_source_frame_index": actual_first_source_frame,
                "actual_last_source_frame_index": actual_last_source_frame,
                "actual_first_processed_frame_index": actual_first_processed_frame,
                "actual_last_processed_frame_index": actual_last_processed_frame,
                "original_summary": original_summary,
                "rebuilt_metadata": {
                    "start_timestamp_seconds": actual_start,
                    "end_timestamp_seconds": actual_end,
                    "duration_seconds": actual_duration,
                    "first_source_frame_index": actual_first_source_frame,
                    "last_source_frame_index": actual_last_source_frame,
                    "first_processed_frame_index": actual_first_processed_frame,
                    "last_processed_frame_index": actual_last_processed_frame,
                    "median_timestamp_error_seconds": round(float(median(expected_errors)) if expected_errors else 0.0, 6),
                    "maximum_timestamp_error_seconds": round(max(expected_errors), 6) if expected_errors else 0.0,
                },
                "integrity_flags": sorted(set(integrity_flags)),
                "summary_mismatch_fields": summary_mismatch_fields,
                "mismatch_details": mismatch_details,
            }
        )
    report = {
        "status": "success",
        "timeline_timestamp_tolerance_seconds": round(float(timeline_timestamp_tolerance_seconds), 6),
        "source_fps": round(source_fps, 6),
        "source_frame_step_hint": int(source_step_hint),
        "track_count": len(rebuilt_tracks),
        "timelines_rebuilt": len(rebuilt_tracks),
        "timeline_corrections": corrected_track_count,
        "integrity_flag_counts": dict(sorted(mismatch_counter.items())),
        "tracks": [
            {
                "track_id": int(item.get("track_id", 0)),
                "original_summary": dict(item.get("original_summary", {})),
                "rebuilt_timeline": dict(item.get("rebuilt_metadata", {})),
                "integrity_flags": list(item.get("integrity_flags", [])),
                "summary_mismatch_fields": list(item.get("summary_mismatch_fields", [])),
                "missing_frame_gap_count": len(list(item.get("missing_frame_gaps", []))),
                "full_observation_count": int(item.get("full_observation_count", 0)),
            }
            for item in rebuilt_tracks
        ],
    }
    markdown_lines = [
        "# Track Timeline Integrity Report",
        "",
        f"- Timelines rebuilt: {report['timelines_rebuilt']}",
        f"- Timeline corrections: {report['timeline_corrections']}",
        f"- Timestamp tolerance seconds: {report['timeline_timestamp_tolerance_seconds']}",
    ]
    for flag, count in sorted(report["integrity_flag_counts"].items()):
        markdown_lines.append(f"- {flag}: {count}")
    return rebuilt_tracks, report, "\n".join(markdown_lines)


__all__ = ["rebuild_track_timelines"]
