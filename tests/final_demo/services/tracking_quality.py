from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_QA_FRAGMENT_GAP_SECONDS = "FINAL_DEMO_QA_FRAGMENT_GAP_SECONDS"
ENV_FINAL_DEMO_QA_CENTER_DISTANCE_RATIO = "FINAL_DEMO_QA_CENTER_DISTANCE_RATIO"
DEFAULT_QA_FRAGMENT_GAP_SECONDS = 3.0
DEFAULT_QA_CENTER_DISTANCE_RATIO = 0.35


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_optional_json(
    input_path: Path,
    *,
    required: bool,
    warnings: list[str],
    label: str,
) -> dict[str, Any] | None:
    if not input_path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing required Step 5A input: {input_path}. Step 5 must run before Step 5A."
            )
        warnings.append(f"Optional file missing: {label} ({input_path})")
        return None
    return read_json(input_path)


def choose_first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        return round(number, 3)
    return None


def choose_first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def normalize_sequence_count(track: dict[str, Any]) -> int:
    for key in ("detection_count",):
        value = choose_first_int(track.get(key))
        if value is not None:
            return max(0, value)

    for key in ("bbox_sequence", "detection_ids", "center_sequence"):
        items = track.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


def normalize_duration_seconds(track: dict[str, Any]) -> float:
    duration_seconds = choose_first_number(track.get("duration_seconds"))
    if duration_seconds is not None:
        return max(0.0, duration_seconds)

    start_time = choose_first_number(track.get("start_time"))
    end_time = choose_first_number(track.get("end_time"))
    if start_time is None or end_time is None:
        return 0.0
    return round(max(0.0, end_time - start_time), 3)


def normalize_track(track: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    class_name = str(track.get("class_name") or "unknown").lower()
    if "class_name" not in track:
        warnings.append("Missing optional field in 05_tracks.json: class_name")

    detection_count = normalize_sequence_count(track)
    duration_seconds = normalize_duration_seconds(track)
    start_time = choose_first_number(track.get("start_time"))
    end_time = choose_first_number(track.get("end_time"))
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = round(start_time + duration_seconds, 3)

    bbox_sequence = track.get("bbox_sequence")
    if not isinstance(bbox_sequence, list):
        bbox_sequence = []
    center_sequence = track.get("center_sequence")
    if not isinstance(center_sequence, list):
        center_sequence = []
    detection_ids = track.get("detection_ids")
    if not isinstance(detection_ids, list):
        detection_ids = []

    return {
        "local_track_id": str(track.get("local_track_id") or ""),
        "class_name": class_name,
        "chunk_id": str(track.get("chunk_id") or "unknown_chunk"),
        "start_time": round(float(start_time), 3),
        "end_time": round(float(end_time), 3),
        "duration_seconds": round(float(duration_seconds), 3),
        "detection_count": detection_count,
        "bbox_sequence": bbox_sequence,
        "center_sequence": center_sequence,
        "detection_ids": detection_ids,
        "source_fragment_track_ids": list(track.get("source_fragment_track_ids") or []),
        "source_raw_tracker_ids": list(track.get("source_raw_tracker_ids") or []),
    }


def build_track_list(tracks_payload: dict[str, Any] | None, warnings: list[str]) -> list[dict[str, Any]]:
    if tracks_payload is None:
        return []

    raw_tracks = tracks_payload.get("tracks")
    if raw_tracks is None:
        warnings.append("Missing optional field in 05_tracks.json: tracks")
        return []
    if not isinstance(raw_tracks, list):
        warnings.append("Invalid tracks payload in 05_tracks.json; expected a list under 'tracks'.")
        return []

    normalized_tracks: list[dict[str, Any]] = []
    for item in raw_tracks:
        if not isinstance(item, dict):
            warnings.append("Encountered non-dict track item in 05_tracks.json; skipped.")
            continue
        normalized_tracks.append(normalize_track(item, warnings))
    return normalized_tracks


def infer_average_time_gap_from_frames_index(frames_index_payload: dict[str, Any]) -> float | None:
    frames = frames_index_payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        return None

    seen_frame_ids: set[str] = set()
    timestamps: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_id = str(frame.get("frame_id") or "")
        if frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)
        timestamp = choose_first_number(frame.get("global_timestamp_seconds"))
        if timestamp is not None:
            timestamps.append(timestamp)

    if len(timestamps) < 2:
        return None

    timestamps.sort()
    deltas = [current - previous for previous, current in zip(timestamps, timestamps[1:]) if current > previous]
    if not deltas:
        return None
    return round(sum(deltas) / len(deltas), 3)


def extract_input_metrics(
    tracks_payload: dict[str, Any] | None,
    report_payload: dict[str, Any] | None,
    diagnostics_payload: dict[str, Any] | None,
    frames_index_payload: dict[str, Any] | None,
    tracks: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    raw_tracks_before_merge_list = []
    final_tracks_after_merge_list = []
    diagnostics_frames = []
    merge_pairs = []

    if diagnostics_payload is None:
        warnings.append("Missing optional tracking diagnostics file: 05_tracking_diagnostics.json")
    else:
        raw_tracks_before_merge_list = list(diagnostics_payload.get("raw_tracks_before_merge") or [])
        final_tracks_after_merge_list = list(diagnostics_payload.get("final_tracks_after_merge") or [])
        diagnostics_frames = list(
            diagnostics_payload.get("per_frame")
            or diagnostics_payload.get("diagnostics")
            or []
        )
        merge_pairs = list(diagnostics_payload.get("merge_pairs") or [])

    raw_tracks_created_before_merge = choose_first_int(
        len(raw_tracks_before_merge_list) if raw_tracks_before_merge_list else None,
        report_payload.get("raw_tracks_created_before_merge") if report_payload else None,
        tracks_payload.get("total_tracks_created") if tracks_payload else None,
        len(tracks),
    ) or 0

    final_tracks_after_merge = choose_first_int(
        len(final_tracks_after_merge_list) if final_tracks_after_merge_list else None,
        report_payload.get("total_tracks_kept") if report_payload else None,
        tracks_payload.get("total_tracks_kept") if tracks_payload else None,
        len(tracks),
    ) or 0

    no_merge_stage_detected = not final_tracks_after_merge_list
    if no_merge_stage_detected:
        final_tracks_after_merge = raw_tracks_created_before_merge
        warnings.append("No merge stage detected; final track count equals raw track count.")

    fragments_merged_count = choose_first_int(
        diagnostics_payload.get("fragments_merged_count") if diagnostics_payload else None,
        report_payload.get("fragments_merged_count") if report_payload else None,
        max(0, raw_tracks_created_before_merge - len(tracks)),
    ) or 0

    sample_fps = choose_first_number(
        report_payload.get("sample_fps") if report_payload else None,
        diagnostics_payload.get("sample_fps") if diagnostics_payload else None,
        frames_index_payload.get("sample_fps") if frames_index_payload else None,
    )
    source_fps = choose_first_number(
        report_payload.get("source_fps") if report_payload else None,
        diagnostics_payload.get("source_fps") if diagnostics_payload else None,
    )
    frame_skip_ratio = choose_first_number(
        report_payload.get("frame_skip_ratio") if report_payload else None,
        diagnostics_payload.get("frame_skip_ratio") if diagnostics_payload else None,
    )
    average_time_gap_between_frames = choose_first_number(
        diagnostics_payload.get("average_time_gap_between_frames") if diagnostics_payload else None,
        infer_average_time_gap_from_frames_index(frames_index_payload or {}),
    )

    total_input_detections = choose_first_int(
        report_payload.get("total_input_detections") if report_payload else None,
        tracks_payload.get("total_input_detections") if tracks_payload else None,
    ) or 0
    total_tracked_detections = choose_first_int(
        report_payload.get("total_tracked_detections") if report_payload else None,
        sum(track["detection_count"] for track in tracks),
    ) or 0
    total_untracked_detections = choose_first_int(
        report_payload.get("total_untracked_detections") if report_payload else None,
        diagnostics_payload.get("total_untracked_detections") if diagnostics_payload else None,
        max(0, total_input_detections - total_tracked_detections),
    ) or 0

    if total_input_detections == 0 and tracks:
        warnings.append("Missing optional field in 05_tracking_report.json: total_input_detections")
    if not raw_tracks_before_merge_list:
        warnings.append("Missing optional field in 05_tracking_diagnostics.json: raw_tracks_before_merge")
    if not final_tracks_after_merge_list:
        warnings.append("Missing optional field in 05_tracking_diagnostics.json: final_tracks_after_merge")

    return {
        "raw_tracks_created_before_merge": raw_tracks_created_before_merge,
        "final_tracks_after_merge": final_tracks_after_merge,
        "fragments_merged_count": fragments_merged_count,
        "sample_fps": sample_fps,
        "source_fps": source_fps,
        "frame_skip_ratio": frame_skip_ratio,
        "average_time_gap_between_frames": average_time_gap_between_frames,
        "total_input_detections": total_input_detections,
        "total_tracked_detections": total_tracked_detections,
        "total_untracked_detections": total_untracked_detections,
        "raw_tracks_before_merge_list": raw_tracks_before_merge_list,
        "final_tracks_after_merge_list": final_tracks_after_merge_list,
        "diagnostics_frames": diagnostics_frames,
        "merge_pairs": merge_pairs,
        "no_merge_stage_detected": no_merge_stage_detected,
    }


def extract_start_center(track: dict[str, Any]) -> list[float] | None:
    centers = track.get("center_sequence") or []
    if centers and isinstance(centers[0], list) and len(centers[0]) >= 2:
        return [float(centers[0][0]), float(centers[0][1])]

    boxes = track.get("bbox_sequence") or []
    if boxes and isinstance(boxes[0], list) and len(boxes[0]) >= 4:
        box = boxes[0]
        return [round((float(box[0]) + float(box[2])) / 2.0, 3), round((float(box[1]) + float(box[3])) / 2.0, 3)]
    return None


def extract_end_center(track: dict[str, Any]) -> list[float] | None:
    centers = track.get("center_sequence") or []
    if centers and isinstance(centers[-1], list) and len(centers[-1]) >= 2:
        return [float(centers[-1][0]), float(centers[-1][1])]

    boxes = track.get("bbox_sequence") or []
    if boxes and isinstance(boxes[-1], list) and len(boxes[-1]) >= 4:
        box = boxes[-1]
        return [round((float(box[0]) + float(box[2])) / 2.0, 3), round((float(box[1]) + float(box[3])) / 2.0, 3)]
    return None


def extract_frame_dimensions(track: dict[str, Any]) -> tuple[float, float] | None:
    boxes = track.get("bbox_sequence") or []
    if not boxes:
        return None

    max_x = 0.0
    max_y = 0.0
    for box in boxes:
        if isinstance(box, list) and len(box) >= 4:
            max_x = max(max_x, float(box[2]))
            max_y = max(max_y, float(box[3]))

    if max_x <= 0 or max_y <= 0:
        return None
    return max_x, max_y


def compute_center_distance(center_a: list[float], center_b: list[float]) -> float:
    return math.dist(center_a, center_b)


def centers_are_close(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    center_distance_ratio: float,
    warning_counter: dict[str, int],
) -> bool:
    center_a = extract_end_center(track_a)
    center_b = extract_start_center(track_b)
    if center_a is None or center_b is None:
        warning_counter["missing_center_sequences"] += 1
        return False

    dimensions_a = extract_frame_dimensions(track_a)
    dimensions_b = extract_frame_dimensions(track_b)
    if dimensions_a and dimensions_b:
        width = max(dimensions_a[0], dimensions_b[0])
        height = max(dimensions_a[1], dimensions_b[1])
        frame_diagonal = math.hypot(width, height)
        if frame_diagonal <= 0:
            warning_counter["missing_frame_dimensions"] += 1
            return False
        return compute_center_distance(center_a, center_b) <= frame_diagonal * center_distance_ratio

    if all(0.0 <= value <= 1.0 for value in center_a + center_b):
        return compute_center_distance(center_a, center_b) <= center_distance_ratio

    warning_counter["missing_frame_dimensions"] += 1
    return False


def estimate_fragmented_pairs(
    tracks: list[dict[str, Any]],
    *,
    fragment_gap_seconds: float,
    center_distance_ratio: float,
    warning_counter: dict[str, int],
) -> int:
    fragmented_pairs = 0
    ordered_tracks = sorted(tracks, key=lambda item: (item["chunk_id"], item["start_time"], item["end_time"]))
    for index, track_a in enumerate(ordered_tracks):
        for track_b in ordered_tracks[index + 1 :]:
            if track_a["chunk_id"] != track_b["chunk_id"]:
                continue
            if track_a["end_time"] > track_b["start_time"]:
                continue
            gap_seconds = round(track_b["start_time"] - track_a["end_time"], 3)
            if gap_seconds > fragment_gap_seconds:
                continue
            if centers_are_close(
                track_a,
                track_b,
                center_distance_ratio=center_distance_ratio,
                warning_counter=warning_counter,
            ):
                fragmented_pairs += 1
    return fragmented_pairs


def estimate_duplicate_pairs(
    tracks: list[dict[str, Any]],
    *,
    center_distance_ratio: float,
    warning_counter: dict[str, int],
) -> int:
    duplicate_pairs = 0
    ordered_tracks = sorted(tracks, key=lambda item: (item["chunk_id"], item["start_time"], item["end_time"]))
    for index, track_a in enumerate(ordered_tracks):
        for track_b in ordered_tracks[index + 1 :]:
            if track_a["chunk_id"] != track_b["chunk_id"]:
                continue
            overlaps = min(track_a["end_time"], track_b["end_time"]) - max(track_a["start_time"], track_b["start_time"])
            if overlaps <= 0:
                continue

            close_start = centers_are_close(
                {**track_a, "center_sequence": track_a.get("center_sequence") or [extract_start_center(track_a)]},
                {**track_b, "center_sequence": track_b.get("center_sequence") or [extract_start_center(track_b)]},
                center_distance_ratio=center_distance_ratio,
                warning_counter=warning_counter,
            )
            close_end = centers_are_close(
                track_a,
                {**track_b, "center_sequence": track_b.get("center_sequence") or [extract_end_center(track_b)]},
                center_distance_ratio=center_distance_ratio,
                warning_counter=warning_counter,
            )
            if close_start or close_end:
                duplicate_pairs += 1
    return duplicate_pairs


def compute_max_active_tracks(tracks: list[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for track in tracks:
        events.append((float(track["start_time"]), 1))
        events.append((float(track["end_time"]), -1))

    if not events:
        return 0

    events.sort(key=lambda item: (item[0], -item[1]))
    active_tracks = 0
    max_active_tracks = 0
    for _, delta in events:
        active_tracks += delta
        max_active_tracks = max(max_active_tracks, active_tracks)
    return max_active_tracks


def build_quality_by_class(
    tracks: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    fragment_gap_seconds: float,
    center_distance_ratio: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    diagnostics_raw_tracks = metrics["raw_tracks_before_merge_list"]
    raw_tracks_by_class: dict[str, int] = defaultdict(int)
    for item in diagnostics_raw_tracks:
        if isinstance(item, dict):
            raw_tracks_by_class[str(item.get("class_name") or "unknown").lower()] += 1

    final_tracks_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        final_tracks_by_class[track["class_name"]].append(track)

    all_classes = sorted(set(raw_tracks_by_class) | set(final_tracks_by_class))
    quality_items: list[dict[str, Any]] = []
    warning_counter: dict[str, int] = defaultdict(int)

    for class_name in all_classes:
        class_tracks = sorted(
            final_tracks_by_class.get(class_name, []),
            key=lambda item: (item["chunk_id"], item["start_time"], item["end_time"]),
        )
        durations = [float(track["duration_seconds"]) for track in class_tracks]
        detection_counts = [int(track["detection_count"]) for track in class_tracks]
        tracks_under_1_second = sum(1 for value in durations if value < 1.0)
        tracks_under_3_detections = sum(1 for value in detection_counts if value < 3)
        tracks_under_5_detections = sum(1 for value in detection_counts if value < 5)
        avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0
        median_duration = round(statistics.median(durations), 3) if durations else 0.0
        avg_detections = round(sum(detection_counts) / len(detection_counts), 3) if detection_counts else 0.0
        fragmented_pairs = estimate_fragmented_pairs(
            class_tracks,
            fragment_gap_seconds=fragment_gap_seconds,
            center_distance_ratio=center_distance_ratio,
            warning_counter=warning_counter,
        )
        duplicate_pairs = estimate_duplicate_pairs(
            class_tracks,
            center_distance_ratio=center_distance_ratio,
            warning_counter=warning_counter,
        )
        max_visible = compute_max_active_tracks(class_tracks)

        estimated_clean_track_count = len(class_tracks) - sum(
            1
            for track in class_tracks
            if float(track["duration_seconds"]) < 1.0 and int(track["detection_count"]) < 3
        )
        estimated_clean_track_count = max(max_visible, estimated_clean_track_count, 0)

        if len(class_tracks) == 0 and raw_tracks_by_class.get(class_name, 0) > 0:
            tracking_status = "bad_tracking"
        elif len(class_tracks) <= max_visible + 2:
            tracking_status = "good"
        elif len(class_tracks) <= max_visible + 5:
            tracking_status = "acceptable_needs_review"
        else:
            tracking_status = "needs_cleanup"

        item = {
            "class_name": class_name,
            "raw_track_count_before_merge": int(raw_tracks_by_class.get(class_name, len(class_tracks))),
            "final_track_count_after_merge": len(class_tracks),
            "tracks_under_1_second": tracks_under_1_second,
            "tracks_under_3_detections": tracks_under_3_detections,
            "tracks_under_5_detections": tracks_under_5_detections,
            "average_track_duration_seconds": avg_duration,
            "median_track_duration_seconds": median_duration,
            "average_detections_per_track": avg_detections,
            "possible_fragmented_track_pairs_count": fragmented_pairs,
            "possible_duplicate_track_pairs_count": duplicate_pairs,
            "max_objects_visible_at_once": max_visible,
            "estimated_clean_track_count": estimated_clean_track_count,
            "tracking_status": tracking_status,
        }

        if class_name == "person":
            lower_bound = max_visible
            upper_bound = max(lower_bound, estimated_clean_track_count)
            item["raw_person_tracks"] = int(raw_tracks_by_class.get(class_name, len(class_tracks)))
            item["clean_person_tracks"] = estimated_clean_track_count
            item["max_people_visible_at_once"] = max_visible
            item["estimated_unique_people_range"] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

        quality_items.append(item)

    if warning_counter["missing_frame_dimensions"] > 0:
        warnings.append(
            "Some fragment/duplicate checks could not use frame-diagonal distance because frame dimensions were missing."
        )
    if warning_counter["missing_center_sequences"] > 0:
        warnings.append(
            "Some fragment/duplicate checks had to skip center-based comparison because center data was missing."
        )

    return quality_items


def build_main_problem(metrics: dict[str, Any], quality_by_class: list[dict[str, Any]]) -> list[str]:
    main_problem: list[str] = []
    total_final_tracks = sum(int(item["final_track_count_after_merge"]) for item in quality_by_class)
    short_tracks = sum(int(item["tracks_under_1_second"]) for item in quality_by_class)
    very_short_detection_tracks = sum(int(item["tracks_under_3_detections"]) for item in quality_by_class)
    fragmented_pairs = sum(int(item["possible_fragmented_track_pairs_count"]) for item in quality_by_class)
    duplicate_pairs = sum(int(item["possible_duplicate_track_pairs_count"]) for item in quality_by_class)

    if total_final_tracks > 0 and (short_tracks / total_final_tracks >= 0.25 or very_short_detection_tracks / total_final_tracks >= 0.25):
        main_problem.append("short_track_noise")
    if metrics["total_input_detections"] > 0 and metrics["total_untracked_detections"] / metrics["total_input_detections"] >= 0.1:
        main_problem.append("dropped_tracker_ids")
    if fragmented_pairs >= 3:
        main_problem.append("track_fragmentation")
    if duplicate_pairs >= 2:
        main_problem.append("duplicate_tracks")
    if metrics["sample_fps"] is not None and metrics["sample_fps"] < 5:
        main_problem.append("low_sample_fps")
    if metrics["average_time_gap_between_frames"] is not None and metrics["average_time_gap_between_frames"] > 0.5:
        main_problem.append("frame_skipping_risk")
    if metrics["diagnostics_frames"] == [] and metrics["raw_tracks_before_merge_list"] == [] and metrics["final_tracks_after_merge_list"] == []:
        main_problem.append("missing_diagnostics")
    if metrics["no_merge_stage_detected"]:
        main_problem.append("no_merge_stage_detected")
    return main_problem


def build_overall_status(quality_by_class: list[dict[str, Any]]) -> str:
    statuses = {str(item["tracking_status"]) for item in quality_by_class}
    if "bad_tracking" in statuses:
        return "bad_tracking"
    if "needs_cleanup" in statuses:
        return "needs_cleanup"
    if "acceptable_needs_review" in statuses:
        return "acceptable_needs_review"
    return "good"


def build_recommendations(metrics: dict[str, Any], quality_by_class: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []

    if metrics["sample_fps"] is None or metrics["sample_fps"] < 5:
        recommendations.append("Increase FINAL_DEMO_SAMPLE_FPS to 5 or 8.")
    if metrics["total_input_detections"] > 0 and metrics["total_untracked_detections"] / metrics["total_input_detections"] >= 0.1:
        recommendations.append("Lower FINAL_DEMO_YOLO_CONF to 0.15 if tracker is dropping detections.")
    if metrics["merge_pairs"]:
        recommendations.append("Review merge_pairs in 05_tracking_diagnostics.json.")
    else:
        recommendations.append("Enable or tune fragment merging.")

    fragmented_pairs = sum(int(item["possible_fragmented_track_pairs_count"]) for item in quality_by_class)
    if fragmented_pairs >= 3:
        recommendations.append("If fragmentation remains high, use BoT-SORT/ReID or stronger tracker.")
    recommendations.append("For production unique identity, add cross-chunk/global ReID after local tracking.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in recommendations:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_tracking_quality_report(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    fragment_gap_seconds = round(
        read_non_negative_float_env(
            ENV_FINAL_DEMO_QA_FRAGMENT_GAP_SECONDS,
            DEFAULT_QA_FRAGMENT_GAP_SECONDS,
        ),
        3,
    )
    center_distance_ratio = round(
        read_non_negative_float_env(
            ENV_FINAL_DEMO_QA_CENTER_DISTANCE_RATIO,
            DEFAULT_QA_CENTER_DISTANCE_RATIO,
        ),
        3,
    )

    tracks_path = run_dir / "05_tracks.json"
    report_path = run_dir / "05_tracking_report.json"
    diagnostics_path = run_dir / "05_tracking_diagnostics.json"
    chunk_manifest_path = run_dir / "02_chunk_manifest.json"
    frames_index_path = run_dir / "03_sampled_frames_index.json"
    detection_report_path = run_dir / "04_yolo_detection_report.json"
    focus_path = run_dir / "05_tracking_focus.json"

    tracks_payload = read_optional_json(tracks_path, required=True, warnings=warnings, label="05_tracks.json")
    report_payload = read_optional_json(report_path, required=True, warnings=warnings, label="05_tracking_report.json")
    diagnostics_payload = read_optional_json(
        diagnostics_path,
        required=True,
        warnings=warnings,
        label="05_tracking_diagnostics.json",
    )
    chunk_manifest_payload = read_optional_json(
        chunk_manifest_path,
        required=True,
        warnings=warnings,
        label="02_chunk_manifest.json",
    )
    frames_index_payload = read_optional_json(
        frames_index_path,
        required=False,
        warnings=warnings,
        label="03_sampled_frames_index.json",
    )
    detection_report_payload = read_optional_json(
        detection_report_path,
        required=False,
        warnings=warnings,
        label="04_yolo_detection_report.json",
    )
    focus_payload = read_optional_json(
        focus_path,
        required=False,
        warnings=warnings,
        label="05_tracking_focus.json",
    )

    tracks = build_track_list(tracks_payload or {}, warnings)
    if isinstance(report_payload, dict):
        for item in list(report_payload.get("warnings") or []):
            if isinstance(item, str):
                warnings.append(item)
    metrics = extract_input_metrics(
        tracks_payload,
        report_payload,
        diagnostics_payload,
        frames_index_payload,
        tracks,
        warnings,
    )
    quality_by_class = build_quality_by_class(
        tracks,
        metrics,
        fragment_gap_seconds=fragment_gap_seconds,
        center_distance_ratio=center_distance_ratio,
        warnings=warnings,
    )
    overall_status = build_overall_status(quality_by_class)
    main_problem = build_main_problem(metrics, quality_by_class)

    if metrics["sample_fps"] is not None and metrics["sample_fps"] < 5:
        warnings.append("Sample FPS below 5 may cause identity fragmentation for people tracking.")
    warnings.append(
        "Tracking is performed on sampled frames, so frame skipping may increase identity fragmentation and reduce temporal continuity."
    )
    if any(item["class_name"] == "person" for item in quality_by_class):
        warnings.append(
            "Person track count may still include fragments; unique person count requires fragment merging and later cross-chunk/global ReID."
        )
    warnings = list(dict.fromkeys(warnings))

    report = {
        "created_at": current_timestamp(),
        "input_files": {
            "05_tracks": str(tracks_path),
            "05_tracking_report": str(report_path),
            "05_tracking_diagnostics": str(diagnostics_path),
            "02_chunk_manifest": str(chunk_manifest_path),
            "03_sampled_frames_index": str(frames_index_path) if frames_index_payload is not None else None,
            "04_yolo_detection_report": str(detection_report_path) if detection_report_payload is not None else None,
            "05_tracking_focus": str(focus_path) if focus_payload is not None else None,
        },
        "tracking_focus": {
            "focus_mode": focus_payload.get("focus_mode") if isinstance(focus_payload, dict) else None,
            "selected_focus_profile": (
                focus_payload.get("selected_focus_profile") if isinstance(focus_payload, dict) else None
            ),
            "selected_track_classes": (
                list(focus_payload.get("selected_track_classes") or [])
                if isinstance(focus_payload, dict)
                else []
            ),
            "focus_confidence": (
                focus_payload.get("focus_confidence") if isinstance(focus_payload, dict) else None
            ),
            "reason": focus_payload.get("reason") if isinstance(focus_payload, dict) else None,
        },
        "overall_status": overall_status,
        "main_problem": main_problem,
        "sample_fps": metrics["sample_fps"],
        "source_fps": metrics["source_fps"],
        "frame_skip_ratio": metrics["frame_skip_ratio"],
        "average_time_gap_between_frames": metrics["average_time_gap_between_frames"],
        "total_input_detections": metrics["total_input_detections"],
        "total_tracked_detections": metrics["total_tracked_detections"],
        "total_untracked_detections": metrics["total_untracked_detections"],
        "raw_tracks_created_before_merge": metrics["raw_tracks_created_before_merge"],
        "final_tracks_after_merge": metrics["final_tracks_after_merge"],
        "fragments_merged_count": metrics["fragments_merged_count"],
        "quality_by_class": quality_by_class,
        "warnings": warnings,
        "recommendations": build_recommendations(metrics, quality_by_class),
        "qa_settings": {
            "fragment_gap_seconds": fragment_gap_seconds,
            "center_distance_ratio": center_distance_ratio,
        },
        "context": {
            "chunk_manifest_total_chunks": int(chunk_manifest_payload.get("total_chunks", 0) or 0)
            if chunk_manifest_payload
            else 0,
            "detection_report_frames_processed": choose_first_int(
                detection_report_payload.get("total_frames_processed") if detection_report_payload else None
            ),
        },
    }
    return report


def update_run_manifest_for_tracking_quality(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "05A_tracking_quality" not in completed_steps:
        completed_steps.append("05A_tracking_quality")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "06_attribute_extraction"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
