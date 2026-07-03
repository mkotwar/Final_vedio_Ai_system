from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_optional_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_list(path: Path) -> list[dict[str, Any]]:
    payload = _load_optional_json(path)
    if not isinstance(payload, list):
        raise FileNotFoundError(f"Missing or invalid required coverage guardrail input file: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() == "true"


def _read_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _parse_timestamps(value: str) -> list[float]:
    timestamps: list[float] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        try:
            if len(pieces) == 1:
                timestamps.append(float(pieces[0]))
            elif len(pieces) == 2:
                timestamps.append((int(pieces[0]) * 60) + float(pieces[1]))
            elif len(pieces) == 3:
                timestamps.append((int(pieces[0]) * 3600) + (int(pieces[1]) * 60) + float(pieces[2]))
        except ValueError:
            print(f"[tender-demo] Warning: invalid critical timestamp ignored: {part!r}")
    return timestamps


def _clip_contains_time(clip: dict[str, Any], timestamp_seconds: float, padding_seconds: float = 0.0) -> bool:
    start_time = _safe_float(clip.get("expanded_start_time", clip.get("start_time")), 0.0) - padding_seconds
    end_time = _safe_float(clip.get("expanded_end_time", clip.get("end_time")), 0.0) + padding_seconds
    return start_time <= timestamp_seconds <= end_time


def _add_clip(
    selected_by_clip_id: dict[str, dict[str, Any]],
    clip: dict[str, Any],
    reason: str,
) -> bool:
    clip_id = str(clip.get("clip_id", "")).strip()
    if not clip_id:
        return False
    existing = selected_by_clip_id.get(clip_id)
    if existing is None:
        selected_by_clip_id[clip_id] = {
            **clip,
            "selection_reasons": list(clip.get("selection_reasons", []) or []) + [reason],
        }
        return True
    reasons = list(existing.get("selection_reasons", []) or [])
    if reason not in reasons:
        reasons.append(reason)
    existing["selection_reasons"] = reasons
    return False


def apply_incident_coverage_guardrails(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 14B: incident coverage guardrails")
    enabled = _read_env_bool("TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS", False)
    input_selected = _load_required_list(run_dir / "14_selected_top_clips.json")
    ranked_clips = _load_required_list(run_dir / "13_ranked_clips.json")
    yolo_scores = _load_required_list(run_dir / "11_yolo_object_scores.json")
    adaptive_payload = _load_optional_json(run_dir / "02b_adaptive_frames.json")
    candidate_pool_payload = _load_optional_json(run_dir / "02c_frame_candidate_pool.json")
    _load_optional_json(run_dir / "03_motion_scores.json")
    _load_optional_json(run_dir / "10_yolo_detections.json")

    adaptive_items = adaptive_payload.get("items", []) if isinstance(adaptive_payload, dict) and isinstance(adaptive_payload.get("items"), list) else []
    candidate_pool_items = candidate_pool_payload.get("items", []) if isinstance(candidate_pool_payload, dict) and isinstance(candidate_pool_payload.get("items"), list) else []
    ranked_by_clip_id = {
        str(item.get("clip_id", "")).strip(): item
        for item in ranked_clips
        if str(item.get("clip_id", "")).strip()
    }
    selected_by_clip_id = {
        str(item.get("clip_id", "")).strip(): {**item, "selection_reasons": list(item.get("selection_reasons", []) or [])}
        for item in input_selected
        if str(item.get("clip_id", "")).strip()
    }

    top_k_max = max(1, _read_env_int("TENDER_DEMO_TOP_K_MAX_CLIPS", max(len(input_selected), 25)))
    critical_window_seconds = _read_env_float("TENDER_DEMO_CRITICAL_WINDOW_SECONDS", 8.0)
    critical_timestamps = _parse_timestamps(os.environ.get("TENDER_DEMO_CRITICAL_TIMESTAMPS", ""))

    manual_critical_added = 0
    adaptive_target_added = 0
    adaptive_high_change_added = 0
    temporal_coverage_added = 0
    person_person_added = 0
    object_interaction_added = 0

    if enabled:
        for timestamp_seconds in critical_timestamps:
            for clip in ranked_clips:
                if _clip_contains_time(clip, timestamp_seconds, critical_window_seconds):
                    if _add_clip(selected_by_clip_id, clip, "manual_critical_timestamp"):
                        manual_critical_added += 1
                    break

        high_change_frames = []
        for item in adaptive_items:
            if not isinstance(item, dict):
                continue
            reasons = set(str(reason) for reason in item.get("keep_reasons", []) or [])
            if "target_timestamp_window" in reasons:
                high_change_frames.append((item, "adaptive_target_timestamp"))
            elif {"motion_change", "similarity_drop", "histogram_change"} & reasons:
                high_change_frames.append((item, "adaptive_high_change"))

        for frame_item, reason in high_change_frames:
            timestamp_seconds = _safe_float(frame_item.get("timestamp_seconds"), -1.0)
            for clip in ranked_clips:
                if _clip_contains_time(clip, timestamp_seconds, 2.0):
                    if _add_clip(selected_by_clip_id, clip, reason):
                        if reason == "adaptive_target_timestamp":
                            adaptive_target_added += 1
                        else:
                            adaptive_high_change_added += 1
                    break

        duration_seconds = 0.0
        for clip in ranked_clips:
            duration_seconds = max(duration_seconds, _safe_float(clip.get("end_time"), 0.0))
        if candidate_pool_items:
            duration_seconds = max(
                duration_seconds,
                max(_safe_float(item.get("timestamp_seconds"), 0.0) for item in candidate_pool_items),
            )
        if duration_seconds > 0:
            bucket_size = max(duration_seconds / 4.0, 1.0)
            for bucket_index in range(4):
                bucket_start = bucket_index * bucket_size
                bucket_end = bucket_start + bucket_size
                bucket_candidates = [
                    clip for clip in ranked_clips if _safe_float(clip.get("start_time"), 0.0) <= bucket_end and _safe_float(clip.get("end_time"), 0.0) >= bucket_start
                ]
                if bucket_candidates:
                    bucket_candidates.sort(key=lambda clip: -_safe_float(clip.get("ranked_clip_score"), 0.0))
                    if _add_clip(selected_by_clip_id, bucket_candidates[0], "temporal_coverage_clip"):
                        temporal_coverage_added += 1

        interaction_by_clip: dict[str, dict[str, bool]] = {}
        for frame in yolo_scores:
            clip_id = ""
            timestamp_seconds = _safe_float(frame.get("timestamp_seconds"), -1.0)
            for clip in ranked_clips:
                if _clip_contains_time(clip, timestamp_seconds):
                    clip_id = str(clip.get("clip_id", "")).strip()
                    break
            if not clip_id:
                continue
            state = interaction_by_clip.setdefault(clip_id, {"person_person": False, "person_object": False})
            person_count = int(frame.get("person_count", 0) or 0)
            important_object_count = int(frame.get("important_object_count", 0) or 0)
            if person_count >= 2:
                state["person_person"] = True
            if person_count >= 1 and important_object_count >= 1:
                state["person_object"] = True

        for clip_id, state in interaction_by_clip.items():
            clip = ranked_by_clip_id.get(clip_id)
            if clip is None:
                continue
            if state["person_person"] and _add_clip(selected_by_clip_id, clip, "person_person_interaction_possible"):
                person_person_added += 1
            if state["person_object"] and _add_clip(selected_by_clip_id, clip, "person_object_interaction_possible"):
                object_interaction_added += 1

    ordered_items = list(selected_by_clip_id.values())

    def sort_key(item: dict[str, Any]) -> tuple[int, float, float]:
        reasons = set(str(reason) for reason in item.get("selection_reasons", []) or [])
        if "manual_critical_timestamp" in reasons:
            bucket = 0
        elif "adaptive_target_timestamp" in reasons:
            bucket = 1
        elif "adaptive_high_change" in reasons:
            bucket = 2
        elif "person_person_interaction_possible" in reasons:
            bucket = 3
        elif "person_object_interaction_possible" in reasons:
            bucket = 4
        elif "temporal_coverage_clip" in reasons:
            bucket = 5
        else:
            bucket = 6
        return (
            bucket,
            -_safe_float(item.get("ranked_clip_score"), 0.0),
            _safe_float(item.get("start_time"), 0.0),
        )

    ordered_items.sort(key=sort_key)
    dropped_due_to_cap = max(0, len(ordered_items) - top_k_max)
    final_items = ordered_items[:top_k_max]
    for index, item in enumerate(final_items, start=1):
        item["selection_order"] = index

    report = {
        "enabled": enabled,
        "input_selected_count": len(input_selected),
        "output_selected_count": len(final_items),
        "manual_critical_timestamps": critical_timestamps,
        "manual_critical_clips_added": manual_critical_added,
        "adaptive_target_timestamp_clips_added": adaptive_target_added,
        "adaptive_high_change_clips_added": adaptive_high_change_added,
        "temporal_coverage_clips_added": temporal_coverage_added,
        "person_person_interaction_clips_added": person_person_added,
        "object_interaction_clips_added": object_interaction_added,
        "dropped_due_to_cap": dropped_due_to_cap,
        "selection_reasons_summary": {
            "manual_critical_timestamp": sum(1 for item in final_items if "manual_critical_timestamp" in set(item.get("selection_reasons", []) or [])),
            "adaptive_target_timestamp": sum(1 for item in final_items if "adaptive_target_timestamp" in set(item.get("selection_reasons", []) or [])),
            "adaptive_high_change": sum(1 for item in final_items if "adaptive_high_change" in set(item.get("selection_reasons", []) or [])),
            "person_person_interaction_possible": sum(1 for item in final_items if "person_person_interaction_possible" in set(item.get("selection_reasons", []) or [])),
            "person_object_interaction_possible": sum(1 for item in final_items if "person_object_interaction_possible" in set(item.get("selection_reasons", []) or [])),
            "temporal_coverage_clip": sum(1 for item in final_items if "temporal_coverage_clip" in set(item.get("selection_reasons", []) or [])),
            "top_k_ranked": sum(1 for item in final_items if "top_k_ranked" in set(item.get("selection_reasons", []) or [])),
        },
    }

    selected_output_path = run_dir / "14b_coverage_selected_clips.json"
    report_output_path = run_dir / "14b_coverage_guardrail_report.json"
    selected_output_path.write_text(json.dumps(final_items, indent=2), encoding="utf-8")
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[tender-demo] Coverage guardrail selected clips: {len(final_items)}")
    print(f"[tender-demo] Coverage guardrail selected path: {selected_output_path}")
    print(f"[tender-demo] Coverage guardrail report path: {report_output_path}")
    return {
        "selected_clips": final_items,
        "report": report,
        "selected_output_path": str(selected_output_path),
        "report_output_path": str(report_output_path),
    }
