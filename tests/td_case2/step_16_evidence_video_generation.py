from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage_checks import format_seconds_text, read_json, write_json


EVIDENCE_VIDEO_NAME = "evidence_video.mp4"
EVIDENCE_INDEX_NAME = "evidence_video_index.json"
REPORT_NAME = "16_evidence_video_report.json"
STEP11_FILE = "11_full_scene_event_candidates.json"
STEP12_FILE = "12_selected_top_event_candidates.json"
STEP13_FILE = "13_vlm_event_inputs.json"
STEP14_FILE = "14_vlm_event_reviews.json"
STEP14_SUMMARY_FILE = "14_final_video_summary.json"
STEP15_FILE = "15_searchable_events.json"
SEARCH_INDEX_FILE = "07B_traffic_object_search_index.json"
TRACKS_FILE = "04B_tracks.json"
FRAME_MANIFEST_FILES = ("02A_adaptive_frames.json", "02_sampled_frames.json")


@dataclass(frozen=True)
class EvidenceVideoConfig:
    clip_fps: int
    header_seconds: float
    summary_seconds: float
    object_context_before_seconds: float
    object_context_after_seconds: float
    max_object_events: int
    include_person_events: bool
    include_normal_context_scene_events: bool


def _normalize_rel_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized or None


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    normalized = _normalize_rel_path(path_value)
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_precise_timestamp(total_seconds: float) -> str:
    clamped = max(0.0, float(total_seconds or 0.0))
    hours = int(clamped // 3600)
    minutes = int((clamped % 3600) // 60)
    seconds = clamped % 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _frame_manifest(run_dir: Path) -> list[dict[str, Any]]:
    for file_name in FRAME_MANIFEST_FILES:
        path = run_dir / file_name
        if not path.exists():
            continue
        payload = read_json(path)
        items = payload.get("selected_frames") or payload.get("sampled_frames") or payload.get("frames") or []
        return [item for item in items if isinstance(item, dict)]
    return []


def _frame_catalog(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    frames = _frame_manifest(run_dir)
    ordered = sorted(frames, key=lambda item: _safe_float(item.get("timestamp_seconds"), 0.0))
    by_id = {
        str(item.get("frame_id", "") or ""): item
        for item in ordered
        if str(item.get("frame_id", "") or "")
    }
    return ordered, by_id


def _tracks_payload(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / TRACKS_FILE)


def _track_map_and_detection_index(tracks_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    track_map: dict[str, dict[str, Any]] = {}
    detection_index: dict[tuple[str, str], dict[str, Any]] = {}
    for track in list(tracks_payload.get("tracks", [])):
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("track_id", "") or "")
        if not track_id:
            continue
        track_map[track_id] = track
        for detection in list(track.get("detections", [])):
            if not isinstance(detection, dict):
                continue
            frame_id = str(detection.get("frame_id", "") or "")
            if frame_id:
                detection_index[(track_id, frame_id)] = detection
    return track_map, detection_index


def _step11_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(run_dir / STEP11_FILE)
    return {
        str(item.get("candidate_event_id", "") or ""): item
        for item in list(payload.get("candidate_events", []))
        if isinstance(item, dict) and str(item.get("candidate_event_id", "") or "")
    }


def _step12_selected(run_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(run_dir / STEP12_FILE)
    return [item for item in list(payload.get("selected_candidates", [])) if isinstance(item, dict)]


def _step14_reviews(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any] | None]:
    review_path = run_dir / STEP14_FILE
    summary_path = run_dir / STEP14_SUMMARY_FILE
    summary_payload = read_json(summary_path) if summary_path.exists() else None
    if not review_path.exists():
        return [], {}, summary_payload
    payload = read_json(review_path)
    reviews = [item for item in list(payload.get("reviews", [])) if isinstance(item, dict)]
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        for candidate_id in list(review.get("source_candidate_ids", [])):
            key = str(candidate_id or "")
            if key and key not in by_candidate_id:
                by_candidate_id[key] = review
    return reviews, by_candidate_id, summary_payload


def _step15_scene_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / STEP15_FILE
    if not path.exists():
        return []
    payload = read_json(path)
    return [item for item in list(payload.get("records", [])) if isinstance(item, dict)]


def _search_records(run_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(run_dir / SEARCH_INDEX_FILE)
    return [item for item in list(payload.get("records", [])) if isinstance(item, dict)]


def _step13_vlm_inputs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / STEP13_FILE
    if not path.exists():
        return []
    payload = read_json(path)
    return [item for item in list(payload.get("vlm_inputs", [])) if isinstance(item, dict)]


def _recording_datetime_strings(video_info: dict[str, Any], run_dir: Path) -> tuple[str, str]:
    date_value = str(video_info.get("recording_date", "") or "").strip()
    time_value = str(video_info.get("recording_time", "") or "").strip()
    if date_value and time_value:
        return date_value, time_value
    input_path = _resolve_run_path(run_dir, str(video_info.get("input_video_path", "") or ""))
    if input_path is not None and input_path.exists():
        timestamp = time.localtime(input_path.stat().st_mtime)
        return time.strftime("%Y-%m-%d", timestamp), time.strftime("%H:%M:%S", timestamp)
    return "unavailable", "unavailable"


def _candidate_review_decision(review: dict[str, Any] | None) -> str:
    if not review:
        return "not_reviewed"
    return str(dict(review.get("model_review", {})).get("review_decision", "not_reviewed") or "not_reviewed")


def _candidate_summary(candidate: dict[str, Any], review: dict[str, Any] | None) -> str:
    if review:
        text = str(dict(review.get("model_review", {})).get("summary_caption", "") or "").strip()
        if text:
            return text
    event_type = str(candidate.get("event_type", "candidate_event") or "candidate_event").replace("_", " ")
    return f"{event_type.capitalize()} around {candidate.get('best_timestamp_text', '')}."


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _normalized_plate(value: Any) -> str:
    return "".join(char for char in str(value or "").upper() if char.isalnum())


def _unique_strings(values: list[Any], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def _scene_event_entries(
    *,
    selected_candidates: list[dict[str, Any]],
    step11_map: dict[str, dict[str, Any]],
    review_map: dict[str, dict[str, Any]],
    config: EvidenceVideoConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        candidate_id = str(candidate.get("candidate_event_id", "") or "")
        step11_candidate = dict(step11_map.get(candidate_id, {}))
        review = review_map.get(candidate_id)
        review_decision = _candidate_review_decision(review)
        include_event = review_decision == "event_visible"
        if not include_event and config.include_normal_context_scene_events:
            include_event = review_decision in {"normal_context", "uncertain", "not_reviewed"}
        if not include_event:
            continue

        track_ids = list(step11_candidate.get("involved_track_ids", []))
        events.append(
            {
                "event_id": candidate_id,
                "searchable_event_id": candidate_id,
                "search_id": candidate_id,
                "source_type": "scene_event",
                "title": str(step11_candidate.get("event_type", candidate.get("event_type", "scene_event")) or "scene_event").replace("_", " ").title(),
                "event_type": str(step11_candidate.get("event_type", candidate.get("event_type", "scene_event")) or "scene_event"),
                "summary": _candidate_summary(step11_candidate or candidate, review),
                "start_seconds": _safe_float(step11_candidate.get("context_start_seconds", candidate.get("context_start_seconds")), 0.0),
                "end_seconds": _safe_float(step11_candidate.get("context_end_seconds", candidate.get("context_end_seconds")), 0.0),
                "best_timestamp_seconds": _safe_float(step11_candidate.get("best_timestamp_seconds", candidate.get("best_timestamp_seconds")), 0.0),
                "track_ids": [str(track_id or "") for track_id in track_ids if str(track_id or "")],
                "class_names": list(step11_candidate.get("involved_classes", [])),
                "confidence": float(dict(review.get("model_review", {})).get("confidence", candidate.get("ranking_score", 0.0)) or 0.0) if review else _safe_float(candidate.get("ranking_score"), 0.0),
                "event_confidence_label": str(step11_candidate.get("confidence_label", candidate.get("vlm_priority", "medium")) or "medium"),
                "vehicle_info": {},
                "license_plate": None,
                "ocr_text": None,
                "bbox_source": "tracks",
                "representative_frame_path": step11_candidate.get("representative_frame", {}).get("image_path") or candidate.get("representative_frame_path"),
                "review_decision": review_decision,
                "review_payload": review,
                "source_candidate_ids": [candidate_id],
                "priority_rank": 3,
            }
        )
    return events


def _important_object_record(record: dict[str, Any], include_person_events: bool) -> tuple[bool, str | None, int]:
    object_type = str(record.get("object_type", "") or "")
    class_name = str(record.get("class_name", "") or "")
    verified_plate = str(record.get("verified_license_plate", "") or "").strip()
    verified_color = str(record.get("verified_vehicle_color", "") or "").strip()
    possible_color = str(record.get("possible_vehicle_color", "") or "").strip()
    possible_plate = str(record.get("possible_plate_text", "") or "").strip()
    weak_ocr = [str(item or "").strip() for item in list(record.get("weak_ocr_text", [])) if str(item or "").strip()]
    quality = str(record.get("quality", "") or "")
    confidence = _safe_float(record.get("confidence"), 0.0)
    meaningful_ocr = []
    for text in weak_ocr:
        compact = "".join(char for char in text.upper() if char.isalnum())
        if len(compact) < 5:
            continue
        if len(set(compact)) == 1:
            continue
        if not any(char.isdigit() for char in compact):
            continue
        meaningful_ocr.append(compact)

    if verified_plate:
        return True, "verified_plate_detected", 4
    if len("".join(char for char in possible_plate.upper() if char.isalnum())) >= 6:
        return True, "possible_plate_detected", 3
    if object_type == "vehicle" and (verified_color or possible_color) and quality in {"primary", "fallback"} and confidence >= 0.35:
        return True, "searchable_vehicle_detected", 2
    if meaningful_ocr:
        return True, "important_ocr_detected", 1
    if include_person_events and (object_type == "person" or class_name == "person") and confidence >= 0.55 and quality in {"primary", "fallback"}:
        return True, "person_detected", 1
    return False, None, 0


def _object_event_group_key(record: dict[str, Any]) -> tuple[str, ...]:
    track_id = str(record.get("track_id", "") or "").strip()
    if track_id:
        return ("track", track_id)
    plate_value = _normalized_plate(record.get("verified_license_plate") or record.get("possible_plate_text"))
    if plate_value:
        return ("plate", plate_value)
    object_type = str(record.get("object_type", "") or "").strip() or "object"
    class_name = str(record.get("class_name", "") or "").strip() or object_type
    bucket = int(_safe_float(record.get("timestamp_seconds"), 0.0) // 5.0)
    return ("class_time", object_type, class_name, str(bucket))


def _object_record_rank(record: dict[str, Any], reason: str, priority_rank: int) -> tuple[float, float, float, float]:
    confidence = _safe_float(record.get("confidence"), 0.0)
    quality = str(record.get("quality", "") or "")
    quality_bonus = 0.2 if quality == "primary" else 0.1 if quality == "fallback" else 0.0
    color_bonus = 0.05 if str(record.get("verified_vehicle_color", "") or "").strip() else 0.0
    plate_bonus = 0.25 if _normalized_plate(record.get("verified_license_plate")) else 0.15 if _normalized_plate(record.get("possible_plate_text")) else 0.0
    reason_bonus = 0.05 if reason == "important_ocr_detected" else 0.0
    return (
        float(priority_rank),
        confidence + quality_bonus + color_bonus + plate_bonus + reason_bonus,
        _safe_float(record.get("duration_seconds"), 0.0),
        -_safe_float(record.get("timestamp_seconds"), 0.0),
    )


def _merge_object_event_group(
    group_items: list[tuple[dict[str, Any], str, int]],
    config: EvidenceVideoConfig,
) -> dict[str, Any] | None:
    if not group_items:
        return None
    best_record, best_reason, best_priority = max(
        group_items,
        key=lambda item: _object_record_rank(item[0], item[1], item[2]),
    )
    start_seconds = min(
        _safe_float(record.get("first_seen_seconds", record.get("timestamp_seconds")), 0.0)
        for record, _, _ in group_items
    )
    end_seconds = max(
        _safe_float(record.get("last_seen_seconds", record.get("timestamp_seconds")), 0.0)
        for record, _, _ in group_items
    )
    if end_seconds < start_seconds:
        end_seconds = start_seconds
    start_seconds = max(0.0, start_seconds - config.object_context_before_seconds)
    end_seconds = max(start_seconds, end_seconds + config.object_context_after_seconds)
    track_ids = _unique_strings([record.get("track_id") for record, _, _ in group_items])
    class_names = _unique_strings([record.get("class_name") for record, _, _ in group_items]) or ["object"]
    event_type = max(group_items, key=lambda item: (item[2], _safe_float(item[0].get("confidence"), 0.0)))[1]
    plate_candidates = _unique_strings(
        [record.get("verified_license_plate") for record, _, _ in group_items]
        + [record.get("possible_plate_text") for record, _, _ in group_items]
        + [text for record, _, _ in group_items for text in list(record.get("weak_ocr_text", []))]
    )
    best_timestamp_seconds = _safe_float(best_record.get("timestamp_seconds"), start_seconds)
    best_full_frame_path = (
        best_record.get("full_frame_path")
        or best_record.get("best_full_frame_path")
        or best_record.get("representative_frame_path")
    )
    vehicle_info = {
        "type": best_record.get("class_name"),
        "color": best_record.get("verified_vehicle_color") or best_record.get("possible_vehicle_color") or best_record.get("vehicle_color"),
        "make": best_record.get("vehicle_make"),
        "model": best_record.get("vehicle_model"),
    }
    return {
        "event_id": str(best_record.get("object_record_id", "") or ""),
        "searchable_event_id": str(best_record.get("object_record_id", "") or ""),
        "search_id": str(best_record.get("object_record_id", "") or ""),
        "source_type": "searchable_object_event",
        "title": str(event_type).replace("_", " ").title(),
        "event_type": str(event_type),
        "summary": str(best_record.get("search_text", "") or f"{class_names[0]} detected"),
        "start_seconds": round(start_seconds, 6),
        "end_seconds": round(end_seconds, 6),
        "best_timestamp_seconds": best_timestamp_seconds,
        "track_ids": track_ids,
        "class_names": class_names,
        "confidence": max(_safe_float(record.get("confidence"), 0.0) for record, _, _ in group_items),
        "event_confidence_label": "high" if max(_safe_float(record.get("confidence"), 0.0) for record, _, _ in group_items) >= 0.75 else "medium" if max(_safe_float(record.get("confidence"), 0.0) for record, _, _ in group_items) >= 0.5 else "low",
        "vehicle_info": vehicle_info,
        "license_plate": next((value for value in plate_candidates if _normalized_plate(value)), None),
        "ocr_text": ", ".join(plate_candidates[:4]) or None,
        "bbox_source": "track_or_record",
        "representative_frame_path": best_full_frame_path,
        "record": best_record,
        "source_records": [record for record, _, _ in group_items],
        "source_object_record_ids": _unique_strings([record.get("object_record_id") for record, _, _ in group_items]),
        "priority_rank": best_priority,
    }


def _merge_events(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["start_seconds"] = min(_safe_float(left.get("start_seconds"), 0.0), _safe_float(right.get("start_seconds"), 0.0))
    merged["end_seconds"] = max(_safe_float(left.get("end_seconds"), 0.0), _safe_float(right.get("end_seconds"), 0.0))
    merged["best_timestamp_seconds"] = (
        _safe_float(right.get("best_timestamp_seconds"), 0.0)
        if _safe_float(right.get("confidence"), 0.0) > _safe_float(left.get("confidence"), 0.0)
        else _safe_float(left.get("best_timestamp_seconds"), 0.0)
    )
    merged["confidence"] = max(_safe_float(left.get("confidence"), 0.0), _safe_float(right.get("confidence"), 0.0))
    merged["track_ids"] = _unique_strings(list(left.get("track_ids", [])) + list(right.get("track_ids", [])))
    merged["class_names"] = _unique_strings(list(left.get("class_names", [])) + list(right.get("class_names", [])))
    merged["license_plate"] = left.get("license_plate") or right.get("license_plate")
    merged["ocr_text"] = ", ".join(
        _unique_strings(
            str(left.get("ocr_text") or "").split(",")
            + str(right.get("ocr_text") or "").split(","),
            limit=4,
        )
    ) or None
    merged["source_object_record_ids"] = _unique_strings(
        list(left.get("source_object_record_ids", [])) + list(right.get("source_object_record_ids", []))
    )
    merged["source_records"] = list(left.get("source_records", [])) + list(right.get("source_records", []))
    if _safe_float(right.get("confidence"), 0.0) > _safe_float(left.get("confidence"), 0.0):
        merged["event_id"] = right.get("event_id")
        merged["searchable_event_id"] = right.get("searchable_event_id")
        merged["search_id"] = right.get("search_id")
        merged["summary"] = right.get("summary")
        merged["representative_frame_path"] = right.get("representative_frame_path")
        merged["record"] = right.get("record")
        merged["vehicle_info"] = right.get("vehicle_info")
        merged["event_type"] = right.get("event_type")
        merged["title"] = right.get("title")
    return merged


def _object_event_entries(
    *,
    records: list[dict[str, Any]],
    config: EvidenceVideoConfig,
) -> list[dict[str, Any]]:
    grouped_records: dict[tuple[str, ...], list[tuple[dict[str, Any], str, int]]] = {}
    for record in records:
        include, reason, priority_rank = _important_object_record(record, config.include_person_events)
        if not include or reason is None:
            continue
        grouped_records.setdefault(_object_event_group_key(record), []).append((record, reason, priority_rank))

    candidate_events = [
        merged_event
        for merged_event in (
            _merge_object_event_group(group_items, config)
            for _, group_items in sorted(
                grouped_records.items(),
                key=lambda item: min(
                    _safe_float(record.get("first_seen_seconds", record.get("timestamp_seconds")), 0.0)
                    for record, _, _ in item[1]
                ),
            )
        )
        if merged_event is not None
    ]

    deduped: list[dict[str, Any]] = []
    for event in sorted(candidate_events, key=lambda item: (_safe_float(item.get("best_timestamp_seconds"), 0.0), -_safe_float(item.get("priority_rank"), 0), -_safe_float(item.get("confidence"), 0.0))):
        merged = False
        event_tracks = set(str(track_id) for track_id in list(event.get("track_ids", [])) if str(track_id))
        event_plate = _normalized_plate(event.get("license_plate"))
        event_start = _safe_float(event.get("start_seconds"), 0.0)
        event_end = _safe_float(event.get("end_seconds"), 0.0)
        for index, existing in enumerate(deduped):
            existing_tracks = set(str(track_id) for track_id in list(existing.get("track_ids", [])) if str(track_id))
            existing_plate = _normalized_plate(existing.get("license_plate"))
            existing_start = _safe_float(existing.get("start_seconds"), 0.0)
            existing_end = _safe_float(existing.get("end_seconds"), 0.0)
            same_track = bool(event_tracks and existing_tracks and event_tracks.intersection(existing_tracks))
            same_plate = bool(event_plate and existing_plate and event_plate == existing_plate)
            overlapping = not (event_end < existing_start - 1.0 or existing_end < event_start - 1.0)
            nearby = abs(_safe_float(event.get("best_timestamp_seconds"), 0.0) - _safe_float(existing.get("best_timestamp_seconds"), 0.0)) <= 4.0
            if (same_track and (overlapping or nearby)) or (same_plate and (overlapping or nearby)):
                deduped[index] = _merge_events(existing, event)
                merged = True
                break
        if not merged:
            deduped.append(event)
    return deduped


def select_evidence_events(run_dir: Path, config: EvidenceVideoConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step11_map = _step11_map(run_dir)
    selected_candidates = _step12_selected(run_dir)
    reviews, review_map, step14_summary = _step14_reviews(run_dir)
    step15_scene_events = _step15_scene_events(run_dir)
    search_records = _search_records(run_dir)

    scene_events = step15_scene_events or _scene_event_entries(
        selected_candidates=selected_candidates,
        step11_map=step11_map,
        review_map=review_map,
        config=config,
    )
    object_events = _object_event_entries(records=search_records, config=config)
    all_events = sorted(scene_events + object_events, key=lambda item: (_safe_float(item.get("best_timestamp_seconds"), 0.0), -_safe_float(item.get("priority_rank"), 0)))

    diagnostics = {
        "step12_selected_candidates": len(selected_candidates),
        "step14_reviews_loaded": len(reviews),
        "step14_summary_status": (step14_summary or {}).get("overall_status"),
        "step15_scene_events_loaded": len(step15_scene_events),
        "scene_events_selected": len(scene_events),
        "object_events_selected": len(object_events),
        "search_records_loaded": len(search_records),
    }
    return all_events, diagnostics


def _incident_summary(selected_events: list[dict[str, Any]]) -> tuple[str, str]:
    critical_events = [
        item
        for item in selected_events
        if str(item.get("source_type", "") or "") in {"scene_event_review", "scene_event"}
        and (
            bool(item.get("critical_event"))
            or str(item.get("event_type", "") or "").strip().lower() in {"collision", "near_miss", "sudden_stop"}
        )
    ]
    if not critical_events:
        return "No critical incident retained in final evidence.", "Reviewed scene events were normal-context only."
    labels = []
    timestamps = []
    for item in critical_events[:4]:
        labels.append(str(item.get("event_type", "") or "event").replace("_", " "))
        timestamp_value = _safe_float(item.get("best_timestamp_seconds"), 0.0)
        timestamps.append(format_seconds_text(timestamp_value))
    unique_labels = ", ".join(dict.fromkeys(labels))
    unique_times = ", ".join(dict.fromkeys(timestamps))
    return (
        f"Critical incident evidence: {unique_labels}.",
        f"Key timestamps: {unique_times}.",
    )


def _nearest_frames_within_window(frames: list[dict[str, Any]], start_seconds: float, end_seconds: float) -> list[dict[str, Any]]:
    selected = [
        item
        for item in frames
        if start_seconds <= _safe_float(item.get("timestamp_seconds"), 0.0) <= end_seconds
    ]
    if selected:
        return selected
    if not frames:
        return []
    center = (start_seconds + end_seconds) / 2.0
    nearest = min(frames, key=lambda item: abs(_safe_float(item.get("timestamp_seconds"), 0.0) - center))
    nearest_index = frames.index(nearest)
    start_index = max(0, nearest_index - 1)
    end_index = min(len(frames), nearest_index + 2)
    return frames[start_index:end_index]


def _event_frames(event: dict[str, Any], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_seconds = _safe_float(event.get("start_seconds"), 0.0)
    end_seconds = _safe_float(event.get("end_seconds"), start_seconds)
    selected = _nearest_frames_within_window(frames, start_seconds, end_seconds)
    if len(selected) >= 2 or not frames:
        return selected
    only_frame = selected[0]
    nearest_index = frames.index(only_frame)
    start_index = max(0, nearest_index - 1)
    end_index = min(len(frames), nearest_index + 2)
    return frames[start_index:end_index]


def _bbox_entries_for_frame(
    *,
    event: dict[str, Any],
    frame_id: str,
    track_map: dict[str, dict[str, Any]],
    detection_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    track_ids = [str(track_id or "") for track_id in list(event.get("track_ids", [])) if str(track_id or "")]
    if track_ids:
        for track_id in track_ids:
            detection = detection_index.get((track_id, frame_id))
            if not detection:
                continue
            track = track_map.get(track_id, {})
            entries.append(
                {
                    "track_id": track_id,
                    "class_name": detection.get("class_name") or track.get("dominant_class_name") or "object",
                    "bbox_xyxy": list(detection.get("bbox_xyxy", [])),
                }
            )
    elif event.get("source_type") == "searchable_object_event":
        record = dict(event.get("record", {}))
        if str(record.get("frame_id", "") or "") == frame_id and len(list(record.get("bbox_xyxy", []))) == 4:
            entries.append(
                {
                    "track_id": record.get("track_id") or "-",
                    "class_name": record.get("class_name") or "object",
                    "bbox_xyxy": list(record.get("bbox_xyxy", [])),
                }
            )
    return entries


def _draw_bboxes(frame: np.ndarray, bbox_entries: list[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for item in bbox_entries[:10]:
        bbox = list(item.get("bbox_xyxy", []))
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 215, 255), 2)
        label = f"{item.get('class_name', 'object')} | {item.get('track_id', '-')}"
        cv2.putText(output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
    return output


def _draw_overlay(
    *,
    frame: np.ndarray,
    event: dict[str, Any],
    frame_item: dict[str, Any],
    recording_date: str,
    recording_time: str,
    bbox_entries: list[dict[str, Any]],
) -> np.ndarray:
    output = _draw_bboxes(frame, bbox_entries)
    height, width = output.shape[:2]
    header_height = 92
    footer_height = 88
    cv2.rectangle(output, (0, 0), (width, header_height), (0, 0, 0), thickness=-1)
    cv2.rectangle(output, (0, height - footer_height), (width, height), (0, 0, 0), thickness=-1)

    frame_seconds = _safe_float(frame_item.get("timestamp_seconds"), 0.0)
    video_time_text = _format_precise_timestamp(frame_seconds)
    clip_time_text = f"{format_seconds_text(_safe_float(event.get('start_seconds'), 0.0))} - {format_seconds_text(_safe_float(event.get('end_seconds'), 0.0))}"
    original_timestamp_text = f"{recording_date} {recording_time}" if recording_date != "unavailable" and recording_time != "unavailable" else "unavailable"
    line1 = f"Video Time: {video_time_text} | Original Timestamp: {original_timestamp_text}"
    line2 = f"Event ID: {event.get('event_id')} | Search ID: {event.get('searchable_event_id')} | Confidence: {round(_safe_float(event.get('confidence'), 0.0), 3)}"
    cv2.putText(output, line1, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, line2, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    track_text = ", ".join(list(event.get("track_ids", []))[:4]) or "-"
    vehicle_info = event.get("vehicle_info", {}) if isinstance(event.get("vehicle_info"), dict) else {}
    vehicle_text = " | ".join(
        [
            str(vehicle_info.get("type") or "-"),
            str(vehicle_info.get("color") or "-"),
            str(vehicle_info.get("make") or "-"),
            str(vehicle_info.get("model") or "-"),
        ]
    )
    class_text = ", ".join(_unique_strings(list(event.get("class_names", [])), limit=4)) or "-"
    plate_text = str(event.get("license_plate") or event.get("ocr_text") or "-")
    line3 = f"{str(event.get('title') or '').strip()} | Clip Window: {clip_time_text}"
    line4 = f"Tracks: {track_text} | Labels: {class_text} | Vehicle: {vehicle_text}"
    line5 = f"Plate/OCR: {plate_text}"
    cv2.putText(output, line3, (16, height - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, line4, (16, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, line5, (16, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _draw_gallery_overlay(
    *,
    frame: np.ndarray,
    section_title: str,
    subtitle: str,
    detail_lines: list[str],
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    header_height = 74
    footer_height = 74
    cv2.rectangle(output, (0, 0), (width, header_height), (0, 0, 0), thickness=-1)
    cv2.rectangle(output, (0, height - footer_height), (width, height), (0, 0, 0), thickness=-1)
    cv2.putText(output, section_title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, subtitle, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    trimmed_lines = [str(line or "").strip() for line in detail_lines if str(line or "").strip()][:2]
    for index, line in enumerate(trimmed_lines):
        cv2.putText(
            output,
            line,
            (16, height - 34 + (index * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _create_title_card(width: int, height: int, lines: list[str]) -> np.ndarray:
    card = np.zeros((height, width, 3), dtype=np.uint8)
    y = 130
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        cv2.putText(card, text, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += 54
    return card


def _fit_image_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / float(source_width), height / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    offset_x = max(0, (width - resized_width) // 2)
    offset_y = max(0, (height - resized_height) // 2)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _write_repeated_frames(writer: cv2.VideoWriter, frame: np.ndarray, *, fps: int, seconds: float) -> int:
    count = max(1, int(round(float(fps) * float(seconds))))
    for _ in range(count):
        writer.write(frame)
    return count


def _summary_card(
    *,
    width: int,
    height: int,
    video_info: dict[str, Any],
    evidence_duration_seconds: float,
    searchable_event_count: int,
    gallery_frame_count: int,
    vlm_gallery_frame_count: int,
    vehicle_count: int,
    person_count: int,
    plate_count: int,
    object_count: int,
    average_confidence: float,
    generation_time_seconds: float,
    incident_headline: str,
    incident_detail: str,
) -> np.ndarray:
    lines = [
        "Evidence Video Summary",
        f"Original video duration: {str(video_info.get('duration_text', '-'))}",
        f"Evidence video duration: {format_seconds_text(evidence_duration_seconds)}",
        f"Searchable events: {searchable_event_count}",
        incident_headline,
        incident_detail,
        f"Unique gallery frames: {gallery_frame_count} | VLM gallery frames: {vlm_gallery_frame_count}",
        f"Vehicles: {vehicle_count} | Persons: {person_count} | Plates: {plate_count} | Objects: {object_count}",
        f"Average confidence: {round(average_confidence, 3)} | Generation time: {round(generation_time_seconds, 2)}s",
    ]
    return _create_title_card(width, height, lines)


def _unique_object_frame_entries(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    frames_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        frame_path_value = (
            record.get("full_frame_path")
            or record.get("best_full_frame_path")
            or record.get("representative_frame_path")
        )
        frame_id = str(record.get("frame_id", "") or "")
        normalized_path = _normalize_rel_path(frame_path_value)
        if not normalized_path and frame_id:
            normalized_path = _normalize_rel_path(dict(frames_by_id.get(frame_id, {})).get("image_path"))
        if not normalized_path:
            continue
        key = frame_id or normalized_path
        current = by_key.get(key)
        if current is None:
            by_key[key] = {
                "frame_id": frame_id,
                "frame_path": normalized_path,
                "timestamp_seconds": _safe_float(record.get("timestamp_seconds"), 0.0),
                "records": [record],
            }
            continue
        current["records"].append(record)
        current["timestamp_seconds"] = min(
            _safe_float(current.get("timestamp_seconds"), 0.0),
            _safe_float(record.get("timestamp_seconds"), 0.0),
        )
    aggregated = sorted(
        by_key.values(),
        key=lambda item: (_safe_float(item.get("timestamp_seconds"), 0.0), str(item.get("frame_path", ""))),
    )
    unique_entries: list[dict[str, Any]] = []
    last_kept_by_track_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    last_global: dict[str, Any] | None = None
    for entry in aggregated:
        frame_path = _resolve_run_path(run_dir, str(entry.get("frame_path", "") or ""))
        if frame_path is None or not frame_path.exists():
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        record_items = [record for record in list(entry.get("records", [])) if isinstance(record, dict)]
        track_ids = tuple(sorted(_unique_strings([record.get("track_id") for record in record_items])))
        class_names = tuple(sorted(_unique_strings([record.get("class_name") for record in record_items])))
        plates = tuple(sorted(_unique_strings([record.get("verified_license_plate") for record in record_items], limit=4)))
        bbox_signature: list[tuple[str, str, int, int, int, int]] = []
        for record in record_items:
            bbox = list(record.get("bbox_xyxy", []))
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            bbox_signature.append(
                (
                    str(record.get("track_id", "") or ""),
                    str(record.get("class_name", "") or "object"),
                    int(round((x1 / max(1.0, image_width)) * 20.0)),
                    int(round((y1 / max(1.0, image_height)) * 20.0)),
                    int(round((x2 / max(1.0, image_width)) * 20.0)),
                    int(round((y2 / max(1.0, image_height)) * 20.0)),
                )
            )
        bbox_signature.sort()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hash_image = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        mean_value = float(hash_image.mean())
        fingerprint = (hash_image > mean_value).astype(np.uint8).flatten()
        dedupe_candidate = {
            **entry,
            "track_ids": list(track_ids),
            "class_names": list(class_names),
            "plates": list(plates),
            "bbox_signature": bbox_signature,
            "fingerprint": fingerprint,
            "ocr_text": ", ".join(
                _unique_strings(
                    [record.get("verified_license_plate") for record in record_items]
                    + [record.get("possible_plate_text") for record in record_items]
                    + [text for record in record_items for text in list(record.get("weak_ocr_text", []))],
                    limit=4,
                )
            ) or None,
            "vehicle_descriptions": _unique_strings(
                [
                    " ".join(
                        part
                        for part in [
                            str(record.get("class_name") or "").strip(),
                            str(record.get("verified_vehicle_color") or record.get("possible_vehicle_color") or "").strip(),
                            str(record.get("vehicle_make") or "").strip(),
                            str(record.get("vehicle_model") or "").strip(),
                        ]
                        if part
                    )
                    for record in record_items
                ],
                limit=4,
            ),
        }
        duplicate = False
        candidate_time = _safe_float(dedupe_candidate.get("timestamp_seconds"), 0.0)
        compare_targets = [last_global]
        if track_ids:
            compare_targets.append(last_kept_by_track_signature.get(track_ids))
        for prior in [target for target in compare_targets if target is not None]:
            prior_time = _safe_float(prior.get("timestamp_seconds"), 0.0)
            time_gap = abs(candidate_time - prior_time)
            same_tracks = tuple(prior.get("track_ids", [])) == track_ids and bool(track_ids)
            same_layout = list(prior.get("bbox_signature", [])) == bbox_signature and bool(bbox_signature)
            same_classes = tuple(prior.get("class_names", [])) == class_names
            same_plates = tuple(prior.get("plates", [])) == plates
            hamming_distance = int(np.count_nonzero(prior.get("fingerprint") != fingerprint))
            if same_tracks and same_layout and time_gap <= 3.0 and hamming_distance <= 18:
                duplicate = True
                break
            if same_tracks and same_classes and same_plates and time_gap <= 1.5 and hamming_distance <= 10:
                duplicate = True
                break
            if not track_ids and same_layout and same_classes and time_gap <= 1.0 and hamming_distance <= 8:
                duplicate = True
                break
        if duplicate:
            continue
        unique_entries.append(dedupe_candidate)
        last_global = dedupe_candidate
        if track_ids:
            last_kept_by_track_signature[track_ids] = dedupe_candidate
    return unique_entries


def _vlm_gallery_entries(run_dir: Path) -> list[dict[str, Any]]:
    step11_map = _step11_map(run_dir)
    _, review_map, _ = _step14_reviews(run_dir)
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in _step13_vlm_inputs(run_dir):
        media = dict(item.get("media", {}))
        candidates = [
            ("temporal_strip", _normalize_rel_path(media.get("temporal_strip_path"))),
            ("contact_sheet", _normalize_rel_path(media.get("contact_sheet_path"))),
            ("primary_frame", _normalize_rel_path(media.get("primary_frame_path"))),
        ]
        chosen_type = None
        chosen_path = None
        for media_type, path_value in candidates:
            if not path_value or path_value in seen_paths:
                continue
            resolved = _resolve_run_path(run_dir, path_value)
            if resolved is not None and resolved.exists():
                chosen_type = media_type
                chosen_path = path_value
                break
        if not chosen_path or not chosen_type:
            continue
        seen_paths.add(chosen_path)
        source_candidate_ids = [str(candidate_id or "") for candidate_id in list(item.get("source_candidate_ids", [])) if str(candidate_id or "")]
        matched_review = next((review_map.get(candidate_id) for candidate_id in source_candidate_ids if review_map.get(candidate_id)), None)
        matched_candidate = next((step11_map.get(candidate_id) for candidate_id in source_candidate_ids if step11_map.get(candidate_id)), None)
        summary_text = ""
        if matched_review:
            summary_text = str(dict(matched_review.get("model_review", {})).get("summary_caption", "") or "").strip()
        if not summary_text and matched_candidate:
            summary_text = _candidate_summary(matched_candidate, None)
        entries.append(
            {
                "vlm_input_id": str(item.get("vlm_input_id", "") or ""),
                "best_timestamp_seconds": _safe_float(item.get("best_timestamp_seconds"), 0.0),
                "best_timestamp_text": str(item.get("best_timestamp_text", "") or ""),
                "source_candidate_ids": source_candidate_ids,
                "source_event_types": [str(event_type or "") for event_type in list(item.get("source_event_types", [])) if str(event_type or "")],
                "media_type": chosen_type,
                "media_path": chosen_path,
                "vlm_summary": summary_text or None,
            }
        )
    return sorted(entries, key=lambda item: (_safe_float(item.get("best_timestamp_seconds"), 0.0), str(item.get("vlm_input_id", ""))))


def build_evidence_video(run_dir: Path, config: EvidenceVideoConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_started = time.perf_counter()
    video_info = read_json(run_dir / "01_video_info.json")
    frames, frames_by_id = _frame_catalog(run_dir)
    tracks_payload = _tracks_payload(run_dir)
    track_map, detection_index = _track_map_and_detection_index(tracks_payload)
    selected_events, diagnostics = select_evidence_events(run_dir, config)
    search_records = _search_records(run_dir)
    object_frame_entries = _unique_object_frame_entries(run_dir=run_dir, records=search_records, frames_by_id=frames_by_id)
    vlm_gallery_entries = _vlm_gallery_entries(run_dir)
    recording_date, recording_time = _recording_datetime_strings(video_info, run_dir)

    if not frames:
        raise FileNotFoundError("No sampled/adaptive frames are available for evidence video generation.")
    if not selected_events and not object_frame_entries and not vlm_gallery_entries:
        raise ValueError("No evidence events, object frames, or VLM input frames were selected from the current run outputs.")

    first_frame_path = _resolve_run_path(run_dir, str(frames[0].get("image_path", "") or ""))
    if first_frame_path is None or not first_frame_path.exists():
        raise FileNotFoundError(f"Could not resolve the first frame image: {frames[0].get('image_path')}")
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame image: {first_frame_path}")
    frame_height, frame_width = first_frame.shape[:2]

    video_path = run_dir / EVIDENCE_VIDEO_NAME
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.clip_fps),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    index_entries: list[dict[str, Any]] = []
    evidence_cursor_seconds = 0.0
    unique_vehicle_tracks: set[str] = set()
    unique_person_tracks: set[str] = set()
    unique_plates: set[str] = set()
    unique_objects: set[str] = set()
    confidence_values: list[float] = []
    searchable_event_count = 0
    try:
        title_card = _create_title_card(
            frame_width,
            frame_height,
            [
                "Evidence Video",
                str(video_info.get("video_name") or video_info.get("input_video_path") or "Unknown Video"),
                f"Recording: {recording_date} {recording_time}",
                f"Searchable events: {len(selected_events)} | Detection gallery frames: {len(object_frame_entries)} | VLM inputs: {len(vlm_gallery_entries)}",
            ],
        )
        _write_repeated_frames(writer, title_card, fps=config.clip_fps, seconds=config.header_seconds)
        evidence_cursor_seconds += config.header_seconds

        for event_number, event in enumerate(selected_events, start=1):
            event_frames = _event_frames(event, frames)
            renderable_frames: list[tuple[dict[str, Any], Path]] = []
            for frame_item in event_frames:
                frame_path = _resolve_run_path(run_dir, str(frame_item.get("image_path", "") or ""))
                if frame_path is None or not frame_path.exists():
                    continue
                renderable_frames.append((frame_item, frame_path))
            if not renderable_frames:
                continue

            title_card = _create_title_card(
                frame_width,
                frame_height,
                [
                    f"Event {event_number}",
                    str(event.get("title", "Evidence Event")),
                    f"Time: {format_seconds_text(_safe_float(event.get('best_timestamp_seconds'), 0.0))}",
                    f"Duration: {round(max(0.0, _safe_float(event.get('end_seconds'), 0.0) - _safe_float(event.get('start_seconds'), 0.0)), 2)} seconds",
                ],
            )
            clip_start_seconds = evidence_cursor_seconds
            _write_repeated_frames(writer, title_card, fps=config.clip_fps, seconds=config.header_seconds)
            evidence_cursor_seconds += config.header_seconds
            content_start_seconds = evidence_cursor_seconds

            for frame_item, frame_path in renderable_frames:
                frame_id = str(frame_item.get("frame_id", "") or "")
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                if image.shape[0] != frame_height or image.shape[1] != frame_width:
                    image = cv2.resize(image, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                bbox_entries = _bbox_entries_for_frame(
                    event=event,
                    frame_id=frame_id,
                    track_map=track_map,
                    detection_index=detection_index,
                )
                overlay_frame = _draw_overlay(
                    frame=image,
                    event=event,
                    frame_item=frame_item,
                    recording_date=recording_date,
                    recording_time=recording_time,
                    bbox_entries=bbox_entries,
                )
                writer.write(overlay_frame)
                evidence_cursor_seconds += 1.0 / float(config.clip_fps)

            clip_end_seconds = evidence_cursor_seconds
            searchable_event_count += 1
            class_names = [str(item) for item in list(event.get("class_names", [])) if str(item)]
            track_ids = [str(item) for item in list(event.get("track_ids", [])) if str(item)]
            for track_id in track_ids:
                if track_id.startswith("person_"):
                    unique_person_tracks.add(track_id)
                else:
                    unique_vehicle_tracks.add(track_id)
                unique_objects.add(track_id)
            if not track_ids:
                unique_objects.add(str(event.get("event_id") or event.get("search_id") or f"event_{event_number}"))
            plate_value = str(event.get("license_plate") or "").strip()
            if plate_value:
                unique_plates.add(plate_value)
            confidence_values.append(_safe_float(event.get("confidence"), 0.0))

            index_entries.append(
                {
                    "event_id": event.get("event_id"),
                    "searchable_event_id": event.get("searchable_event_id"),
                    "search_id": event.get("search_id"),
                    "source_type": event.get("source_type"),
                    "clip_start_seconds": round(clip_start_seconds, 3),
                    "clip_content_start_seconds": round(content_start_seconds, 3),
                    "clip_end_seconds": round(clip_end_seconds, 3),
                    "clip_start_text": format_seconds_text(clip_start_seconds),
                    "clip_content_start_text": format_seconds_text(content_start_seconds),
                    "clip_end_text": format_seconds_text(clip_end_seconds),
                    "original_video_time_seconds": round(_safe_float(event.get("best_timestamp_seconds"), 0.0), 3),
                    "original_video_time_text": format_seconds_text(_safe_float(event.get("best_timestamp_seconds"), 0.0)),
                    "summary": event.get("summary"),
                    "event_type": event.get("event_type"),
                    "track_ids": track_ids,
                    "class_names": class_names,
                    "vehicles": event.get("vehicle_info"),
                    "plates": [event.get("license_plate")] if event.get("license_plate") else [],
                    "persons": [track_id for track_id in track_ids if track_id.startswith("person_")],
                    "ocr_text": event.get("ocr_text"),
                    "representative_frame_path": event.get("representative_frame_path"),
                    "confidence": round(_safe_float(event.get("confidence"), 0.0), 6),
                }
            )

        if object_frame_entries:
            section_card = _create_title_card(
                frame_width,
                frame_height,
                [
                    "Unique Object Detection Frames",
                    f"Frames: {len(object_frame_entries)}",
                    "Deduplicated full-scene frames with searchable detected objects",
                ],
            )
            _write_repeated_frames(writer, section_card, fps=config.clip_fps, seconds=config.header_seconds)
            evidence_cursor_seconds += config.header_seconds
            for gallery_index, entry in enumerate(object_frame_entries, start=1):
                frame_path = _resolve_run_path(run_dir, str(entry.get("frame_path", "") or ""))
                if frame_path is None or not frame_path.exists():
                    continue
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                image = _fit_image_to_canvas(image, frame_width, frame_height)
                bbox_entries = []
                class_names: set[str] = set()
                track_ids: set[str] = set()
                plate_values: list[str] = []
                for record in list(entry.get("records", [])):
                    class_names.add(str(record.get("class_name", "") or "object"))
                    track_id = str(record.get("track_id", "") or "")
                    if track_id:
                        track_ids.add(track_id)
                    plate_value = str(record.get("verified_license_plate", "") or "").strip()
                    if plate_value and plate_value not in plate_values:
                        plate_values.append(plate_value)
                    bbox = list(record.get("bbox_xyxy", []))
                    if len(bbox) == 4:
                        bbox_entries.append(
                            {
                                "track_id": track_id or "-",
                                "class_name": str(record.get("class_name", "") or "object"),
                                "bbox_xyxy": bbox,
                            }
                        )
                overlay = _draw_gallery_overlay(
                    frame=_draw_bboxes(image, bbox_entries),
                    section_title=f"Object Frame {gallery_index}/{len(object_frame_entries)}",
                    subtitle=(
                        f"Video Time: {_format_precise_timestamp(_safe_float(entry.get('timestamp_seconds'), 0.0))} | "
                        f"Classes: {', '.join(sorted(class_names)[:4]) or '-'}"
                    ),
                    detail_lines=[
                        f"Tracks: {', '.join(sorted(track_ids)[:4]) or '-'} | Plates: {', '.join(plate_values[:3]) or '-'}",
                        f"Vehicle/OCR: {', '.join(list(entry.get('vehicle_descriptions', []))[:3]) or '-'} | OCR: {entry.get('ocr_text') or '-'}",
                    ],
                )
                clip_start_seconds = evidence_cursor_seconds
                _write_repeated_frames(writer, overlay, fps=config.clip_fps, seconds=1.0)
                evidence_cursor_seconds += 1.0
                index_entries.append(
                    {
                        "event_id": f"object_frame_{gallery_index:03d}",
                        "searchable_event_id": f"object_frame_{gallery_index:03d}",
                        "search_id": f"object_frame_{gallery_index:03d}",
                        "source_type": "object_detection_frame_gallery",
                        "clip_start_seconds": round(clip_start_seconds, 3),
                        "clip_content_start_seconds": round(clip_start_seconds, 3),
                        "clip_end_seconds": round(evidence_cursor_seconds, 3),
                        "clip_start_text": format_seconds_text(clip_start_seconds),
                        "clip_content_start_text": format_seconds_text(clip_start_seconds),
                        "clip_end_text": format_seconds_text(evidence_cursor_seconds),
                        "original_video_time_seconds": round(_safe_float(entry.get("timestamp_seconds"), 0.0), 3),
                        "original_video_time_text": format_seconds_text(_safe_float(entry.get("timestamp_seconds"), 0.0)),
                        "summary": "Unique full-scene frame with detected searchable objects.",
                        "event_type": "object_detection_frame_gallery",
                        "track_ids": sorted(track_ids),
                        "class_names": sorted(class_names),
                        "vehicles": None,
                        "plates": plate_values[:3],
                        "persons": [track_id for track_id in sorted(track_ids) if track_id.startswith("person_")],
                        "ocr_text": entry.get("ocr_text"),
                        "representative_frame_path": entry.get("frame_path"),
                        "confidence": round(
                            max((_safe_float(record.get("confidence"), 0.0) for record in list(entry.get("records", [])) if isinstance(record, dict)), default=0.0),
                            6,
                        ),
                    }
                )

        if vlm_gallery_entries:
            section_card = _create_title_card(
                frame_width,
                frame_height,
                [
                    "VLM Input Gallery",
                    f"Inputs: {len(vlm_gallery_entries)}",
                    "Temporal strips, contact sheets, and primary frames prepared for VLM review",
                ],
            )
            _write_repeated_frames(writer, section_card, fps=config.clip_fps, seconds=config.header_seconds)
            evidence_cursor_seconds += config.header_seconds
            for gallery_index, entry in enumerate(vlm_gallery_entries, start=1):
                media_path = _resolve_run_path(run_dir, str(entry.get("media_path", "") or ""))
                if media_path is None or not media_path.exists():
                    continue
                image = cv2.imread(str(media_path))
                if image is None:
                    continue
                fitted = _fit_image_to_canvas(image, frame_width, frame_height)
                overlay = _draw_gallery_overlay(
                    frame=fitted,
                    section_title=f"VLM Input {gallery_index}/{len(vlm_gallery_entries)}",
                    subtitle=(
                        f"Input ID: {entry.get('vlm_input_id') or '-'} | "
                        f"Video Time: {_format_precise_timestamp(_safe_float(entry.get('best_timestamp_seconds'), 0.0))}"
                    ),
                    detail_lines=[
                        (
                            f"Media: {entry.get('media_type') or '-'} | "
                            f"Candidates: {', '.join(list(entry.get('source_candidate_ids', []))[:3]) or '-'} | "
                            f"Types: {', '.join(list(entry.get('source_event_types', []))[:3]) or '-'}"
                        ),
                        f"Summary: {str(entry.get('vlm_summary') or '-').strip()}",
                    ],
                )
                clip_start_seconds = evidence_cursor_seconds
                _write_repeated_frames(writer, overlay, fps=config.clip_fps, seconds=1.0)
                evidence_cursor_seconds += 1.0
                index_entries.append(
                    {
                        "event_id": str(entry.get("vlm_input_id") or f"vlm_input_{gallery_index:03d}"),
                        "searchable_event_id": str(entry.get("vlm_input_id") or f"vlm_input_{gallery_index:03d}"),
                        "search_id": str(entry.get("vlm_input_id") or f"vlm_input_{gallery_index:03d}"),
                        "source_type": "vlm_input_gallery",
                        "clip_start_seconds": round(clip_start_seconds, 3),
                        "clip_content_start_seconds": round(clip_start_seconds, 3),
                        "clip_end_seconds": round(evidence_cursor_seconds, 3),
                        "clip_start_text": format_seconds_text(clip_start_seconds),
                        "clip_content_start_text": format_seconds_text(clip_start_seconds),
                        "clip_end_text": format_seconds_text(evidence_cursor_seconds),
                        "original_video_time_seconds": round(_safe_float(entry.get("best_timestamp_seconds"), 0.0), 3),
                        "original_video_time_text": format_seconds_text(_safe_float(entry.get("best_timestamp_seconds"), 0.0)),
                        "summary": "Prepared VLM input media.",
                        "event_type": "vlm_input_gallery",
                        "track_ids": [],
                        "class_names": list(entry.get("source_event_types", [])),
                        "vehicles": None,
                        "plates": [],
                        "persons": [],
                        "ocr_text": None,
                        "representative_frame_path": entry.get("media_path"),
                        "vlm_summary": entry.get("vlm_summary"),
                    }
                )

        if not index_entries:
            raise ValueError("Evidence video generation found no renderable event, object-frame, or VLM-input imagery.")

        evidence_duration_before_summary = evidence_cursor_seconds
        generation_time_seconds = time.perf_counter() - generation_started
        incident_headline, incident_detail = _incident_summary(selected_events)
        summary_card = _summary_card(
            width=frame_width,
            height=frame_height,
            video_info=video_info,
            evidence_duration_seconds=evidence_duration_before_summary,
            searchable_event_count=searchable_event_count,
            gallery_frame_count=len(object_frame_entries),
            vlm_gallery_frame_count=len(vlm_gallery_entries),
            vehicle_count=len(unique_vehicle_tracks),
            person_count=len(unique_person_tracks),
            plate_count=len(unique_plates),
            object_count=len(unique_objects),
            average_confidence=(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
            generation_time_seconds=generation_time_seconds,
            incident_headline=incident_headline,
            incident_detail=incident_detail,
        )
        _write_repeated_frames(writer, summary_card, fps=config.clip_fps, seconds=config.summary_seconds)
    finally:
        writer.release()

    index_payload = {
        "status": "success",
        "video_file": EVIDENCE_VIDEO_NAME,
        "event_count": len(index_entries),
        "searchable_event_count": searchable_event_count,
        "object_detection_gallery_frame_count": len(object_frame_entries),
        "vlm_input_gallery_frame_count": len(vlm_gallery_entries),
        "clips": index_entries,
    }
    searchable_source_types = {"searchable_object_event", "scene_event", "scene_event_review"}
    searchable_ids = [
        str(item.get("searchable_event_id") or item.get("event_id") or "")
        for item in index_entries
        if item.get("source_type") in searchable_source_types
    ]
    source_types = [str(item.get("source_type") or "") for item in index_entries]
    object_gallery_start = next((idx for idx, source_type in enumerate(source_types) if source_type == "object_detection_frame_gallery"), None)
    vlm_gallery_start = next((idx for idx, source_type in enumerate(source_types) if source_type == "vlm_input_gallery"), None)
    evidence_duration_seconds = round(evidence_duration_before_summary + config.summary_seconds, 3)
    report_payload = {
        "status": "success",
        "video_file": EVIDENCE_VIDEO_NAME,
        "index_file": EVIDENCE_INDEX_NAME,
        "event_count": len(index_entries),
        "searchable_event_count": searchable_event_count,
        "evidence_duration_seconds": evidence_duration_seconds,
        "original_video_duration_seconds": _safe_float(video_info.get("duration_seconds"), 0.0),
        "vehicles_detected": len(unique_vehicle_tracks),
        "persons_detected": len(unique_person_tracks),
        "license_plates_detected": len(unique_plates),
        "objects_detected": len(unique_objects),
        "average_confidence": round((sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0, 6),
        "final_incident_summary": {
            "headline": incident_headline,
            "detail": incident_detail,
        },
        "generation_time_seconds": round(generation_time_seconds, 3),
        "diagnostics": diagnostics,
        "object_detection_gallery_frame_count": len(object_frame_entries),
        "vlm_input_gallery_frame_count": len(vlm_gallery_entries),
        "compression_ratio": round(
            evidence_duration_seconds / max(_safe_float(video_info.get("duration_seconds"), 0.0), 1e-6),
            6,
        ),
        "validation": {
            "searchable_events_unique": len(searchable_ids) == len(set(searchable_ids)),
            "duplicate_searchable_events_removed": len(searchable_ids) == searchable_event_count,
            "gallery_frames_unique_by_path": len(
                [item.get("representative_frame_path") for item in index_entries if item.get("source_type") == "object_detection_frame_gallery"]
            )
            == len(
                {
                    str(item.get("representative_frame_path") or "")
                    for item in index_entries
                    if item.get("source_type") == "object_detection_frame_gallery"
                }
            ),
            "vlm_gallery_appended_at_end": vlm_gallery_start is not None
            and all(source_type == "vlm_input_gallery" for source_type in source_types[vlm_gallery_start:]),
            "gallery_after_searchable_events": object_gallery_start is None
            or all(
                source_type in {"object_detection_frame_gallery", "vlm_input_gallery"}
                for source_type in source_types[object_gallery_start:]
            ),
            "chronological_order_preserved": searchable_ids == [
                searchable_id
                for _, searchable_id in sorted(
                    (
                        (_safe_float(item.get("original_video_time_seconds"), 0.0), str(item.get("searchable_event_id") or item.get("event_id") or ""))
                        for item in index_entries
                        if item.get("source_type") in searchable_source_types
                    ),
                    key=lambda pair: pair[0],
                )
            ],
            "final_video_shorter_than_original": evidence_duration_seconds < _safe_float(video_info.get("duration_seconds"), 0.0),
        },
        "recording_date": recording_date,
        "recording_time": recording_time,
    }

    write_json(run_dir / EVIDENCE_INDEX_NAME, index_payload)
    write_json(run_dir / REPORT_NAME, report_payload)
    return index_payload, report_payload
