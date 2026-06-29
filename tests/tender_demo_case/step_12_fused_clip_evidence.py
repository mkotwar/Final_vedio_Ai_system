from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

JSONISH_TOKENS_TO_REMOVE = {
    "scene_type",
    "caption",
    "people_count",
    "objects",
    "activities",
    "relationships",
    "events",
    "event_type",
    "description",
    "actors",
    "severity",
    "id",
    "type",
    "subtype",
    "color",
    "condition",
}


def _load_required_json(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required fusion input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> list[dict[str, Any]] | dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round6(value: float) -> float:
    return round(float(value), 6)


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown time"
    total_seconds = _safe_float(seconds, default=-1.0)
    if total_seconds < 0:
        return "unknown time"
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds - (hours * 3600) - (minutes * 60)
    if float(remaining_seconds).is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:04.1f}"


def clean_evidence_text(text: str) -> str:
    cleaned = str(text or "")
    for token in JSONISH_TOKENS_TO_REMOVE:
        cleaned = cleaned.replace(f'"{token}"', " ")
        cleaned = cleaned.replace(f"{token} :", " ")
        cleaned = cleaned.replace(f"{token}:", " ")

    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    cleaned = cleaned.replace('"', " ").replace("'", " ")
    cleaned = cleaned.replace(",", " ")
    cleaned = cleaned.replace("_", " ")

    for fragment in [
        "person 1",
        "person 2",
        "vehicle 1",
        "id :",
        "type : person",
        "subtype : man",
        "color : black",
        "condition : bending",
    ]:
        cleaned = cleaned.replace(fragment, " ")

    cleaned = " ".join(cleaned.split()).strip(" .,:;-")
    if not cleaned:
        return ""
    return cleaned


def normalize_visual_evidence_sentence(text: str) -> str:
    cleaned = clean_evidence_text(text)
    replacements = {
        "Person bending over display case reaching into it": "A person bends over and reaches into a display case",
        "Person bending over display case reaching into": "A person bends over and reaches into a display case",
        "Person interacting with display case": "A person interacts with a display case",
        "Person standing near display case": "A person stands near a display case",
    }
    for source, target in replacements.items():
        if cleaned.lower() == source.lower():
            cleaned = target
            break

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned or "The selected clip contains visually important activity."


def extract_clean_visual_evidence(vlm_data: dict, raw_text: str = "") -> str:
    events = vlm_data.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                event_description = normalize_visual_evidence_sentence(str(event.get("description", "")))
                if event_description and event_description != "The selected clip contains visually important activity.":
                    return event_description

    description = normalize_visual_evidence_sentence(str(vlm_data.get("description", "")))
    if description and description != "The selected clip contains visually important activity.":
        return description

    raw_lower = raw_text.lower()
    if '"description"' in raw_lower:
        marker_index = raw_lower.find('"description"')
        trailing_text = raw_text[marker_index + len('"description"') :]
        extracted = normalize_visual_evidence_sentence(trailing_text)
        if extracted and extracted != "The selected clip contains visually important activity.":
            return extracted

    caption = normalize_visual_evidence_sentence(str(vlm_data.get("caption", "")))
    if caption and caption != "The selected clip contains visually important activity.":
        return caption

    return "The selected clip contains visually important activity."


def _match_yolo_frames_for_clip(
    clip_item: dict[str, Any],
    yolo_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded_start = clip_item.get("expanded_start_time")
    expanded_end = clip_item.get("expanded_end_time")
    start_time = _safe_float(
        expanded_start if expanded_start is not None else clip_item.get("start_time"),
        0.0,
    )
    end_time = _safe_float(
        expanded_end if expanded_end is not None else clip_item.get("end_time"),
        0.0,
    )
    return [
        item
        for item in yolo_items
        if start_time <= _safe_float(item.get("timestamp_seconds"), -1.0) <= end_time
    ]


def _find_timeline_item(
    timeline: list[dict[str, Any]],
    clip_id: str | None,
    vlm_input_id: str | None,
) -> dict[str, Any] | None:
    for item in timeline:
        if clip_id and item.get("clip_id") == clip_id:
            return item
        if vlm_input_id and item.get("vlm_input_id") == vlm_input_id:
            return item
    return None


def _find_clip_metadata(
    clips: list[dict[str, Any]],
    clip_id: str | None,
) -> dict[str, Any] | None:
    if not clip_id:
        return None
    for item in clips:
        if item.get("clip_id") == clip_id:
            return item
    return None


def _build_yolo_section(matching_yolo_frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not matching_yolo_frames:
        return {
            "matching_frames_count": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "person_max": 0,
            "person_avg": 0.0,
            "vehicle_max": 0,
            "important_object_max": 0,
            "top_classes": [],
            "top_evidence_frames": [],
        }

    frames_with_detections = sum(
        1 for item in matching_yolo_frames if _safe_int(item.get("detection_count")) > 0
    )
    total_detections = sum(_safe_int(item.get("detection_count")) for item in matching_yolo_frames)
    person_counts = [_safe_int(item.get("person_count")) for item in matching_yolo_frames]
    vehicle_counts = [_safe_int(item.get("vehicle_count")) for item in matching_yolo_frames]
    important_counts = [_safe_int(item.get("important_object_count")) for item in matching_yolo_frames]

    class_counts: dict[str, int] = {}
    for item in matching_yolo_frames:
        for class_name, count in item.get("object_counts", {}).items():
            class_counts[str(class_name)] = class_counts.get(str(class_name), 0) + _safe_int(count)

    top_classes = [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    ]

    top_evidence_frames = [
        {
            "frame_idx": item.get("frame_idx"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "object_importance_score": item.get("object_importance_score", 0.0),
            "person_count": item.get("person_count", 0),
            "vehicle_count": item.get("vehicle_count", 0),
            "important_object_count": item.get("important_object_count", 0),
            "object_classes_present": item.get("object_classes_present", []),
            "evidence_labels": item.get("evidence_labels", []),
            "frame_path": item.get("frame_path"),
            "annotated_frame_path": item.get("annotated_frame_path"),
        }
        for item in sorted(
            matching_yolo_frames,
            key=lambda frame: _safe_float(frame.get("object_importance_score"), 0.0),
            reverse=True,
        )[:3]
    ]

    return {
        "matching_frames_count": len(matching_yolo_frames),
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "person_max": max(person_counts) if person_counts else 0,
        "person_avg": _round6(sum(person_counts) / len(person_counts)) if person_counts else 0.0,
        "vehicle_max": max(vehicle_counts) if vehicle_counts else 0,
        "important_object_max": max(important_counts) if important_counts else 0,
        "top_classes": top_classes,
        "top_evidence_frames": top_evidence_frames,
    }


def _build_vlm_section(
    vlm_item: dict[str, Any],
    timeline_item: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed_json = vlm_item.get("parsed_json")
    parse_success = bool(vlm_item.get("parse_success"))

    if isinstance(parsed_json, dict):
        scene_type = parsed_json.get("scene_type", "unknown")
        caption = parsed_json.get("caption", "")
        people_count = parsed_json.get("people_count", 0)
        activities = parsed_json.get("activities", [])
        events = parsed_json.get("events", [])
        keywords = parsed_json.get("keywords", [])
    else:
        scene_type = "unknown"
        caption = ""
        people_count = 0
        activities = []
        events = []
        keywords = []

    return {
        "parse_success": parse_success,
        "scene_type": timeline_item.get("scene_type", scene_type) if timeline_item else scene_type,
        "caption": timeline_item.get("caption", caption) if timeline_item else caption,
        "people_count": timeline_item.get("people_count", people_count) if timeline_item else people_count,
        "activities": timeline_item.get("activities", activities) if timeline_item else activities,
        "events": timeline_item.get("events", events) if timeline_item else events,
        "keywords": timeline_item.get("keywords", keywords) if timeline_item else keywords,
        "risk_level": timeline_item.get("risk_level", "unknown") if timeline_item else "unknown",
        "event_label": timeline_item.get("event_label", "uncertain_activity") if timeline_item else "uncertain_activity",
        "suspicious_activity": timeline_item.get("suspicious_activity", "unclear") if timeline_item else "unclear",
        "confidence": timeline_item.get("confidence", "low") if timeline_item else "low",
        "description": timeline_item.get("description", "") if timeline_item else "",
    }


def _build_motion_section(
    clip_metadata: dict[str, Any] | None,
    expanded_clip: dict[str, Any],
    vlm_item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "clip_score": vlm_item.get("clip_score", clip_metadata.get("clip_score") if clip_metadata else 0.0),
        "clip_motion_score": vlm_item.get(
            "clip_motion_score",
            clip_metadata.get("clip_motion_score", clip_metadata.get("max_motion_score_norm", 0.0))
            if clip_metadata
            else 0.0,
        ),
        "selection_reason": (
            clip_metadata.get("reason")
            if clip_metadata
            else expanded_clip.get("reason", "unknown")
        ),
    }


def _compute_vlm_risk_component(vlm_section: dict[str, Any], parse_success: bool) -> float:
    suspicious_activity = str(vlm_section.get("suspicious_activity", "unclear"))
    risk_level = str(vlm_section.get("risk_level", "unknown"))

    if suspicious_activity == "yes":
        if risk_level == "high":
            return 1.0
        if risk_level == "medium":
            return 0.85
        return 0.65
    if suspicious_activity == "unclear":
        return 0.45
    if suspicious_activity == "no":
        return 0.20
    if not parse_success:
        return 0.35
    return 0.35


def _compute_review_priority(
    suspicious_activity: str,
    risk_level: str,
    fused_evidence_score: float,
) -> str:
    if suspicious_activity == "yes":
        if risk_level == "high":
            return "critical"
        return "high"
    if suspicious_activity == "unclear":
        return "medium" if fused_evidence_score >= 0.6 else "low"
    return "medium" if fused_evidence_score >= 0.7 else "low"


def _build_evidence_summary(
    vlm_section: dict[str, Any],
    yolo_section: dict[str, Any],
) -> str:
    suspicious_activity = str(vlm_section.get("suspicious_activity", "unclear"))
    event_label = str(vlm_section.get("event_label", "uncertain_activity"))
    description = normalize_visual_evidence_sentence(str(vlm_section.get("description", "")).strip())
    person_max = _safe_int(yolo_section.get("person_max"))

    if suspicious_activity == "yes":
        clean_visual_evidence = extract_clean_visual_evidence(vlm_section, str(vlm_section.get("description", "")))
        summary = f"VLM labeled this clip as {event_label}."
        summary += f" The visual evidence shows {clean_visual_evidence.rstrip('.').lower()}."
        if person_max > 0:
            summary += f" YOLO found up to {person_max} people in matching frames and provides annotated evidence frames."
        elif yolo_section.get("matching_frames_count", 0) > 0:
            summary += " YOLO provides annotated evidence frames."
        return summary

    if suspicious_activity == "no":
        summary = "VLM describes routine activity"
        if description:
            summary += f": {clean_evidence_text(description).rstrip('.')}"
        summary += "."
        if person_max > 0:
            summary += " YOLO confirms people are visible in the matching frames."
        elif yolo_section.get("matching_frames_count", 0) > 0:
            summary += " YOLO provides additional frame-level object evidence."
        return summary

    summary = "VLM output was uncertain or partially failed."
    if yolo_section.get("matching_frames_count", 0) > 0:
        summary += " YOLO evidence frames are attached for manual inspection."
    return summary


def run_fused_clip_evidence(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 12: fused clip evidence")

    candidate_clips = _load_required_json(run_dir / "05_candidate_clips.json")
    expanded_clips = _load_required_json(run_dir / "06_expanded_clips.json")
    vlm_inputs = _load_required_json(run_dir / "07_vlm_inputs.json")
    vlm_outputs = _load_required_json(run_dir / "08_vlm_outputs.json")
    final_summary = _load_optional_json(run_dir / "09_final_summary.json")
    yolo_scores = _load_optional_json(run_dir / "11_yolo_object_scores.json")
    _load_optional_json(run_dir / "11_yolo_usefulness_report.json")

    if not isinstance(candidate_clips, list):
        raise ValueError("Expected a list in 05_candidate_clips.json")
    if not isinstance(expanded_clips, list):
        raise ValueError("Expected a list in 06_expanded_clips.json")
    if not isinstance(vlm_inputs, list):
        raise ValueError("Expected a list in 07_vlm_inputs.json")
    if not isinstance(vlm_outputs, list):
        raise ValueError("Expected a list in 08_vlm_outputs.json")
    if yolo_scores is None:
        print("[tender-demo] Warning: 11_yolo_object_scores.json is missing; proceeding without YOLO evidence")
        yolo_scores = []
    if not isinstance(yolo_scores, list):
        raise ValueError("Expected a list in 11_yolo_object_scores.json")

    timeline = []
    video_info: dict[str, Any] = {}
    if isinstance(final_summary, dict):
        timeline = final_summary.get("event_timeline", [])
        video_info = final_summary.get("video_info", {})
    if not isinstance(timeline, list):
        timeline = []
    if not isinstance(video_info, dict):
        video_info = {}

    expanded_by_clip_id = {item.get("clip_id"): item for item in expanded_clips if isinstance(item, dict)}
    inputs_by_vlm_input_id = {item.get("vlm_input_id"): item for item in vlm_inputs if isinstance(item, dict)}

    fused_items: list[dict[str, Any]] = []
    clips_with_yolo_evidence = 0
    clips_with_people = 0
    clips_with_important_objects = 0
    suspicious_clips = 0
    uncertain_clips = 0
    normal_clips = 0
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for vlm_item in vlm_outputs:
        clip_id = vlm_item.get("clip_id")
        vlm_input_id = vlm_item.get("vlm_input_id")
        expanded_clip = expanded_by_clip_id.get(clip_id, {})
        input_item = inputs_by_vlm_input_id.get(vlm_input_id, {})
        timeline_item = _find_timeline_item(timeline, clip_id, vlm_input_id)
        clip_metadata = _find_clip_metadata(candidate_clips, clip_id)
        matching_yolo_frames = _match_yolo_frames_for_clip(expanded_clip or vlm_item, yolo_scores)
        if not matching_yolo_frames and yolo_scores:
            print(f"[tender-demo] Warning: no YOLO frames matched clip {clip_id or vlm_input_id}")

        motion_section = _build_motion_section(clip_metadata, expanded_clip, vlm_item)
        vlm_section = _build_vlm_section(vlm_item, timeline_item)
        yolo_section = _build_yolo_section(matching_yolo_frames)

        motion_score_component = _clamp01(
            _safe_float(
                motion_section.get("clip_motion_score", motion_section.get("clip_score", 0.0)),
                _safe_float(motion_section.get("clip_score", 0.0)),
            )
        )
        vlm_risk_component = _clamp01(
            _compute_vlm_risk_component(vlm_section, bool(vlm_section.get("parse_success")))
        )
        top_yolo_frame_score = 0.0
        if yolo_section["top_evidence_frames"]:
            top_yolo_frame_score = _safe_float(
                yolo_section["top_evidence_frames"][0].get("object_importance_score"),
                0.0,
            )
        yolo_object_component = _clamp01(top_yolo_frame_score)
        person_density_component = _clamp01(_safe_int(yolo_section.get("person_max")) / 3.0)
        important_object_component = 1.0 if _safe_int(yolo_section.get("important_object_max")) > 0 else 0.0

        fused_evidence_score = _round6(
            (0.35 * vlm_risk_component)
            + (0.25 * motion_score_component)
            + (0.20 * yolo_object_component)
            + (0.10 * person_density_component)
            + (0.10 * important_object_component)
        )

        suspicious_activity = str(vlm_section.get("suspicious_activity", "unclear"))
        risk_level = str(vlm_section.get("risk_level", "unknown"))
        review_priority = _compute_review_priority(
            suspicious_activity=suspicious_activity,
            risk_level=risk_level,
            fused_evidence_score=fused_evidence_score,
        )

        evidence_summary = _build_evidence_summary(vlm_section=vlm_section, yolo_section=yolo_section)

        fused_item = {
            "clip_id": clip_id,
            "vlm_input_id": vlm_input_id,
            "start_time": vlm_item.get("start_time", expanded_clip.get("start_time")),
            "end_time": vlm_item.get("end_time", expanded_clip.get("end_time")),
            "current_time": vlm_item.get("current_time", input_item.get("current_time")),
            "start_time_formatted": format_seconds(vlm_item.get("start_time", expanded_clip.get("start_time"))),
            "end_time_formatted": format_seconds(vlm_item.get("end_time", expanded_clip.get("end_time"))),
            "current_time_formatted": format_seconds(vlm_item.get("current_time", input_item.get("current_time"))),
            "expanded_start_time": expanded_clip.get("expanded_start_time", input_item.get("expanded_start_time")),
            "expanded_end_time": expanded_clip.get("expanded_end_time", input_item.get("expanded_end_time")),
            "strip_path": vlm_item.get("strip_path", input_item.get("strip_path")),
            "motion": motion_section,
            "vlm": vlm_section,
            "yolo": yolo_section,
            "score_components": {
                "motion_score_component": _round6(motion_score_component),
                "vlm_risk_component": _round6(vlm_risk_component),
                "yolo_object_component": _round6(yolo_object_component),
                "person_density_component": _round6(person_density_component),
                "important_object_component": _round6(important_object_component),
            },
            "fused_evidence_score": fused_evidence_score,
            "review_priority": review_priority,
            "evidence_summary": evidence_summary,
        }
        fused_items.append(fused_item)

        if yolo_section["matching_frames_count"] > 0:
            clips_with_yolo_evidence += 1
        if _safe_int(yolo_section.get("person_max")) > 0:
            clips_with_people += 1
        if _safe_int(yolo_section.get("important_object_max")) > 0:
            clips_with_important_objects += 1
        if suspicious_activity == "yes":
            suspicious_clips += 1
        elif suspicious_activity == "unclear":
            uncertain_clips += 1
        else:
            normal_clips += 1
        priority_counts[review_priority] = priority_counts.get(review_priority, 0) + 1

    fused_items.sort(
        key=lambda item: (
            PRIORITY_ORDER.get(str(item.get("review_priority", "low")), 99),
            -_safe_float(item.get("fused_evidence_score"), 0.0),
            _safe_float(item.get("start_time"), 0.0),
        )
    )

    fused_output_path = run_dir / "12_fused_clip_evidence.json"
    fused_output_path.write_text(json.dumps(fused_items, indent=2), encoding="utf-8")

    top_priority_clips = []
    for item in fused_items[:10]:
        top_annotated_frame_path = None
        top_frames = item.get("yolo", {}).get("top_evidence_frames", [])
        if top_frames:
            top_annotated_frame_path = top_frames[0].get("annotated_frame_path")
        top_priority_clips.append(
            {
                "clip_id": item.get("clip_id"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "start_time_formatted": format_seconds(item.get("start_time")),
                "end_time_formatted": format_seconds(item.get("end_time")),
                "review_priority": item.get("review_priority"),
                "fused_evidence_score": item.get("fused_evidence_score"),
                "event_label": item.get("vlm", {}).get("event_label"),
                "suspicious_activity": item.get("vlm", {}).get("suspicious_activity"),
                "risk_level": item.get("vlm", {}).get("risk_level"),
                "evidence_summary": item.get("evidence_summary"),
                "strip_path": item.get("strip_path"),
                "top_annotated_frame_path": top_annotated_frame_path,
            }
        )

    if suspicious_clips > 0:
        top_clip = fused_items[0]
        clip_word = "clip" if suspicious_clips == 1 else "clips"
        fused_summary = (
            "The fusion stage combined motion selection, VLM event labels, and YOLO object evidence. "
            f"It identified {suspicious_clips} {clip_word} for priority review, including "
            f"{top_clip.get('vlm', {}).get('event_label', 'suspicious_activity')} around "
            f"{format_seconds(top_clip.get('start_time'))}."
        )
        recommendation = (
            "Review the high-priority fused clips first using both the temporal strip and annotated YOLO evidence frames."
        )
    elif clips_with_people > 0 or clips_with_important_objects > 0:
        fused_summary = (
            "The fusion stage confirms that the selected motion clips contain visible human activity and object-rich scenes. "
            "No priority incident label was produced by the VLM stage."
        )
        recommendation = (
            "Use the fused evidence file to inspect the highest-motion and highest-object-density clips first."
        )
    else:
        fused_summary = (
            "The fusion stage used VLM and motion evidence primarily because YOLO detections were limited."
        )
        recommendation = (
            "Use the fused evidence file to inspect the highest-motion and highest-object-density clips first."
        )

    report = {
        "video_name": video_info.get("video_name"),
        "duration_seconds": video_info.get("duration_seconds"),
        "total_clips": len(fused_items),
        "clips_with_yolo_evidence": clips_with_yolo_evidence,
        "clips_with_people": clips_with_people,
        "clips_with_important_objects": clips_with_important_objects,
        "suspicious_clips": suspicious_clips,
        "uncertain_clips": uncertain_clips,
        "normal_clips": normal_clips,
        "priority_counts": priority_counts,
        "top_priority_clips": top_priority_clips,
        "fused_summary": fused_summary,
        "recommendation": recommendation,
    }

    report_output_path = run_dir / "12_fused_evidence_report.json"
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total clips fused: {len(fused_items)}")
    print(f"[tender-demo] Clips with YOLO evidence: {clips_with_yolo_evidence}")
    print(f"[tender-demo] Suspicious clips: {suspicious_clips}")
    print(f"[tender-demo] Uncertain clips: {uncertain_clips}")
    print(f"[tender-demo] Normal clips: {normal_clips}")
    print(f"[tender-demo] Priority counts: {priority_counts}")
    print(f"[tender-demo] Fused clip evidence output path: {fused_output_path}")
    print(f"[tender-demo] Fused evidence report output path: {report_output_path}")

    return {
        "fused_items": fused_items,
        "report": report,
        "fused_output_path": str(fused_output_path),
        "report_output_path": str(report_output_path),
    }
