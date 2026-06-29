from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


PRIORITY_SELECTION_REASONS = {
    "mandatory_suspicious_or_high_priority",
    "mandatory_summary_event",
}

PRIORITY_EVENT_LABEL_TERMS = {
    "theft",
    "robbery",
    "intrusion",
    "fight",
    "weapon",
    "fall",
    "collision",
}

STRONG_SUSPICIOUS_TERMS = [
    "theft",
    "robbery",
    "steal",
    "stolen",
    "shoplifting",
    "object removed",
    "hides object",
    "conceals object",
    "reaches into display case",
    "reaching into display case",
    "reaching into a display case",
    "display case",
    "suspicious reaching",
    "weapon",
    "fight",
    "fall",
    "collision",
    "intrusion",
]

WEAK_SUSPICIOUS_TERMS = [
    "reaching",
    "bending",
    "interacting with counter",
    "interacting with display",
    "inspecting items",
    "customer interaction",
]

AMBIGUITY_TERMS = [
    "could indicate",
    "appears to",
    "possibly",
    "possible",
    "may be",
    "might be",
    "inspect",
    "inspecting",
    "unclear",
]


def _load_required_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Top-K final summary input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
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


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown time"
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown time"
    if total_seconds < 0:
        return "unknown time"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds - (hours * 3600) - (minutes * 60)
    if float(remaining_seconds).is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:04.1f}"


def clean_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"```(?:json)?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("`", " ")
    cleaned = cleaned.replace("*", " ")
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"([!?.,])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([!?.,])", r"\1", cleaned)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def extract_best_event_description(parsed_json: dict[str, Any]) -> str:
    events = parsed_json.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                description = clean_text(str(event.get("description", "")))
                if description:
                    return description

    activities = parsed_json.get("activities", [])
    if isinstance(activities, list):
        for activity in activities:
            if isinstance(activity, dict):
                description = clean_text(str(activity.get("description", "")))
                if description:
                    return description

    caption = clean_text(str(parsed_json.get("caption", "")))
    if caption:
        return caption
    return "Selected clip contains visually important activity."


def contains_strong_suspicious_terms(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(term in normalized for term in STRONG_SUSPICIOUS_TERMS)


def contains_weak_suspicious_terms(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(term in normalized for term in WEAK_SUSPICIOUS_TERMS)


def _contains_ambiguity_terms(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(term in normalized for term in AMBIGUITY_TERMS)


def _relative_to_repo(path_value: Any) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if path.is_absolute():
            return path.resolve().relative_to(repo_root).as_posix()
        return path.as_posix()
    except Exception:
        return str(path_value)


def _load_step16_items(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_required_json(run_dir / "16_topk_vlm_outputs.json")
    if not isinstance(payload, dict):
        raise ValueError("Expected an object in 16_topk_vlm_outputs.json")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Expected an 'items' list in 16_topk_vlm_outputs.json")
    return payload, items


def _selection_reason_summary(selection_reasons: list[str]) -> str:
    reasons = {str(reason) for reason in selection_reasons}
    if reasons & PRIORITY_SELECTION_REASONS:
        return "Selected by safety guardrail because earlier analysis marked this clip as suspicious/high-priority."
    if "top_k_ranked" in reasons:
        return "Selected because it was among the highest ranked motion + YOLO clips."
    if "high_motion_guardrail" in reasons:
        return "Selected by high-motion guardrail."
    if "minimum_count_fill" in reasons:
        return "Selected to ensure a minimum review set for the optimized pipeline."
    return "Selected as part of the optimized Top-K review set."


def _build_review_note(final_category: str, best_event_description: str) -> str:
    if final_category == "priority_suspicious_event":
        lowered = best_event_description.lower()
        if "display case" in lowered:
            return "Priority review recommended. The clip contains suspicious visual evidence such as reaching into a display case."
        return f"Priority review recommended. {best_event_description}"
    if final_category == "possible_review_clip":
        return "Review if needed. The clip contains reaching/bending near counters, but may be normal shop activity."
    if final_category == "normal_activity":
        return "Routine activity visible."
    return "Output was unclear or failed parsing; manual review recommended if needed."


def _event_label_has_priority_signal(event_label: str) -> bool:
    normalized = event_label.lower()
    return any(term in normalized for term in PRIORITY_EVENT_LABEL_TERMS)


def _classify_clip(
    parse_success: bool,
    suspicious_activity: str,
    event_label: str,
    risk_level: str,
    selection_reasons: list[str],
    evidence_text: str,
) -> str:
    suspicious_value = suspicious_activity.lower()
    label_value = event_label.lower()
    risk_value = risk_level.lower()
    reasons = {str(reason) for reason in selection_reasons}

    if not parse_success or suspicious_value == "unclear" or label_value == "uncertain_activity" or risk_value == "unknown":
        return "uncertain_activity"

    if suspicious_value == "yes":
        if reasons & PRIORITY_SELECTION_REASONS:
            return "priority_suspicious_event"
        if _event_label_has_priority_signal(label_value):
            return "priority_suspicious_event"
        if contains_strong_suspicious_terms(evidence_text) and not _contains_ambiguity_terms(evidence_text):
            return "priority_suspicious_event"
        if risk_value == "high":
            return "priority_suspicious_event"
        if contains_weak_suspicious_terms(evidence_text) or suspicious_value == "yes":
            return "possible_review_clip"

    if suspicious_value == "no" and label_value == "normal_activity" and risk_value == "low":
        return "normal_activity"

    return "uncertain_activity"


def _extract_yolo_top_classes(yolo_data: dict[str, Any]) -> list[str]:
    top_classes = yolo_data.get("top_classes", [])
    if not isinstance(top_classes, list):
        return []
    class_names: list[str] = []
    for item in top_classes[:5]:
        if isinstance(item, dict):
            class_name = str(item.get("class_name", "")).strip()
            if class_name:
                class_names.append(class_name)
    return class_names


def _safe_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_preserve_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = clean_text(str(value)).rstrip(".")
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if limit is not None and len(result) >= limit:
            break
    return result


def _join_terms_for_sentence(values: list[str], limit: int = 3) -> str:
    items = _dedupe_preserve_order(values, limit=limit)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _scene_area_phrase(scene_type: str) -> str:
    normalized = str(scene_type or "unknown").replace("_", " ").strip()
    if not normalized or normalized == "unknown":
        return "the scene"
    article = "an" if normalized[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {normalized} area"


def _extract_visible_people_notes(parsed_json: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for person in _safe_list_of_dicts(parsed_json.get("visible_people")):
        appearance = clean_text(str(person.get("appearance", ""))).rstrip(".")
        pose = clean_text(str(person.get("pose_or_action", ""))).rstrip(".")
        location = clean_text(str(person.get("location", ""))).rstrip(".")
        note = " ".join(part for part in [appearance, pose, location] if part).strip()
        if note:
            notes.append(note)
    return _dedupe_preserve_order(notes, limit=6)


def _extract_activity_phrases(parsed_json: dict[str, Any], caption: str, description: str) -> list[str]:
    values: list[str] = []
    for activity in _safe_list_of_dicts(parsed_json.get("activities")):
        activity_type = clean_text(str(activity.get("activity_type", ""))).rstrip(".")
        activity_desc = clean_text(str(activity.get("description", ""))).rstrip(".")
        if activity_desc:
            values.append(activity_desc)
        elif activity_type:
            values.append(activity_type.replace("_", " "))
    for text in [caption, description]:
        cleaned = clean_text(text).rstrip(".")
        if cleaned:
            values.append(cleaned)
    return _dedupe_preserve_order(values, limit=8)


def _extract_object_names(parsed_json: dict[str, Any], yolo_top_classes: list[str]) -> list[str]:
    values: list[str] = []
    for obj in _safe_list_of_dicts(parsed_json.get("objects")):
        name = clean_text(str(obj.get("name", ""))).rstrip(".")
        subtype = clean_text(str(obj.get("subtype", ""))).rstrip(".")
        obj_type = clean_text(str(obj.get("type", ""))).rstrip(".")
        label = name or subtype or obj_type
        if label and label.lower() != "unknown":
            values.append(label.replace("_", " "))
    values.extend(str(class_name).replace("_", " ") for class_name in yolo_top_classes if str(class_name).strip())
    return _dedupe_preserve_order(values, limit=8)


def _collect_scene_overview(event_timeline: list[dict[str, Any]]) -> dict[str, Any]:
    scene_counts: dict[str, int] = {}
    activity_terms: list[str] = []
    object_terms: list[str] = []
    people_counts: list[int] = []

    for item in event_timeline:
        scene_type = clean_text(str(item.get("scene_type", ""))).rstrip(".").lower()
        if scene_type and scene_type != "unknown":
            scene_counts[scene_type] = scene_counts.get(scene_type, 0) + 1
        activity_terms.extend(item.get("activity_descriptions", []) if isinstance(item.get("activity_descriptions"), list) else [])
        object_terms.extend(item.get("object_names", []) if isinstance(item.get("object_names"), list) else [])
        people_counts.append(_safe_int(item.get("people_count"), 0))

    dominant_scene_type = "unknown"
    if scene_counts:
        dominant_scene_type = sorted(scene_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]

    return {
        "dominant_scene_type": dominant_scene_type,
        "common_activities": _dedupe_preserve_order(activity_terms, limit=6),
        "common_objects": _dedupe_preserve_order(object_terms, limit=6),
        "people_count_observed": {
            "min": min(people_counts) if people_counts else 0,
            "max": max(people_counts) if people_counts else 0,
        },
    }


def build_descriptive_video_summary(event_timeline: list[dict[str, Any]], processing_summary: dict) -> str:
    total_clips = _safe_int(processing_summary.get("topk_inputs"), len(event_timeline))
    priority_items = [item for item in event_timeline if item.get("final_category") == "priority_suspicious_event"]
    review_items = [item for item in event_timeline if item.get("final_category") == "possible_review_clip"]
    normal_items = [item for item in event_timeline if item.get("final_category") == "normal_activity"]

    scene_overview = _collect_scene_overview(event_timeline)
    scene_type = scene_overview.get("dominant_scene_type", "unknown")
    activity_text = _join_terms_for_sentence(scene_overview.get("common_activities", []), limit=3)
    object_text = _join_terms_for_sentence(scene_overview.get("common_objects", []), limit=3)
    people_range = scene_overview.get("people_count_observed", {})
    max_people = _safe_int(people_range.get("max"), 0) if isinstance(people_range, dict) else 0

    if priority_items:
        first_priority = sorted(
            priority_items,
            key=lambda item: (_safe_float(item.get("start_time"), 0.0), _safe_int(item.get("selection_order"), 999999)),
        )[0]
        first_time = format_seconds(_safe_float(first_priority.get("start_time"), 0.0))
        priority_description = clean_text(str(first_priority.get("best_event_description", ""))).rstrip(".")
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips. "
            f"It identified {len(priority_items)} priority suspicious event(s). "
            f"The main priority event occurs around {first_time}, where {priority_description}."
        )
        if review_items:
            summary += (
                f" Additional clips were marked for review because they show {activity_text or 'similar movement near the scene'}."
            )
        return clean_text(summary)

    if review_items:
        scene_phrase = f"in {_scene_area_phrase(scene_type)}" if scene_type != "unknown" else "in the selected scene"
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips. "
            f"No priority suspicious event was confirmed, but {len(review_items)} clip(s) were marked for review. "
            f"The selected clips show people moving {scene_phrase}"
        )
        if activity_text:
            summary += f", including {activity_text}"
        if object_text:
            summary += f", around {object_text}"
        summary += ". Review is recommended for clips where people are bending or reaching near display areas."
        return clean_text(summary)

    if normal_items:
        scene_phrase = f"in {_scene_area_phrase(scene_type)}" if scene_type != "unknown" else "in the scene"
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips from the video. "
            f"The selected evidence mainly shows routine activity {scene_phrase}. "
        )
        if max_people > 0 and activity_text:
            summary += f" People are visible, with activity such as {activity_text}"
        elif activity_text:
            summary += f" The clips mainly show {activity_text}"
        elif max_people > 0:
            summary += " People are visible in the selected clips"
        if object_text:
            summary += f" around {object_text}"
        summary += ". No priority suspicious event was detected in the selected clips."
        return clean_text(summary)

    return clean_text(
        f"The optimized Top-K pipeline analyzed {total_clips} selected clips. Several clips remain uncertain and may need review."
    )


def _build_summary_item(item: dict[str, Any]) -> dict[str, Any]:
    parse_success = bool(item.get("parse_success"))
    parsed_json = item.get("parsed_json", {})
    if not isinstance(parsed_json, dict):
        parsed_json = {}
        parse_success = False

    selection_reasons = item.get("selection_reasons", [])
    if not isinstance(selection_reasons, list):
        selection_reasons = []

    ranking_reasons = item.get("ranking_reasons", [])
    if not isinstance(ranking_reasons, list):
        ranking_reasons = []

    motion = item.get("motion", {})
    if not isinstance(motion, dict):
        motion = {}

    yolo = item.get("yolo", {})
    if not isinstance(yolo, dict):
        yolo = {}

    events = parsed_json.get("events", [])
    activities = parsed_json.get("activities", [])
    keywords = parsed_json.get("keywords", [])

    if not isinstance(events, list):
        events = []
    if not isinstance(activities, list):
        activities = []
    if not isinstance(keywords, list):
        keywords = []

    caption = clean_text(str(parsed_json.get("caption", "")))
    best_event_description = extract_best_event_description(parsed_json)
    scene_type = clean_text(str(parsed_json.get("scene_type", ""))).rstrip(".").lower() or "unknown"
    yolo_top_classes = _extract_yolo_top_classes(yolo)
    visible_people = _extract_visible_people_notes(parsed_json)
    activity_descriptions = _extract_activity_phrases(parsed_json, caption, best_event_description)
    object_names = _extract_object_names(parsed_json, yolo_top_classes)
    event_descriptions = _dedupe_preserve_order(
        [clean_text(str(event.get("description", ""))).rstrip(".") for event in events if isinstance(event, dict)],
        limit=6,
    )
    combined_evidence_text = " ".join(
        [
            caption,
            best_event_description,
            clean_text(str(parsed_json.get("event_label", ""))),
            " ".join(clean_text(str(event.get("description", ""))) for event in events if isinstance(event, dict)),
            " ".join(clean_text(str(activity.get("description", ""))) for activity in activities if isinstance(activity, dict)),
        ]
    )

    suspicious_activity = str(parsed_json.get("suspicious_activity", "unclear")).strip().lower() or "unclear"
    risk_level = str(parsed_json.get("risk_level", "unknown")).strip().lower() or "unknown"
    event_label = clean_text(str(parsed_json.get("event_label", ""))).rstrip(".")
    confidence = clean_text(str(parsed_json.get("confidence", ""))).rstrip(".") or "unknown"

    final_category = _classify_clip(
        parse_success=parse_success,
        suspicious_activity=suspicious_activity,
        event_label=event_label,
        risk_level=risk_level,
        selection_reasons=selection_reasons,
        evidence_text=combined_evidence_text,
    )

    start_time = item.get("start_time")
    end_time = item.get("end_time")
    yolo_person_max = _safe_int(yolo.get("person_max"), 0)
    summary_item = {
        "clip_id": item.get("source_clip_id"),
        "selection_order": item.get("selection_order"),
        "rank": item.get("rank"),
        "time_range": f"{format_seconds(_safe_float(start_time, 0.0))} - {format_seconds(_safe_float(end_time, 0.0))}",
        "start_time": start_time,
        "end_time": end_time,
        "expanded_start_time": item.get("expanded_start_time"),
        "expanded_end_time": item.get("expanded_end_time"),
        "final_category": final_category,
        "event_label": event_label or "unknown",
        "risk_level": risk_level,
        "confidence": confidence,
        "suspicious_activity": suspicious_activity,
        "caption": caption,
        "best_event_description": best_event_description,
        "people_count": _safe_int(parsed_json.get("people_count"), yolo_person_max),
        "selection_reasons": selection_reasons,
        "ranking_reasons": ranking_reasons,
        "ranked_clip_score": item.get("ranked_clip_score"),
        "motion_score": motion.get("clip_motion_score", motion.get("clip_score", 0.0)),
        "yolo_person_max": yolo_person_max,
        "yolo_top_classes": yolo_top_classes,
        "scene_type": scene_type,
        "visible_people": visible_people,
        "activity_descriptions": activity_descriptions,
        "object_names": object_names,
        "event_descriptions": event_descriptions,
        "strip_path": item.get("strip_path"),
        "top_annotated_frame_path": item.get("top_annotated_frame_path"),
        "why_selected": _selection_reason_summary(selection_reasons),
        "review_note": "",
    }
    summary_item["review_note"] = _build_review_note(final_category, best_event_description)
    return summary_item


def _build_top_keywords(items: list[dict[str, Any]], limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        for keyword in item.get("keywords", []):
            normalized = clean_text(str(keyword)).rstrip(".")
            if normalized:
                counts[normalized] = counts.get(normalized, 0) + 1
    sorted_keywords = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0].lower()))
    return [keyword for keyword, _ in sorted_keywords[:limit]]


def _build_final_summary_text(
    total_clips: int,
    priority_items: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
    normal_items: list[dict[str, Any]],
) -> str:
    if priority_items:
        def priority_sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
            reasons = {str(reason) for reason in item.get("selection_reasons", [])}
            if reasons & PRIORITY_SELECTION_REASONS:
                return (0, _safe_int(item.get("selection_order"), 999999), _safe_float(item.get("start_time"), 0.0))
            return (1, _safe_int(item.get("selection_order"), 999999), _safe_float(item.get("start_time"), 0.0))

        first_priority = sorted(priority_items, key=priority_sort_key)[0]
        first_time = format_seconds(_safe_float(first_priority.get("start_time"), 0.0))
        description = clean_text(str(first_priority.get("best_event_description", ""))).rstrip(".")
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips instead of all motion clips. "
            f"It identified {len(priority_items)} priority suspicious event(s). "
            f"The main priority event occurs around {first_time}, where {description}."
        )
        if review_items:
            summary += (
                " Additional clips are marked for possible review because they show reaching or bending near "
                "counters/display cases."
            )
        return clean_text(summary)

    if review_items:
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips. "
            f"No confirmed priority event was produced, but {len(review_items)} clips were marked for possible review "
            "because they show reaching or bending near counters/display cases."
        )
        return clean_text(summary)

    if normal_items:
        summary = (
            f"The optimized Top-K pipeline analyzed {total_clips} selected clips. "
            "The selected clips mainly show routine activity with people interacting near counters or display areas."
        )
        return clean_text(summary)

    return clean_text(
        f"The optimized Top-K pipeline analyzed {total_clips} selected clips. Several clips remain uncertain and may need manual review."
    )


def _markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["| Clip | Time | Description |", "| --- | --- | --- |"]
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _build_markdown_report(summary: dict[str, Any]) -> str:
    processing = summary.get("processing_summary", {})
    priority_items = summary.get("priority_suspicious_events", [])
    review_items = summary.get("possible_review_clips", [])
    normal_items = summary.get("normal_activity_clips", [])
    evidence_assets = summary.get("evidence_assets", {})

    lines = [
        "# Top-K Tender Demo Final Summary",
        "",
        "## Processing Summary",
        "",
        f"* Top-K/guardrail clips analyzed: {processing.get('topk_inputs', 0)}",
        f"* Successful parses: {processing.get('successful_parses', 0)}",
        f"* Failed parses: {processing.get('failed_parses', 0)}",
        f"* Priority suspicious events: {processing.get('priority_suspicious_events', 0)}",
        f"* Possible review clips: {processing.get('possible_review_clips', 0)}",
        f"* Normal clips: {processing.get('normal_activity_clips', 0)}",
        f"* Uncertain clips: {processing.get('uncertain_clips', 0)}",
        "",
        "## Overall Summary",
        "",
        summary.get("final_summary_text", ""),
        "",
        "## Priority Suspicious Events",
        "",
    ]

    if priority_items:
        for item in priority_items:
            lines.extend(
                [
                    f"### {item.get('clip_id', 'unknown_clip')}",
                    "",
                    f"* Time: {item.get('time_range', 'unknown time')}",
                    f"* Risk: {item.get('risk_level', 'unknown')}",
                    f"* Confidence: {item.get('confidence', 'unknown')}",
                    f"* Event label: {item.get('event_label', 'unknown')}",
                    f"* Description: {item.get('best_event_description', '')}",
                    f"* Selection reasons: {', '.join(item.get('selection_reasons', [])) or 'none'}",
                    f"* Temporal strip: {item.get('strip_path', '')}",
                    f"* Annotated YOLO frame: {item.get('top_annotated_frame_path', '')}",
                    "",
                ]
            )
    else:
        lines.extend(["No priority suspicious events were identified in the optimized Top-K set.", ""])

    lines.extend(["## Possible Review Clips", ""])
    review_rows = [["Clip", "Time", "Reason", "Description"]]
    for item in review_items:
        review_rows.append(
            [
                str(item.get("clip_id", "")),
                str(item.get("time_range", "")),
                str(item.get("event_label", "")),
                str(item.get("best_event_description", "")),
            ]
        )
    lines.extend(_markdown_table(review_rows))
    lines.append("")

    lines.extend(["## Normal Activity Clips", ""])
    normal_rows = [["Clip", "Time", "Description"]]
    for item in normal_items:
        normal_rows.append(
            [
                str(item.get("clip_id", "")),
                str(item.get("time_range", "")),
                str(item.get("best_event_description", "")),
            ]
        )
    lines.extend(_markdown_table(normal_rows))
    lines.append("")

    lines.extend(["## Evidence Assets", ""])
    lines.append(f"* Top-K VLM inputs folder: {evidence_assets.get('topk_vlm_inputs_folder', '')}")
    lines.append(f"* Top-K VLM outputs: {evidence_assets.get('topk_vlm_outputs', '')}")
    for asset in evidence_assets.get("priority_clip_assets", []):
        if not isinstance(asset, dict):
            continue
        lines.append(
            f"* {asset.get('clip_id', 'unknown_clip')}: strip={asset.get('strip_path', '')}, annotated={asset.get('top_annotated_frame_path', '')}"
        )
    lines.append("")
    return "\n".join(lines)


def create_topk_final_summary(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 17: Top-K final summary")

    payload, items = _load_step16_items(run_dir)
    print(f"[tender-demo] Top-K outputs read: {len(items)}")

    video_info = _load_optional_json(run_dir / "01_video_info.json")
    _load_optional_json(run_dir / "14_selected_top_clips.json")
    _load_optional_json(run_dir / "14_selected_top_clips_report.json")
    _load_optional_json(run_dir / "11_yolo_usefulness_report.json")

    if not isinstance(video_info, dict):
        video_info = {}

    summary_items: list[dict[str, Any]] = []
    keyword_source_items: list[dict[str, Any]] = []
    for item in items:
        try:
            summary_item = _build_summary_item(item if isinstance(item, dict) else {})
            parsed_json = item.get("parsed_json", {}) if isinstance(item, dict) else {}
            if not isinstance(parsed_json, dict):
                parsed_json = {}
            summary_item["keywords"] = parsed_json.get("keywords", []) if isinstance(parsed_json.get("keywords", []), list) else []
            summary_items.append(summary_item)
            keyword_source_items.append(summary_item)
        except Exception as exc:
            start_time = item.get("start_time") if isinstance(item, dict) else None
            end_time = item.get("end_time") if isinstance(item, dict) else None
            summary_items.append(
                {
                    "clip_id": item.get("source_clip_id") if isinstance(item, dict) else None,
                    "selection_order": item.get("selection_order") if isinstance(item, dict) else None,
                    "rank": item.get("rank") if isinstance(item, dict) else None,
                    "time_range": f"{format_seconds(_safe_float(start_time, 0.0))} - {format_seconds(_safe_float(end_time, 0.0))}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "expanded_start_time": item.get("expanded_start_time") if isinstance(item, dict) else None,
                    "expanded_end_time": item.get("expanded_end_time") if isinstance(item, dict) else None,
                    "final_category": "uncertain_activity",
                    "event_label": "uncertain_activity",
                    "risk_level": "unknown",
                    "confidence": "unknown",
                    "suspicious_activity": "unclear",
                    "caption": "",
                    "best_event_description": "Selected clip contains visually important activity.",
                    "people_count": 0,
                    "selection_reasons": item.get("selection_reasons", []) if isinstance(item, dict) else [],
                    "ranking_reasons": item.get("ranking_reasons", []) if isinstance(item, dict) else [],
                    "ranked_clip_score": item.get("ranked_clip_score") if isinstance(item, dict) else None,
                    "motion_score": item.get("motion", {}).get("clip_motion_score", 0.0) if isinstance(item, dict) and isinstance(item.get("motion"), dict) else 0.0,
                    "yolo_person_max": 0,
                    "yolo_top_classes": [],
                    "strip_path": item.get("strip_path") if isinstance(item, dict) else None,
                    "top_annotated_frame_path": item.get("top_annotated_frame_path") if isinstance(item, dict) else None,
                    "why_selected": _selection_reason_summary(item.get("selection_reasons", []) if isinstance(item, dict) and isinstance(item.get("selection_reasons"), list) else []),
                    "review_note": f"Output was unclear or failed parsing; manual review recommended if needed. Classification error: {exc}",
                    "keywords": [],
                }
            )

    summary_items.sort(key=lambda entry: (_safe_float(entry.get("start_time"), 0.0), _safe_int(entry.get("selection_order"), 999999)))

    priority_items = [item for item in summary_items if item.get("final_category") == "priority_suspicious_event"]
    review_items = [item for item in summary_items if item.get("final_category") == "possible_review_clip"]
    normal_items = [item for item in summary_items if item.get("final_category") == "normal_activity"]
    uncertain_items = [item for item in summary_items if item.get("final_category") == "uncertain_activity"]

    scene_overview = _collect_scene_overview(summary_items)
    descriptive_summary = build_descriptive_video_summary(
        event_timeline=summary_items,
        processing_summary={
            "topk_inputs": payload.get("total_inputs", len(items)),
            "priority_suspicious_events": len(priority_items),
            "possible_review_clips": len(review_items),
            "normal_activity_clips": len(normal_items),
        },
    )
    normal_activity_summary = (
        build_descriptive_video_summary(
            event_timeline=normal_items,
            processing_summary={"topk_inputs": len(normal_items)},
        )
        if normal_items
        else ""
    )
    final_summary_text = descriptive_summary or _build_final_summary_text(
        total_clips=len(summary_items),
        priority_items=priority_items,
        review_items=review_items,
        normal_items=normal_items,
    )

    output_json_path = run_dir / "17_topk_final_summary.json"
    output_md_path = run_dir / "17_topk_final_summary.md"

    summary = {
        "video_info": video_info,
        "processing_summary": {
            "topk_inputs": payload.get("total_inputs", len(items)),
            "successful_parses": payload.get("successful_outputs", sum(1 for item in items if isinstance(item, dict) and item.get("parse_success") is True)),
            "failed_parses": payload.get("failed_outputs", sum(1 for item in items if not isinstance(item, dict) or item.get("parse_success") is not True)),
            "priority_suspicious_events": len(priority_items),
            "possible_review_clips": len(review_items),
            "normal_activity_clips": len(normal_items),
            "uncertain_clips": len(uncertain_items),
        },
        "scene_overview": scene_overview,
        "descriptive_summary": descriptive_summary,
        "normal_activity_summary": normal_activity_summary,
        "final_summary_text": final_summary_text,
        "overall_summary": final_summary_text,
        "priority_suspicious_events": priority_items,
        "possible_review_clips": review_items,
        "normal_activity_clips": normal_items,
        "uncertain_clips": uncertain_items,
        "event_timeline": summary_items,
        "evidence_assets": {
            "topk_vlm_inputs_folder": "15_topk_vlm_inputs/",
            "topk_vlm_outputs": "16_topk_vlm_outputs.json",
            "priority_clip_assets": [
                {
                    "clip_id": item.get("clip_id"),
                    "strip_path": item.get("strip_path"),
                    "top_annotated_frame_path": item.get("top_annotated_frame_path"),
                }
                for item in priority_items
            ],
        },
        "files": {
            "topk_vlm_outputs": "16_topk_vlm_outputs.json",
            "topk_final_summary_json": "17_topk_final_summary.json",
            "topk_final_summary_markdown": "17_topk_final_summary.md",
        },
    }

    for timeline_item in summary["event_timeline"]:
        timeline_item.pop("keywords", None)
        if timeline_item.get("strip_path"):
            timeline_item["strip_path"] = _relative_to_repo(timeline_item["strip_path"])
        if timeline_item.get("top_annotated_frame_path"):
            timeline_item["top_annotated_frame_path"] = _relative_to_repo(timeline_item["top_annotated_frame_path"])

    for asset in summary["evidence_assets"]["priority_clip_assets"]:
        if asset.get("strip_path"):
            asset["strip_path"] = _relative_to_repo(asset["strip_path"])
        if asset.get("top_annotated_frame_path"):
            asset["top_annotated_frame_path"] = _relative_to_repo(asset["top_annotated_frame_path"])

    summary["video_info"] = {
        **summary["video_info"],
        "video_path": _relative_to_repo(summary["video_info"].get("video_path")) or summary["video_info"].get("video_path"),
    }
    summary["processing_summary"]["top_keywords"] = _build_top_keywords(keyword_source_items)

    output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md_path.write_text(_build_markdown_report(summary), encoding="utf-8")

    print(f"[tender-demo] Priority suspicious events: {len(priority_items)}")
    print(f"[tender-demo] Possible review clips: {len(review_items)}")
    print(f"[tender-demo] Normal clips: {len(normal_items)}")
    print(f"[tender-demo] Uncertain clips: {len(uncertain_items)}")
    print(f"[tender-demo] Output path for 17_topk_final_summary.json: {output_json_path}")
    print(f"[tender-demo] Output path for 17_topk_final_summary.md: {output_md_path}")

    return summary
