from __future__ import annotations

import json
from pathlib import Path


HIGH_RISK_EVENT_TYPES = {
    "weapon_visible",
    "fire",
    "medical_emergency",
    "physical_altercation",
}

MEDIUM_RISK_EVENT_TYPES = {
    "possible_robbery",
    "possible_theft",
    "abandoned_object",
    "object_removed",
    "collision",
    "fall",
    "crowd_formation",
}

SUSPICIOUS_EVENT_TYPES = HIGH_RISK_EVENT_TYPES | MEDIUM_RISK_EVENT_TYPES | {
    "intrusion",
    "unauthorized_entry",
    "possible_vandalism",
}

POSITIVE_SUSPICIOUS_KEYWORDS = {
    "suspicious",
    "robbery",
    "theft",
    "stealing",
    "assault",
    "fight",
    "violence",
    "attack",
    "fall",
    "fallen",
    "chase",
    "running away",
    "panic",
    "weapon",
    "aggressive",
    "abnormal",
    "unusual",
    "emergency",
    "collision",
    "accident",
    "intrusion",
    "trespassing",
}

NEGATIVE_NORMAL_PHRASES = {
    "no obvious signs of suspicious activity",
    "no suspicious activity",
    "no immediate indications of suspicious behavior",
    "routine",
    "normal",
    "standard",
    "typical",
    "calm",
    "no clear robbery",
    "no clear assault",
    "no clear theft",
    "appears routine",
}

LOW_CONFIDENCE_PHRASES = {
    "uncertain",
    "not enough",
    "not clear",
    "may indicate",
    "possible",
}


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds - (hours * 3600) - (minutes * 60)
    if remaining_seconds.is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:04.1f}"


def _load_json_if_exists(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _determine_risk_level(parse_success: bool, events: list[dict]) -> str:
    if not parse_success:
        return "unknown"

    event_types = {str(event.get("event_type", "")).strip() for event in events}
    if event_types & HIGH_RISK_EVENT_TYPES:
        return "high"
    if event_types & MEDIUM_RISK_EVENT_TYPES:
        return "medium"
    if not events:
        return "low"
    return "low"


def _top_keywords_from_timeline(event_timeline: list[dict], limit: int = 5) -> list[str]:
    keyword_counts: dict[str, int] = {}
    for item in event_timeline:
        for keyword in item.get("keywords", []):
            normalized = str(keyword).strip()
            if not normalized:
                continue
            keyword_counts[normalized] = keyword_counts.get(normalized, 0) + 1

    sorted_keywords = sorted(
        keyword_counts.items(),
        key=lambda pair: (-pair[1], pair[0].lower()),
    )
    return [keyword for keyword, _ in sorted_keywords[:limit]]


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _event_types_from_events(events: list[dict]) -> set[str]:
    return {
        _normalize_text(event.get("event_type")).lower()
        for event in events
        if _normalize_text(event.get("event_type"))
    }


def _clean_description_text(raw_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        while line.startswith("#"):
            line = line[1:].strip()
        while line.startswith(("-", "*", "•")):
            line = line[1:].strip()
        if line.endswith(":") and line.lower() in {
            "visible people:",
            "objects:",
            "actions:",
        }:
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    joined_text = " ".join(cleaned_lines)
    sentence_candidates = [part.strip() for part in joined_text.split(".") if part.strip()]
    if sentence_candidates:
        selected_sentences = sentence_candidates[:2]
        return ". ".join(selected_sentences) + "."
    return joined_text


def _build_description(caption: str, raw_vlm_output: str) -> str:
    if _normalize_text(caption):
        return _normalize_text(caption)
    return _clean_description_text(raw_vlm_output)


def _determine_suspicious_activity(
    parse_success: bool,
    parsed_json: dict | None,
    events: list[dict],
    raw_vlm_output: str,
) -> str:
    if parse_success and isinstance(parsed_json, dict) and isinstance(events, list):
        event_types = _event_types_from_events(events)
        if event_types & SUSPICIOUS_EVENT_TYPES:
            return "yes"
        if not events:
            return "no"

    raw_text = raw_vlm_output.lower()
    if any(phrase in raw_text for phrase in NEGATIVE_NORMAL_PHRASES):
        return "no"
    if any(keyword in raw_text for keyword in POSITIVE_SUSPICIOUS_KEYWORDS):
        return "yes"
    return "unclear"


def _determine_event_label(
    suspicious_activity: str,
    event_types: set[str],
    raw_vlm_output: str,
) -> str:
    raw_text = raw_vlm_output.lower()

    def contains_any(terms: set[str]) -> bool:
        return bool(event_types & terms) or any(term in raw_text for term in terms)

    if suspicious_activity == "yes":
        if contains_any({"possible_robbery", "possible_theft", "robbery", "theft", "stealing"}):
            return "possible_theft_or_robbery"
        if contains_any({"physical_altercation", "fight", "assault", "attack", "violence", "aggressive"}):
            return "possible_physical_altercation"
        if contains_any({"fall", "fallen"}):
            return "possible_fall"
        if contains_any({"collision", "accident"}):
            return "possible_collision_or_accident"
        if contains_any({"intrusion", "trespassing", "unauthorized_entry"}):
            return "possible_intrusion"
        return "suspicious_activity"
    if suspicious_activity == "no":
        return "normal_activity"
    return "uncertain_activity"


def _determine_confidence(
    parse_success: bool,
    suspicious_activity: str,
    event_types: set[str],
    raw_vlm_output: str,
) -> str:
    if not parse_success:
        return "low"
    raw_text = raw_vlm_output.lower()
    if any(phrase in raw_text for phrase in LOW_CONFIDENCE_PHRASES):
        return "low"
    if suspicious_activity == "yes" and event_types & SUSPICIOUS_EVENT_TYPES:
        return "high"
    return "medium"


def _build_final_summary_text(event_timeline: list[dict], summary_stats: dict[str, object]) -> str:
    total_clips = int(summary_stats["total_vlm_clips"])
    scene_types = list(summary_stats["unique_scene_types"])
    scene_phrase = ", ".join(scene_types[:3]) if scene_types else "unknown scenes"

    base_text = (
        f"This video was reduced into {total_clips} important motion-based clips. "
        f"The analyzed clips mostly show {scene_phrase}."
    )

    medium_or_high_items = [
        item for item in event_timeline if item.get("risk_level") in {"medium", "high"}
    ]
    if not medium_or_high_items:
        return (
            base_text
            + " No suspicious or high-risk incident was detected in the selected clips."
        )

    first_item = min(medium_or_high_items, key=lambda item: float(item.get("start_time", 0.0)))
    start_label = format_seconds(float(first_item.get("start_time", 0.0)))
    end_label = format_seconds(float(first_item.get("end_time", 0.0)))

    observed_event_types: list[str] = []
    for item in medium_or_high_items:
        for event in item.get("events", []):
            event_type = str(event.get("event_type", "")).strip()
            if event_type and event_type not in observed_event_types:
                observed_event_types.append(event_type)

    event_phrase = ", ".join(observed_event_types[:3]) if observed_event_types else "potentially important activity"
    return (
        base_text
        + f" Potentially important events were observed between {start_label} and {end_label}, "
        + f"including {event_phrase}."
    )


def _collect_visible_activity_signals(event_timeline: list[dict]) -> dict[str, object]:
    scene_counts: dict[str, int] = {}
    activity_counts: dict[str, int] = {}
    keyword_counts: dict[str, int] = {}
    people_present_clips = 0
    empty_scene_clips = 0
    captions: list[str] = []

    for item in event_timeline:
        scene_type = _normalize_text(item.get("scene_type"))
        if scene_type and scene_type != "unknown":
            scene_counts[scene_type] = scene_counts.get(scene_type, 0) + 1

        people_count = int(item.get("people_count", 0) or 0)
        if people_count > 0:
            people_present_clips += 1
        else:
            empty_scene_clips += 1

        caption = _normalize_text(item.get("caption") or item.get("description"))
        if caption:
            captions.append(caption)

        for activity in item.get("activities", []):
            normalized_activity = _normalize_text(activity)
            if normalized_activity:
                activity_counts[normalized_activity] = activity_counts.get(normalized_activity, 0) + 1

        for keyword in item.get("keywords", []):
            normalized_keyword = _normalize_text(keyword)
            if normalized_keyword:
                keyword_counts[normalized_keyword] = keyword_counts.get(normalized_keyword, 0) + 1

    top_scenes = [
        scene
        for scene, _ in sorted(scene_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
    ]
    top_activities = [
        activity
        for activity, _ in sorted(activity_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
    ]
    top_keywords = [
        keyword
        for keyword, _ in sorted(keyword_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    ]

    return {
        "top_scenes": top_scenes,
        "top_activities": top_activities,
        "top_keywords": top_keywords,
        "people_present_clips": people_present_clips,
        "empty_scene_clips": empty_scene_clips,
        "captions": captions,
    }


def _build_key_observations(suspicious_events: list[dict], normal_events: list[dict]) -> list[str]:
    if not suspicious_events:
        visible_signals = _collect_visible_activity_signals(normal_events)
        observations: list[str] = []

        top_scenes = visible_signals["top_scenes"]
        top_activities = visible_signals["top_activities"]
        people_present_clips = int(visible_signals["people_present_clips"])
        empty_scene_clips = int(visible_signals["empty_scene_clips"])
        captions = list(visible_signals["captions"])

        if top_scenes:
            scene_phrase = ", ".join(top_scenes[:2])
            observations.append(f"The selected clips mainly show routine {scene_phrase} activity.")
        else:
            observations.append("The selected clips mainly show routine activity in the most active segments.")

        if top_activities:
            activity_phrase = ", ".join(top_activities[:3])
            observations.append(f"Common visible actions include {activity_phrase}.")
        elif people_present_clips > 0:
            observations.append("People are seen moving or standing within the selected scenes.")

        if people_present_clips > 0:
            observations.append("People are visible in several selected clips, often near work or activity areas.")

        if empty_scene_clips > 0:
            observations.append("Some selected clips show briefly active but otherwise empty areas.")

        if captions:
            observations.append(captions[0])

        observations.append("Motion-based selection reduced the video to the most active segments.")
        deduped_observations: list[str] = []
        for observation in observations:
            if observation not in deduped_observations:
                deduped_observations.append(observation)
        return deduped_observations[:4]

    else:
        return [
            f"The system detected {len(suspicious_events)} potentially suspicious event(s)."
        ]

    observations = [f"The system detected {len(suspicious_events)} potentially suspicious event(s)."]
    for item in suspicious_events[:3]:
        start_label = format_seconds(float(item.get("start_time", 0.0)))
        end_label = format_seconds(float(item.get("end_time", 0.0)))
        observations.append(
            f"{start_label} to {end_label}: {item.get('event_label', 'suspicious_activity')}."
        )
    return observations


def _build_overall_summary(suspicious_events: list[dict]) -> str:
    if not suspicious_events:
        return ""

    unique_labels: list[str] = []
    timestamps: list[str] = []
    for item in suspicious_events:
        label = _normalize_text(item.get("event_label"))
        if label and label not in unique_labels:
            unique_labels.append(label)
        timestamps.append(format_seconds(float(item.get("start_time", 0.0))))

    labels_text = ", ".join(unique_labels[:3]) if unique_labels else "suspicious_activity"
    timestamp_text = ", ".join(timestamps[:3]) if timestamps else "unknown times"
    return (
        "The video contains multiple motion/activity segments. "
        f"The system identified {len(suspicious_events)} potentially suspicious event(s), including "
        f"{labels_text} around {timestamp_text}. These clips should be reviewed manually."
    )


def _build_normal_overall_summary(event_timeline: list[dict], summary_stats: dict[str, object]) -> str:
    visible_signals = _collect_visible_activity_signals(event_timeline)
    total_clips = int(summary_stats["total_vlm_clips"])
    top_scenes = list(visible_signals["top_scenes"])
    top_activities = list(visible_signals["top_activities"])
    people_present_clips = int(visible_signals["people_present_clips"])
    empty_scene_clips = int(visible_signals["empty_scene_clips"])
    captions = list(visible_signals["captions"])

    summary_parts = [f"The video was reduced into {total_clips} selected motion clips."]

    if top_scenes:
        summary_parts.append(f"The clips mainly show {', '.join(top_scenes[:2])} activity.")
    else:
        summary_parts.append("The clips mainly show routine movement in the selected areas.")

    if captions:
        summary_parts.append(captions[0])
    elif top_activities:
        summary_parts.append(
            f"Visible actions include {', '.join(top_activities[:3])}."
        )

    if people_present_clips > 0 and empty_scene_clips > 0:
        summary_parts.append("Some segments include visible people, while others show briefly active but empty areas.")
    elif people_present_clips > 0:
        summary_parts.append("Several selected clips include visible people in the scene.")
    elif empty_scene_clips > 0:
        summary_parts.append("Several selected clips show empty areas with momentary motion.")

    return " ".join(summary_parts)


def create_final_summary(run_dir: Path) -> dict:
    vlm_outputs_path = run_dir / "08_vlm_outputs.json"
    if not vlm_outputs_path.exists():
        raise FileNotFoundError(f"Missing VLM outputs file: {vlm_outputs_path}")

    vlm_outputs = json.loads(vlm_outputs_path.read_text(encoding="utf-8"))
    if not isinstance(vlm_outputs, list):
        raise ValueError(f"Expected a list in VLM outputs file: {vlm_outputs_path}")

    video_info = _load_json_if_exists(run_dir / "01_video_info.json")
    if not isinstance(video_info, dict):
        video_info = {}

    event_timeline: list[dict] = []
    total_people_observations = 0
    unique_scene_types: set[str] = set()
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    failed_parse_count = 0

    for item in vlm_outputs:
        parsed_json = item.get("parsed_json")
        parse_success = bool(item.get("parse_success"))
        if not parse_success:
            failed_parse_count += 1

        if isinstance(parsed_json, dict):
            scene_type = str(parsed_json.get("scene_type", "unknown"))
            caption = str(parsed_json.get("caption", ""))
            people_count = int(parsed_json.get("people_count", 0) or 0)
            objects = parsed_json.get("objects", [])
            activities = parsed_json.get("activities", [])
            events = parsed_json.get("events", [])
            keywords = parsed_json.get("keywords", [])
        else:
            scene_type = "unknown"
            caption = ""
            people_count = 0
            objects = []
            activities = []
            events = []
            keywords = []
        raw_vlm_output = _normalize_text(item.get("raw_qwen_output"))
        event_types = _event_types_from_events(events if isinstance(events, list) else [])

        risk_level = _determine_risk_level(parse_success=parse_success, events=events if isinstance(events, list) else [])
        if risk_level == "high":
            high_risk_count += 1
        elif risk_level == "medium":
            medium_risk_count += 1
        elif risk_level == "low":
            low_risk_count += 1

        if scene_type:
            unique_scene_types.add(scene_type)

        total_people_observations += people_count

        suspicious_activity = _determine_suspicious_activity(
            parse_success=parse_success,
            parsed_json=parsed_json if isinstance(parsed_json, dict) else None,
            events=events if isinstance(events, list) else [],
            raw_vlm_output=raw_vlm_output,
        )
        event_label = _determine_event_label(
            suspicious_activity=suspicious_activity,
            event_types=event_types,
            raw_vlm_output=raw_vlm_output,
        )
        confidence = _determine_confidence(
            parse_success=parse_success,
            suspicious_activity=suspicious_activity,
            event_types=event_types,
            raw_vlm_output=raw_vlm_output,
        )
        description = _build_description(caption=caption, raw_vlm_output=raw_vlm_output)

        timeline_item = {
            "event_id": f"event_{len(event_timeline) + 1:06d}",
            "clip_id": item.get("clip_id"),
            "vlm_input_id": item.get("vlm_input_id"),
            "start_time": item.get("start_time", item.get("source_start_time")),
            "end_time": item.get("end_time", item.get("source_end_time")),
            "current_time": item.get("current_time"),
            "expanded_start_time": item.get("expanded_start_time"),
            "expanded_end_time": item.get("expanded_end_time"),
            "clip_score": item.get("clip_score", item.get("clip_motion_score")),
            "clip_motion_score": item.get("clip_motion_score", item.get("clip_score")),
            "scene_type": scene_type,
            "caption": caption,
            "people_count": people_count,
            "objects": objects if isinstance(objects, list) else [],
            "activities": activities if isinstance(activities, list) else [],
            "events": events if isinstance(events, list) else [],
            "keywords": keywords if isinstance(keywords, list) else [],
            "risk_level": risk_level,
            "event_label": event_label,
            "suspicious_activity": suspicious_activity,
            "confidence": confidence,
            "description": description,
            "strip_path": item.get("strip_path"),
            "raw_vlm_output": raw_vlm_output,
            "parse_success": parse_success,
            "parse_error": item.get("parse_error"),
        }
        event_timeline.append(timeline_item)

    event_timeline.sort(key=lambda item: float(item.get("start_time", 0.0) or 0.0))
    suspicious_events = [item for item in event_timeline if item.get("suspicious_activity") == "yes"]
    normal_events = [item for item in event_timeline if item.get("suspicious_activity") == "no"]
    uncertain_events = [item for item in event_timeline if item.get("suspicious_activity") not in {"yes", "no"}]

    summary_stats = {
        "video_name": video_info.get("video_name"),
        "duration_seconds": video_info.get("duration_seconds"),
        "total_vlm_clips": len(vlm_outputs),
        "successfully_parsed_outputs": len(vlm_outputs) - failed_parse_count,
        "failed_parses": failed_parse_count,
        "total_people_observations": total_people_observations,
        "high_risk_events": high_risk_count,
        "medium_risk_events": medium_risk_count,
        "low_risk_events": low_risk_count,
        "unique_scene_types": sorted(unique_scene_types),
        "top_keywords": _top_keywords_from_timeline(event_timeline),
    }
    processing_summary = {
        "total_vlm_outputs": len(vlm_outputs),
        "successful_outputs": len(vlm_outputs) - failed_parse_count,
        "failed_outputs": failed_parse_count,
        "suspicious_events_count": len(suspicious_events),
        "normal_events_count": len(normal_events),
        "uncertain_events_count": len(uncertain_events),
    }

    final_summary_text = _build_final_summary_text(event_timeline, summary_stats)
    overall_summary = (
        _build_overall_summary(suspicious_events)
        if suspicious_events
        else _build_normal_overall_summary(event_timeline, summary_stats)
    )
    key_observations = _build_key_observations(suspicious_events, normal_events)

    final_summary = {
        "video_info": video_info,
        "summary_stats": summary_stats,
        "processing_summary": processing_summary,
        "final_summary_text": final_summary_text,
        "overall_summary": overall_summary,
        "event_timeline": event_timeline,
        "suspicious_events": suspicious_events,
        "normal_events": normal_events,
        "key_observations": key_observations,
        "files": {
            "vlm_outputs": "08_vlm_outputs.json",
            "vlm_inputs": "07_vlm_inputs.json",
        },
    }

    output_path = run_dir / "09_final_summary.json"
    output_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total VLM outputs: {len(vlm_outputs)}")
    print(f"[tender-demo] Successful outputs: {len(vlm_outputs) - failed_parse_count}")
    print(f"[tender-demo] Failed outputs: {failed_parse_count}")
    print(f"[tender-demo] Suspicious events count: {len(suspicious_events)}")
    print(f"[tender-demo] Normal events count: {len(normal_events)}")
    print(f"[tender-demo] Final summary output path: {output_path}")
    return final_summary
