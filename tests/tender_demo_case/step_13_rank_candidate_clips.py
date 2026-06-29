from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_required_json(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required ranking input file: {path}")
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


def _match_yolo_frames(
    clip_item: dict[str, Any],
    yolo_scores: list[dict[str, Any]],
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
        for item in yolo_scores
        if start_time <= _safe_float(item.get("timestamp_seconds"), -1.0) <= end_time
    ]


def _build_yolo_summary(matching_frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not matching_frames:
        return {
            "matching_yolo_frames_count": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "person_max": 0,
            "person_avg": 0.0,
            "vehicle_max": 0,
            "important_object_max": 0,
            "object_score_max": 0.0,
            "object_score_avg": 0.0,
            "top_classes": [],
            "top_yolo_evidence_frames": [],
        }

    frames_with_detections = sum(
        1 for item in matching_frames if _safe_int(item.get("detection_count"), 0) > 0
    )
    total_detections = sum(_safe_int(item.get("detection_count"), 0) for item in matching_frames)
    person_counts = [_safe_int(item.get("person_count"), 0) for item in matching_frames]
    vehicle_counts = [_safe_int(item.get("vehicle_count"), 0) for item in matching_frames]
    important_counts = [_safe_int(item.get("important_object_count"), 0) for item in matching_frames]
    object_scores = [_safe_float(item.get("object_importance_score"), 0.0) for item in matching_frames]

    class_counts: dict[str, int] = {}
    for item in matching_frames:
        for class_name, count in item.get("object_counts", {}).items():
            normalized_name = str(class_name)
            class_counts[normalized_name] = class_counts.get(normalized_name, 0) + _safe_int(count, 0)

    top_classes = [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    ]

    top_frames = [
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
            matching_frames,
            key=lambda frame: _safe_float(frame.get("object_importance_score"), 0.0),
            reverse=True,
        )[:3]
    ]

    return {
        "matching_yolo_frames_count": len(matching_frames),
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "person_max": max(person_counts) if person_counts else 0,
        "person_avg": _round6(sum(person_counts) / len(person_counts)) if person_counts else 0.0,
        "vehicle_max": max(vehicle_counts) if vehicle_counts else 0,
        "important_object_max": max(important_counts) if important_counts else 0,
        "object_score_max": max(object_scores) if object_scores else 0.0,
        "object_score_avg": _round6(sum(object_scores) / len(object_scores)) if object_scores else 0.0,
        "top_classes": top_classes,
        "top_yolo_evidence_frames": top_frames,
    }


def _build_ranking_reasons(
    motion_component: float,
    yolo_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if motion_component >= 0.7:
        reasons.append("high_motion")
    if _safe_int(yolo_summary.get("person_max"), 0) > 0:
        reasons.append("person_present")
    if _safe_int(yolo_summary.get("person_max"), 0) >= 2:
        reasons.append("multiple_people")
    if _safe_int(yolo_summary.get("important_object_max"), 0) > 0:
        reasons.append("important_object_present")
    if _safe_float(yolo_summary.get("object_score_max"), 0.0) >= 0.6:
        reasons.append("object_rich_frame")
    if _safe_int(yolo_summary.get("frames_with_detections"), 0) == 0:
        reasons.append("weak_yolo_evidence")
    if _safe_int(yolo_summary.get("frames_with_detections"), 0) == 0 and motion_component > 0:
        reasons.append("motion_only_candidate")
    return reasons


def rank_candidate_clips(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 13: rank candidate clips")

    candidate_clips = _load_required_json(run_dir / "05_candidate_clips.json")
    expanded_clips = _load_required_json(run_dir / "06_expanded_clips.json")
    yolo_scores = _load_required_json(run_dir / "11_yolo_object_scores.json")
    yolo_report = _load_optional_json(run_dir / "11_yolo_usefulness_report.json")
    video_info = _load_optional_json(run_dir / "01_video_info.json")

    if not isinstance(candidate_clips, list):
        raise ValueError("Expected a list in 05_candidate_clips.json")
    if not isinstance(expanded_clips, list):
        raise ValueError("Expected a list in 06_expanded_clips.json")
    if not isinstance(yolo_scores, list):
        raise ValueError("Expected a list in 11_yolo_object_scores.json")
    if not isinstance(video_info, dict):
        video_info = {}
    if not isinstance(yolo_report, dict):
        yolo_report = {}

    expanded_by_clip_id = {item.get("clip_id"): item for item in expanded_clips if isinstance(item, dict)}
    ranked_items: list[dict[str, Any]] = []
    clips_with_yolo_evidence = 0
    clips_with_people = 0
    clips_with_multiple_people = 0
    clips_with_important_objects = 0

    for clip in candidate_clips:
        clip_id = clip.get("clip_id")
        expanded_clip = expanded_by_clip_id.get(clip_id, {})
        merged_clip = {**clip, **expanded_clip}
        matching_frames = _match_yolo_frames(merged_clip, yolo_scores)
        yolo_summary = _build_yolo_summary(matching_frames)

        motion_component = _clamp01(
            _safe_float(
                merged_clip.get("clip_motion_score", merged_clip.get("max_motion_score_norm", merged_clip.get("clip_score", 0.0))),
                _safe_float(merged_clip.get("clip_score", 0.0)),
            )
        )
        person_component = _clamp01(_safe_int(yolo_summary.get("person_max"), 0) / 3.0)
        multi_person_component = 1.0 if _safe_int(yolo_summary.get("person_max"), 0) >= 2 else 0.0
        important_object_component = 1.0 if _safe_int(yolo_summary.get("important_object_max"), 0) > 0 else 0.0
        object_density_component = _clamp01(_safe_float(yolo_summary.get("object_score_max"), 0.0))
        detection_presence_component = 1.0 if _safe_int(yolo_summary.get("frames_with_detections"), 0) > 0 else 0.0

        ranked_clip_score = _round6(
            (0.35 * motion_component)
            + (0.20 * person_component)
            + (0.15 * multi_person_component)
            + (0.10 * important_object_component)
            + (0.15 * object_density_component)
            + (0.05 * detection_presence_component)
        )

        ranking_reasons = _build_ranking_reasons(motion_component, yolo_summary)

        ranked_item = {
            "clip_id": clip_id,
            "start_time": merged_clip.get("start_time"),
            "end_time": merged_clip.get("end_time"),
            "expanded_start_time": merged_clip.get("expanded_start_time"),
            "expanded_end_time": merged_clip.get("expanded_end_time"),
            "duration_seconds": merged_clip.get("duration_seconds"),
            "motion": {
                "clip_score": merged_clip.get("clip_score", 0.0),
                "clip_motion_score": merged_clip.get(
                    "clip_motion_score",
                    merged_clip.get("max_motion_score_norm", merged_clip.get("clip_score", 0.0)),
                ),
                "reason": merged_clip.get("reason", "unknown"),
            },
            "yolo": yolo_summary,
            "score_components": {
                "motion_component": _round6(motion_component),
                "person_component": _round6(person_component),
                "multi_person_component": _round6(multi_person_component),
                "important_object_component": _round6(important_object_component),
                "object_density_component": _round6(object_density_component),
                "detection_presence_component": _round6(detection_presence_component),
            },
            "ranked_clip_score": ranked_clip_score,
            "ranking_reasons": ranking_reasons,
            "recommended_for_vlm": True,
        }
        ranked_items.append(ranked_item)

        if _safe_int(yolo_summary.get("matching_yolo_frames_count"), 0) > 0:
            clips_with_yolo_evidence += 1
        if _safe_int(yolo_summary.get("person_max"), 0) > 0:
            clips_with_people += 1
        if _safe_int(yolo_summary.get("person_max"), 0) >= 2:
            clips_with_multiple_people += 1
        if _safe_int(yolo_summary.get("important_object_max"), 0) > 0:
            clips_with_important_objects += 1

    ranked_items.sort(
        key=lambda item: (
            -_safe_float(item.get("ranked_clip_score"), 0.0),
            -_safe_float(item.get("score_components", {}).get("motion_component"), 0.0),
            _safe_float(item.get("start_time"), 0.0),
        )
    )

    for index, item in enumerate(ranked_items, start=1):
        item["rank"] = index

    ranked_output_path = run_dir / "13_ranked_clips.json"
    ranked_output_path.write_text(json.dumps(ranked_items, indent=2), encoding="utf-8")

    top_ranked_clips = []
    for item in ranked_items[:10]:
        top_annotated_frame_path = None
        top_frames = item.get("yolo", {}).get("top_yolo_evidence_frames", [])
        if top_frames:
            top_annotated_frame_path = top_frames[0].get("annotated_frame_path")
        top_ranked_clips.append(
            {
                "rank": item.get("rank"),
                "clip_id": item.get("clip_id"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "ranked_clip_score": item.get("ranked_clip_score"),
                "clip_motion_score": item.get("motion", {}).get("clip_motion_score"),
                "yolo_person_max": item.get("yolo", {}).get("person_max"),
                "yolo_important_object_max": item.get("yolo", {}).get("important_object_max"),
                "ranking_reasons": item.get("ranking_reasons", []),
                "top_annotated_frame_path": top_annotated_frame_path,
            }
        )

    if clips_with_people > 0:
        ranking_summary = (
            "Candidate clips were re-ranked using motion and YOLO object evidence. "
            "Many selected clips contain people, which makes them useful candidates for VLM review."
        )
    else:
        ranking_summary = (
            "Motion produced candidate clips, but YOLO found limited person/object evidence in some clips. "
            "Top-K selection should prioritize clips with stronger object evidence."
        )

    report = {
        "video_name": video_info.get("video_name"),
        "total_candidate_clips": len(candidate_clips),
        "ranked_clips_count": len(ranked_items),
        "clips_with_yolo_evidence": clips_with_yolo_evidence,
        "clips_with_people": clips_with_people,
        "clips_with_multiple_people": clips_with_multiple_people,
        "clips_with_important_objects": clips_with_important_objects,
        "top_ranked_clips": top_ranked_clips,
        "ranking_summary": ranking_summary,
        "recommendation": "Use 14_selected_top_clips.json in the next step to send only the highest-ranked clips to Qwen.",
    }

    report_output_path = run_dir / "13_ranked_clips_report.json"
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total candidate clips: {len(candidate_clips)}")
    print(f"[tender-demo] Ranked clips count: {len(ranked_items)}")
    print(f"[tender-demo] Clips with YOLO evidence: {clips_with_yolo_evidence}")
    print(f"[tender-demo] Clips with people: {clips_with_people}")
    print(f"[tender-demo] Clips with important objects: {clips_with_important_objects}")
    print(f"[tender-demo] Top 5 ranked clips: {top_ranked_clips[:5]}")
    print(f"[tender-demo] Ranked clips output path: {ranked_output_path}")
    print(f"[tender-demo] Ranked clips report output path: {report_output_path}")

    return {
        "ranked_clips": ranked_items,
        "report": report,
        "ranked_output_path": str(ranked_output_path),
        "report_output_path": str(report_output_path),
    }
