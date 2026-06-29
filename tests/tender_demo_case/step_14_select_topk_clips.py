from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MANDATORY_PRIORITY_REASONS = [
    "mandatory_suspicious_or_high_priority",
    "mandatory_summary_event",
]

MANDATORY_EVENT_KEYWORDS = {
    "theft",
    "robbery",
    "assault",
    "fight",
    "weapon",
    "fall",
    "collision",
    "intrusion",
    "emergency",
}


def _load_required_json(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required selection input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> list[dict[str, Any]] | dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer. Received: {raw_value!r}") from exc


def _contains_mandatory_event_keyword(text: Any) -> bool:
    normalized = str(text or "").lower()
    return any(keyword in normalized for keyword in MANDATORY_EVENT_KEYWORDS)


def _add_selection_reason(
    selected_by_clip_id: dict[str, dict[str, Any]],
    clip_item: dict[str, Any],
    reason: str,
) -> bool:
    clip_id = str(clip_item.get("clip_id", "")).strip()
    if not clip_id:
        return False

    existing = selected_by_clip_id.get(clip_id)
    if existing is None:
        selected_by_clip_id[clip_id] = {
            **clip_item,
            "selection_reasons": [reason],
            "source_ranked_clip": True,
        }
        return True

    if reason not in existing["selection_reasons"]:
        existing["selection_reasons"].append(reason)
    return False


def _is_mandatory_fused_clip(fused_clip: dict[str, Any]) -> bool:
    vlm = fused_clip.get("vlm", {})
    suspicious_activity = str(vlm.get("suspicious_activity", "")).lower()
    review_priority = str(fused_clip.get("review_priority", "")).lower()
    risk_level = str(vlm.get("risk_level", "")).lower()
    event_label = str(vlm.get("event_label", "")).lower()

    if suspicious_activity == "yes":
        return True
    if review_priority in {"critical", "high"}:
        return True
    if risk_level in {"high", "critical", "severe"}:
        return True
    if _contains_mandatory_event_keyword(event_label):
        return True
    return False


def _is_mandatory_summary_event(timeline_item: dict[str, Any]) -> bool:
    suspicious_activity = str(timeline_item.get("suspicious_activity", "")).lower()
    risk_level = str(timeline_item.get("risk_level", "")).lower()
    event_label = str(timeline_item.get("event_label", "")).lower()

    if suspicious_activity == "yes":
        return True
    if risk_level in {"high", "critical", "severe", "medium"}:
        return True
    if _contains_mandatory_event_keyword(event_label):
        return True
    return False


def _selection_sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
    reasons = item.get("selection_reasons", [])
    priority_bucket = 2
    if "mandatory_suspicious_or_high_priority" in reasons:
        priority_bucket = 0
    elif "mandatory_summary_event" in reasons:
        priority_bucket = 1
    return (
        priority_bucket,
        _safe_int(item.get("rank"), 999999),
        _safe_float(item.get("start_time"), 0.0),
    )


def select_topk_clips_for_qwen(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 14: select Top-K clips with guardrails")

    ranked_clips = _load_required_json(run_dir / "13_ranked_clips.json")
    summary_data = _load_optional_json(run_dir / "09_final_summary.json")
    fused_clips = _load_optional_json(run_dir / "12_fused_clip_evidence.json")
    _load_optional_json(run_dir / "12_fused_evidence_report.json")

    if not isinstance(ranked_clips, list):
        raise ValueError("Expected a list in 13_ranked_clips.json")
    if summary_data is not None and not isinstance(summary_data, dict):
        summary_data = None
    if fused_clips is not None and not isinstance(fused_clips, list):
        fused_clips = None

    top_k_requested = _read_env_int("TENDER_DEMO_TOP_K_CLIPS", 10)
    guardrails_enabled = _read_env_bool("TENDER_DEMO_ENABLE_SELECTION_GUARDRAILS", True)
    high_motion_guardrail_count = _read_env_int("TENDER_DEMO_HIGH_MOTION_GUARDRAIL_COUNT", 3)
    min_selected_clips = _read_env_int("TENDER_DEMO_MIN_SELECTED_CLIPS", 5)

    ranked_by_clip_id = {
        str(item.get("clip_id")): item for item in ranked_clips if str(item.get("clip_id", "")).strip()
    }
    selected_by_clip_id: dict[str, dict[str, Any]] = {}

    base_top_k = ranked_clips[: max(top_k_requested, 0)]
    for item in base_top_k:
        _add_selection_reason(selected_by_clip_id, item, "top_k_ranked")

    mandatory_suspicious_added = 0
    mandatory_summary_added = 0
    high_motion_added = 0
    minimum_fill_added = 0

    if guardrails_enabled and fused_clips:
        for fused_clip in fused_clips:
            if not _is_mandatory_fused_clip(fused_clip):
                continue
            clip_id = str(fused_clip.get("clip_id", "")).strip()
            ranked_item = ranked_by_clip_id.get(clip_id)
            if ranked_item is None:
                print(f"[tender-demo] Warning: fused mandatory clip_id not found in ranked clips: {clip_id}")
                continue
            if _add_selection_reason(
                selected_by_clip_id,
                ranked_item,
                "mandatory_suspicious_or_high_priority",
            ):
                mandatory_suspicious_added += 1

    event_timeline = summary_data.get("event_timeline", []) if summary_data else []
    if guardrails_enabled and isinstance(event_timeline, list):
        for timeline_item in event_timeline:
            if not _is_mandatory_summary_event(timeline_item):
                continue
            clip_id = str(timeline_item.get("clip_id", "")).strip()
            ranked_item = ranked_by_clip_id.get(clip_id)
            if ranked_item is None:
                print(f"[tender-demo] Warning: summary mandatory clip_id not found in ranked clips: {clip_id}")
                continue
            if _add_selection_reason(
                selected_by_clip_id,
                ranked_item,
                "mandatory_summary_event",
            ):
                mandatory_summary_added += 1

    if guardrails_enabled:
        high_motion_sorted = sorted(
            ranked_clips,
            key=lambda item: _safe_float(item.get("motion", {}).get("clip_motion_score"), 0.0),
            reverse=True,
        )
        for item in high_motion_sorted[: max(high_motion_guardrail_count, 0)]:
            if _add_selection_reason(selected_by_clip_id, item, "high_motion_guardrail"):
                high_motion_added += 1

    if len(selected_by_clip_id) < max(min_selected_clips, 0):
        for item in ranked_clips:
            if len(selected_by_clip_id) >= min_selected_clips:
                break
            if _add_selection_reason(selected_by_clip_id, item, "minimum_count_fill"):
                minimum_fill_added += 1

    selected_items = list(selected_by_clip_id.values())
    selected_items.sort(key=_selection_sort_key)

    for index, item in enumerate(selected_items, start=1):
        item["selection_order"] = index
        top_annotated_frame_path = None
        top_frames = item.get("yolo", {}).get("top_yolo_evidence_frames", [])
        if top_frames:
            top_annotated_frame_path = top_frames[0].get("annotated_frame_path")
        item["top_annotated_frame_path"] = top_annotated_frame_path

    selected_output_path = run_dir / "14_selected_top_clips.json"
    selected_output_path.write_text(json.dumps(selected_items, indent=2), encoding="utf-8")

    selected_clip_ids = [item.get("clip_id") for item in selected_items]
    if mandatory_suspicious_added or mandatory_summary_added:
        selection_summary = (
            "Selected Top-K ranked clips and added mandatory suspicious/high-priority clips to avoid missing important events."
        )
    else:
        selection_summary = (
            "Selected Top-K ranked clips. No additional suspicious/high-priority guardrail clips were found."
        )
    if high_motion_added:
        selection_summary += (
            " High-motion guardrail clips were also included to reduce the chance of missing action-heavy scenes before Qwen analysis."
        )

    report = {
        "top_k_requested": top_k_requested,
        "guardrails_enabled": guardrails_enabled,
        "total_ranked_clips": len(ranked_clips),
        "base_top_k_selected": len(base_top_k),
        "mandatory_suspicious_or_high_priority_added": mandatory_suspicious_added,
        "mandatory_summary_event_added": mandatory_summary_added,
        "high_motion_guardrail_added": high_motion_added,
        "minimum_fill_added": minimum_fill_added,
        "total_selected_clips": len(selected_items),
        "selected_clip_ids": selected_clip_ids,
        "selection_summary": selection_summary,
        "recommendation": "Use 15_topk_vlm_inputs.json in the next step to generate temporal strips only for selected clips.",
    }

    report_output_path = run_dir / "14_selected_top_clips_report.json"
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Top-K requested: {top_k_requested}")
    print(f"[tender-demo] Guardrails enabled: {guardrails_enabled}")
    print(f"[tender-demo] Total ranked clips: {len(ranked_clips)}")
    print(f"[tender-demo] Base Top-K selected: {len(base_top_k)}")
    print(f"[tender-demo] Mandatory suspicious/high priority added: {mandatory_suspicious_added}")
    print(f"[tender-demo] Summary event added: {mandatory_summary_added}")
    print(f"[tender-demo] High-motion guardrail added: {high_motion_added}")
    print(f"[tender-demo] Total selected clips: {len(selected_items)}")
    print(f"[tender-demo] Selected clip ids: {selected_clip_ids}")
    print(f"[tender-demo] Selected Top-K clips output path: {selected_output_path}")
    print(f"[tender-demo] Selected Top-K clips report output path: {report_output_path}")

    return {
        "selected_clips": selected_items,
        "report": report,
        "selected_output_path": str(selected_output_path),
        "report_output_path": str(report_output_path),
    }
