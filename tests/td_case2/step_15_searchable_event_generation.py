from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from stage_checks import read_json, write_json
from step_09_search_result_packaging import write_json_any


STEP11_FILE = "11_full_scene_event_candidates.json"
STEP12_FILE = "12_selected_top_event_candidates.json"
STEP14_FILE = "14_vlm_event_reviews.json"
STEP14_SUMMARY_FILE = "14_final_video_summary.json"
OUTPUT_FILE = "15_searchable_events.json"
FLAT_FILE = "15_searchable_events_flat.json"
REPORT_FILE = "15_searchable_event_report.json"
CRITICAL_EVENT_TYPES = {
    "collision",
    "accident",
    "crash",
    "impact",
    "vehicle_impact",
    "pedestrian_impact",
    "rollover",
    "fire",
    "explosion",
    "vehicle_hitting_another_vehicle",
    "near_miss",
}
EVENT_TYPE_ALIASES = {
    "accident": "collision",
    "crash": "collision",
    "impact": "collision",
    "vehicle_impact": "collision",
    "pedestrian_impact": "collision",
    "vehicle_hitting_another_vehicle": "collision",
}


def _normalize_event_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_") or "other"
    return EVENT_TYPE_ALIASES.get(text, text)


def _title_for_event_type(event_type: str) -> str:
    return event_type.replace("_", " ").title()


def _priority_rank(event_type: str, risk_level: str) -> int:
    if event_type in CRITICAL_EVENT_TYPES:
        return 10
    if risk_level == "high":
        return 8
    if risk_level == "medium":
        return 6
    if risk_level == "low":
        return 4
    return 2


def _step11_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(run_dir / STEP11_FILE)
    return {
        str(item.get("candidate_event_id", "") or ""): item
        for item in list(payload.get("candidate_events", []))
        if isinstance(item, dict) and str(item.get("candidate_event_id", "") or "")
    }


def _step12_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(run_dir / STEP12_FILE)
    return {
        str(item.get("candidate_event_id", "") or ""): item
        for item in list(payload.get("selected_candidates", []))
        if isinstance(item, dict) and str(item.get("candidate_event_id", "") or "")
    }


def _review_records(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_payload = read_json(run_dir / STEP14_FILE)
    summary_payload = read_json(run_dir / STEP14_SUMMARY_FILE)
    return [item for item in list(review_payload.get("reviews", [])) if isinstance(item, dict)], summary_payload


def _flat_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "searchable_event_id": record.get("searchable_event_id"),
        "source_candidate_id": record.get("source_candidate_ids", [None])[0],
        "event_type": record.get("event_type"),
        "risk_level": record.get("risk_level"),
        "best_timestamp_text": record.get("best_timestamp_text"),
        "confidence": record.get("confidence"),
        "priority_rank": record.get("priority_rank"),
        "summary": record.get("summary"),
    }


def run_searchable_event_generation(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    step11_map = _step11_map(run_dir)
    step12_map = _step12_map(run_dir)
    reviews, final_summary = _review_records(run_dir)

    records: list[dict[str, Any]] = []
    for review in reviews:
        model_review = dict(review.get("model_review", {}))
        if model_review.get("event_visible") is not True:
            continue
        source_candidate_ids = [str(item or "") for item in list(review.get("source_candidate_ids", [])) if str(item or "")]
        if not source_candidate_ids:
            continue
        primary_candidate_id = source_candidate_ids[0]
        step11_candidate = dict(step11_map.get(primary_candidate_id, {}))
        step12_candidate = dict(step12_map.get(primary_candidate_id, {}))
        event_type = _normalize_event_type(model_review.get("event_type") or step11_candidate.get("event_type"))
        risk_level = str(model_review.get("risk_level", "low") or "low").strip().lower()
        confidence = float(model_review.get("confidence", step12_candidate.get("ranking_score", 0.0)) or 0.0)
        records.append(
            {
                "searchable_event_id": primary_candidate_id,
                "event_id": primary_candidate_id,
                "source_type": "scene_event_review",
                "source_candidate_ids": source_candidate_ids,
                "source_vlm_input_id": review.get("vlm_input_id"),
                "event_type": event_type,
                "title": _title_for_event_type(event_type),
                "summary": str(model_review.get("summary_caption", "") or "").strip() or str(review.get("best_timestamp_text", "") or "").strip(),
                "start_seconds": step11_candidate.get("context_start_seconds", review.get("context_start_seconds")),
                "end_seconds": step11_candidate.get("context_end_seconds", review.get("context_end_seconds")),
                "best_timestamp_seconds": step11_candidate.get("best_timestamp_seconds", review.get("best_timestamp_seconds")),
                "best_timestamp_text": review.get("best_timestamp_text") or step11_candidate.get("best_timestamp_text"),
                "confidence": round(confidence, 6),
                "risk_level": risk_level,
                "priority_rank": _priority_rank(event_type, risk_level),
                "critical_event": event_type in CRITICAL_EVENT_TYPES,
                "track_ids": list(step11_candidate.get("involved_track_ids", [])),
                "class_names": list(step11_candidate.get("involved_classes", [])),
                "representative_frame_path": step11_candidate.get("representative_frame", {}).get("image_path")
                or step12_candidate.get("representative_frame_path"),
                "review_decision": model_review.get("review_decision"),
                "review_payload": review,
            }
        )

    records.sort(
        key=lambda item: (
            float(item.get("best_timestamp_seconds", 0.0) or 0.0),
            -int(bool(item.get("critical_event"))),
            -float(item.get("confidence", 0.0) or 0.0),
        )
    )
    flat_records = [_flat_record(item) for item in records]
    event_type_counts = Counter(str(item.get("event_type", "") or "other") for item in records)
    risk_counts = Counter(str(item.get("risk_level", "") or "low") for item in records)

    output_payload = {
        "status": "success",
        "source_files": [STEP11_FILE, STEP12_FILE, STEP14_FILE, STEP14_SUMMARY_FILE],
        "summary": {
            "event_visible_reviews": len(records),
            "critical_event_count": int(sum(1 for item in records if item.get("critical_event"))),
            "ready_for_step16_evidence_video": len(records) > 0,
        },
        "records": records,
    }
    report_payload = {
        "status": "success",
        "event_visible_reviews": len(records),
        "critical_event_count": int(sum(1 for item in records if item.get("critical_event"))),
        "event_type_counts": dict(event_type_counts),
        "risk_counts": dict(risk_counts),
        "step14_overall_status": final_summary.get("overall_status"),
        "step14_headline": final_summary.get("headline"),
        "top_events": flat_records[:10],
        "recommendation": "Proceed to Step 16 evidence video generation."
        if records
        else "No visible reviewed events were available for Step 16 scene-event export.",
    }

    write_json(run_dir / OUTPUT_FILE, output_payload)
    write_json_any(run_dir / FLAT_FILE, flat_records)
    write_json(run_dir / REPORT_FILE, report_payload)
    return output_payload, report_payload, flat_records
