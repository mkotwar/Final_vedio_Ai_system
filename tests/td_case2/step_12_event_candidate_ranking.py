from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from stage_checks import read_json, write_json
from step_09_search_result_packaging import write_json_any


FILTERED_SOURCE_FILE = "11_5_vlm_filtered_event_candidates.json"
STEP11_SOURCE_FILE = "11_full_scene_event_candidates.json"
EVENT_TYPE_PRIORITY = {
    "possible_collision_or_near_miss": 1.00,
    "sudden_stop": 0.90,
    "vehicle_person_interaction": 0.85,
    "traffic_congestion_or_dense_vehicle_activity": 0.75,
    "unusual_motion_spike": 0.65,
    "object_density_spike": 0.55,
    "track_start_stop_activity": 0.45,
    "stationary_vehicle": 0.40,
}
CONFIDENCE_LABEL_SCORE = {"high": 1.0, "medium": 0.7, "low": 0.35}
SEVERITY_LABEL_SCORE = {"high": 1.0, "medium": 0.7, "low": 0.35}
STRONG_TRIGGERS = {
    "vehicle_close_interaction",
    "sudden_speed_change",
    "track_level_stop_signal",
    "bbox_overlap",
}
MEDIUM_TRIGGERS = {
    "motion_spike",
    "motion_pixels_high",
    "histogram_change_high",
    "object_density_high",
    "vehicle_density_high",
}
WEAK_TRIGGERS = {
    "multiple_active_tracks",
    "multiple_trigger_reasons",
    "track_start_stop",
}
TRAFFIC_SAFETY_TYPES = {
    "possible_collision_or_near_miss",
    "sudden_stop",
    "vehicle_person_interaction",
}
CRITICAL_VISIBLE_EVENT_TYPES = {
    "collision",
    "accident",
    "crash",
    "impact",
    "vehicle_impact",
    "pedestrian_impact",
    "fire",
    "explosion",
    "rollover",
    "vehicle_hitting_another_vehicle",
    "near_miss",
}
CRITICAL_REASON_TERMS = (
    "collision",
    "accident",
    "crash",
    "impact",
    "near miss",
    "rollover",
    "overturned",
    "fire",
    "explosion",
    "hit another vehicle",
    "person on the ground",
)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a float into the requested range."""

    return max(low, min(high, value))


def _event_type_priority(event_type: str) -> float:
    """Return priority weight for one event type."""

    return float(EVENT_TYPE_PRIORITY.get(str(event_type or ""), 0.20))


def _score_label(score: float) -> str:
    """Map ranking score to ranking label."""

    if score >= 0.80:
        return "excellent_candidate"
    if score >= 0.65:
        return "strong_candidate"
    if score >= 0.50:
        return "useful_candidate"
    if score >= 0.35:
        return "weak_candidate"
    return "reject_candidate"


def _vlm_priority(score: float) -> str | None:
    """Map ranking score to VLM priority."""

    if score >= 0.65:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.35:
        return "low"
    return None


def _normalize_label(value: Any) -> str:
    """Normalize a free-text label into a snake-like key."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _contains_critical_reason(candidate: dict[str, Any]) -> bool:
    """Return whether Step 11.5 reason text mentions a critical incident."""

    short_reason = str(dict(candidate.get("vlm_filter", {})).get("short_reason", "") or "").lower()
    return any(term in short_reason for term in CRITICAL_REASON_TERMS)


def _critical_event_flags(candidate: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether Step 11.5 marked this as a critical incident."""

    vlm_filter = dict(candidate.get("vlm_filter", {}))
    decision = str(vlm_filter.get("decision", "") or "").strip().lower()
    visible_event_type = _normalize_label(vlm_filter.get("visible_event_type"))
    if decision == "yes" and visible_event_type in CRITICAL_VISIBLE_EVENT_TYPES:
        return True, visible_event_type
    if decision == "yes" and _contains_critical_reason(candidate):
        return True, "critical_reason_text"
    return False, None


def _validate_candidate(candidate: dict[str, Any]) -> dict[str, bool]:
    """Return tolerant ranking validation flags for one candidate."""

    representative_frame = candidate.get("representative_frame", {})
    full_frame_paths = list(candidate.get("full_frame_paths", []))
    involved_track_ids = list(candidate.get("involved_track_ids", []))
    trigger_reasons = list(candidate.get("trigger_reasons", []))
    return {
        "has_candidate_event_id": bool(candidate.get("candidate_event_id")),
        "has_event_type": bool(candidate.get("event_type")),
        "has_best_timestamp_seconds": candidate.get("best_timestamp_seconds") is not None,
        "has_candidate_score": candidate.get("candidate_score") is not None,
        "has_representative_frame": bool(representative_frame.get("image_path")),
        "has_full_frame_paths": bool(full_frame_paths),
        "has_involved_tracks": bool(involved_track_ids),
        "has_trigger_reasons": bool(trigger_reasons),
        "needs_vlm_review_true": bool(candidate.get("needs_vlm_review")) is True,
        "final_event_truth_candidate_only": str(candidate.get("final_event_truth", "")) in {
            "unknown_candidate_only",
            "normal_context_or_uncertain_candidate",
        },
    }


def _trigger_quality(trigger_reasons: list[str]) -> tuple[float, int, int, int]:
    """Return trigger quality score and grouped counts."""

    strong_count = sum(1 for item in trigger_reasons if item in STRONG_TRIGGERS)
    medium_count = sum(1 for item in trigger_reasons if item in MEDIUM_TRIGGERS)
    weak_count = sum(1 for item in trigger_reasons if item in WEAK_TRIGGERS)
    score = min(0.30, strong_count * 0.08 + medium_count * 0.04 + weak_count * 0.02)
    return round(score, 6), strong_count, medium_count, weak_count


def _evidence_quality(candidate: dict[str, Any], validation: dict[str, bool]) -> float:
    """Return evidence quality score."""

    evidence = dict(candidate.get("scene_evidence", {}))
    involved_objects = list(candidate.get("involved_objects", []))
    score = 0.0
    if validation["has_representative_frame"]:
        score += 0.08
    if validation["has_full_frame_paths"]:
        score += 0.08
    if validation["has_involved_tracks"]:
        score += 0.05
    if int(evidence.get("close_pair_count", 0) or 0) > 0:
        score += 0.08
    if int(evidence.get("vehicle_count_max", 0) or 0) >= 2:
        score += 0.05
    if float(evidence.get("motion_score_max", 0.0) or 0.0) > 0:
        score += 0.04
    if any(item.get("search_record_id") or item.get("vehicle_color") not in {None, "", "unknown"} for item in involved_objects):
        score += 0.04
    return round(min(0.30, score), 6)


def _penalty_score(
    candidate: dict[str, Any],
    validation: dict[str, bool],
    *,
    require_full_frame_path: bool,
    include_low_confidence: bool,
    strong_trigger_count: int,
    medium_trigger_count: int,
) -> float:
    """Return accumulated penalty score."""

    penalties = 0.0
    if not validation["has_representative_frame"]:
        penalties += 0.20
    if require_full_frame_path and not validation["has_full_frame_paths"]:
        penalties += 0.20
    if not validation["has_involved_tracks"]:
        penalties += 0.10
    candidate_score = float(candidate.get("candidate_score", 0.0) or 0.0)
    if candidate_score < 0.35:
        penalties += 0.20
    if strong_trigger_count == 0 and medium_trigger_count <= 1:
        penalties += 0.10
    if len(list(candidate.get("involved_track_ids", []))) > 15:
        penalties += 0.10
    if float(candidate.get("context_duration_seconds", 0.0) or 0.0) > 12.0:
        penalties += 0.05
    if str(candidate.get("event_type", "")) == "track_start_stop_activity":
        penalties += 0.05
    if not include_low_confidence and str(candidate.get("confidence_label", "")) == "low":
        penalties += 0.15
    return round(penalties, 6)


def _ranking_fields(
    candidate: dict[str, Any],
    *,
    require_full_frame_path: bool,
    include_low_confidence: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Calculate ranking fields for one candidate."""

    validation = _validate_candidate(candidate)
    trigger_reasons = list(candidate.get("trigger_reasons", []))
    trigger_score, strong_count, medium_count, weak_count = _trigger_quality(trigger_reasons)
    evidence_score = _evidence_quality(candidate, validation)
    penalty_score = _penalty_score(
        candidate,
        validation,
        require_full_frame_path=require_full_frame_path,
        include_low_confidence=include_low_confidence,
        strong_trigger_count=strong_count,
        medium_trigger_count=medium_count,
    )
    event_type_priority = _event_type_priority(str(candidate.get("event_type", "")))
    confidence_score = float(CONFIDENCE_LABEL_SCORE.get(str(candidate.get("confidence_label", "")), 0.35))
    severity_score = float(SEVERITY_LABEL_SCORE.get(str(candidate.get("severity_label", "")), 0.35))
    candidate_score = float(candidate.get("candidate_score", 0.0) or 0.0)
    critical_event, critical_reason = _critical_event_flags(candidate)
    step11_5_bonus = 0.15 if critical_event else 0.0
    ranking_score = _clamp(
        candidate_score * 0.40
        + event_type_priority * 0.20
        + confidence_score * 0.10
        + severity_score * 0.10
        + trigger_score * 0.10
        + evidence_score * 0.10
        + step11_5_bonus
        - penalty_score,
        0.0,
        1.0,
    )
    ranking_label = _score_label(ranking_score)
    vlm_priority = _vlm_priority(ranking_score)
    ranking = {
        "ranking_score": round(ranking_score, 6),
        "ranking_score_percent": round(ranking_score * 100.0, 2),
        "ranking_label": ranking_label,
        "vlm_priority": vlm_priority,
        "event_type_priority": round(event_type_priority, 6),
        "confidence_score": round(confidence_score, 6),
        "severity_score": round(severity_score, 6),
        "trigger_score": trigger_score,
        "evidence_score": evidence_score,
        "penalty_score": penalty_score,
        "strong_trigger_count": strong_count,
        "medium_trigger_count": medium_count,
        "weak_trigger_count": weak_count,
        "step11_5_priority_bonus": round(step11_5_bonus, 6),
        "critical_event": critical_event,
        "critical_reason": critical_reason,
    }
    valid_for_ranking = all(
        [
            validation["has_candidate_event_id"],
            validation["has_event_type"],
            validation["has_best_timestamp_seconds"],
            validation["has_candidate_score"],
            validation["needs_vlm_review_true"],
            validation["final_event_truth_candidate_only"],
        ]
    )
    validation["valid_for_ranking"] = valid_for_ranking
    return ranking, validation


def _cluster_candidates(ranked_candidates: list[dict[str, Any]], min_temporal_gap_seconds: float) -> tuple[list[dict[str, Any]], int]:
    """Assign temporal clusters based on best timestamp."""

    sorted_candidates = sorted(ranked_candidates, key=lambda item: float(item.get("best_timestamp_seconds", 0.0) or 0.0))
    current_cluster = 0
    previous_timestamp = None
    for candidate in sorted_candidates:
        timestamp = float(candidate.get("best_timestamp_seconds", 0.0) or 0.0)
        if previous_timestamp is None or abs(timestamp - previous_timestamp) > min_temporal_gap_seconds:
            current_cluster += 1
        candidate.setdefault("selection", {})
        candidate["selection"]["temporal_cluster_id"] = f"evt_cluster_{current_cluster:04d}"
        previous_timestamp = timestamp
    return ranked_candidates, current_cluster


def _selection_reason(candidate: dict[str, Any], prefer_traffic_safety: bool) -> list[str]:
    """Generate simple selection reasons for a chosen candidate."""

    reasons: list[str] = []
    ranking = dict(candidate.get("ranking", {}))
    if float(ranking.get("ranking_score", 0.0) or 0.0) >= 0.65:
        reasons.append("high_ranking_score")
    if prefer_traffic_safety and str(candidate.get("event_type", "")) in TRAFFIC_SAFETY_TYPES:
        reasons.append("traffic_safety_event")
    if bool(candidate.get("full_frame_paths")):
        reasons.append("full_frame_available")
    reasons.append("temporal_diversity")
    return reasons


def _flat_selected_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Create flat selected-candidate output row."""

    return {
        "selection_rank": int(candidate.get("selection", {}).get("selection_rank", 0) or 0),
        "candidate_event_id": candidate.get("candidate_event_id"),
        "event_type": candidate.get("event_type"),
        "best_timestamp_text": candidate.get("best_timestamp_text"),
        "candidate_score": candidate.get("candidate_score"),
        "ranking_score": candidate.get("ranking", {}).get("ranking_score"),
        "ranking_label": candidate.get("ranking", {}).get("ranking_label"),
        "vlm_priority": candidate.get("ranking", {}).get("vlm_priority"),
        "confidence_label": candidate.get("confidence_label"),
        "severity_label": candidate.get("severity_label"),
        "trigger_reasons": ", ".join(list(candidate.get("trigger_reasons", []))),
        "involved_classes": ", ".join(list(candidate.get("involved_classes", []))),
        "representative_frame_path": candidate.get("representative_frame", {}).get("image_path"),
    }


def run_event_candidate_ranking(
    *,
    run_dir: Path,
    ranking_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Rank Step 11 event candidates and select Top-K for later VLM review."""

    source_path = run_dir / FILTERED_SOURCE_FILE
    if source_path.exists():
        filtered_payload = read_json(source_path)
        if filtered_payload.get("status") == "success" and list(filtered_payload.get("candidate_events", [])):
            candidates_payload = filtered_payload
            source_file = FILTERED_SOURCE_FILE
        else:
            candidates_payload = read_json(run_dir / STEP11_SOURCE_FILE)
            source_file = STEP11_SOURCE_FILE
    else:
        candidates_payload = read_json(run_dir / STEP11_SOURCE_FILE)
        source_file = STEP11_SOURCE_FILE
    source_candidates = list(candidates_payload.get("candidate_events", []))
    if not source_candidates:
        ranked_payload = {
            "status": "success",
            "source_file": source_file,
            "config": ranking_config,
            "summary": {
                "input_candidate_count": 0,
                "ranked_candidate_count": 0,
                "selected_top_k_count": 0,
                "rejected_candidate_count": 0,
                "temporal_cluster_count": 0,
                "ready_for_step13_vlm_input_generation": False,
            },
            "ranked_candidates": [],
        }
        selected_payload = {
            "status": "success",
            "source_file": source_file,
            "top_k": int(ranking_config["top_k"]),
            "selected_count": 0,
            "selected_candidates": [],
        }
        report_payload = {
            "status": "success",
            "source_file": source_file,
            "input_candidate_count": 0,
            "ranked_candidate_count": 0,
            "selected_top_k_count": 0,
            "temporal_cluster_count": 0,
            "selected_event_type_counts": {},
            "selected_confidence_counts": {},
            "selected_severity_counts": {},
            "ranking_label_counts": {},
            "vlm_priority_counts": {},
            "top_selected_candidates": [],
            "suppression_summary": {
                "candidates_suppressed_by_temporal_cluster": 0,
                "candidates_suppressed_by_event_type_cap": 0,
                "candidates_below_min_ranking_score": 0,
            },
            "warnings": [],
            "recommendation": "Lower threshold or inspect Step 11 candidate generation.",
        }
        write_json(run_dir / "12_ranked_event_candidates.json", ranked_payload)
        write_json(run_dir / "12_selected_top_event_candidates.json", selected_payload)
        write_json(run_dir / "12_event_candidate_ranking_report.json", report_payload)
        write_json_any(run_dir / "12_selected_event_candidates_flat.json", [])
        return ranked_payload, selected_payload, report_payload, []

    ranked_candidates: list[dict[str, Any]] = []
    for candidate in source_candidates:
        ranked_candidate = dict(candidate)
        ranking, validation = _ranking_fields(
            ranked_candidate,
            require_full_frame_path=bool(ranking_config["require_full_frame_path"]),
            include_low_confidence=bool(ranking_config["include_low_confidence"]),
        )
        ranked_candidate["validation"] = validation
        ranked_candidate["ranking"] = ranking
        ranked_candidate.setdefault("selection", {})
        ranked_candidate["selection"].update(
            {
                "selected_for_vlm": False,
                "selection_rank": None,
                "selection_reason": [],
                "temporal_cluster_id": None,
            }
        )
        ranked_candidates.append(ranked_candidate)

    ranked_candidates.sort(
        key=lambda item: (
            -int(bool(item["ranking"].get("critical_event"))),
            -float(item["ranking"]["ranking_score"]),
            -float(item.get("candidate_score", 0.0) or 0.0),
            float(item.get("best_timestamp_seconds", 0.0) or 0.0),
        )
    )
    ranked_candidates, temporal_cluster_count = _cluster_candidates(
        ranked_candidates,
        float(ranking_config["min_temporal_gap_seconds"]),
    )

    selected_candidates: list[dict[str, Any]] = []
    selected_by_cluster: Counter[str] = Counter()
    selected_by_type: Counter[str] = Counter()
    suppressed_by_cluster = 0
    suppressed_by_type = 0
    suppressed_by_score = 0
    top_k = int(ranking_config["top_k"])
    min_ranking_score = float(ranking_config["min_ranking_score"])
    critical_candidates = [
        candidate
        for candidate in ranked_candidates
        if bool(candidate.get("validation", {}).get("valid_for_ranking"))
        and bool(candidate.get("ranking", {}).get("critical_event"))
    ]
    selection_limit = max(top_k, len(critical_candidates))

    def _append_selected(candidate: dict[str, Any], *, forced_reason: str | None = None) -> None:
        cluster_id = str(candidate.get("selection", {}).get("temporal_cluster_id", "") or "")
        event_type = str(candidate.get("event_type", "") or "unknown")
        candidate["selection"]["selected_for_vlm"] = True
        candidate["selection"]["selection_rank"] = len(selected_candidates) + 1
        selection_reason = _selection_reason(
            candidate,
            bool(ranking_config["prefer_traffic_safety"]),
        )
        if forced_reason and forced_reason not in selection_reason:
            selection_reason.insert(0, forced_reason)
        candidate["selection"]["selection_reason"] = selection_reason
        selected_candidates.append(candidate)
        selected_by_cluster[cluster_id] += 1
        selected_by_type[event_type] += 1

    for candidate in critical_candidates:
        if bool(ranking_config["require_full_frame_path"]) and not bool(candidate.get("validation", {}).get("has_full_frame_paths")):
            suppressed_by_score += 1
            continue
        if not bool(ranking_config["include_low_confidence"]) and str(candidate.get("confidence_label", "")) == "low":
            suppressed_by_score += 1
            continue
        _append_selected(candidate, forced_reason="critical_accident_priority")

    selected_ids = {str(candidate.get("candidate_event_id", "") or "") for candidate in selected_candidates}
    for candidate in ranked_candidates:
        ranking = dict(candidate.get("ranking", {}))
        selection = dict(candidate.get("selection", {}))
        cluster_id = str(selection.get("temporal_cluster_id", "") or "")
        event_type = str(candidate.get("event_type", "") or "unknown")
        candidate_id = str(candidate.get("candidate_event_id", "") or "")
        if candidate_id in selected_ids:
            continue
        if not bool(candidate.get("validation", {}).get("valid_for_ranking")):
            suppressed_by_score += 1
            continue
        if bool(ranking_config["require_full_frame_path"]) and not bool(candidate.get("validation", {}).get("has_full_frame_paths")):
            suppressed_by_score += 1
            continue
        if not bool(ranking_config["include_low_confidence"]) and str(candidate.get("confidence_label", "")) == "low":
            suppressed_by_score += 1
            continue
        if float(ranking.get("ranking_score", 0.0) or 0.0) < min_ranking_score:
            suppressed_by_score += 1
            continue
        if selected_by_cluster[cluster_id] >= int(ranking_config["max_per_time_cluster"]):
            suppressed_by_cluster += 1
            continue
        if selected_by_type[event_type] >= int(ranking_config["max_per_event_type"]):
            suppressed_by_type += 1
            continue
        _append_selected(candidate)
        selected_ids.add(candidate_id)
        if len(selected_candidates) >= selection_limit:
            break

    selected_event_type_counts = Counter(candidate["event_type"] for candidate in selected_candidates)
    selected_confidence_counts = Counter(candidate["confidence_label"] for candidate in selected_candidates)
    selected_severity_counts = Counter(candidate["severity_label"] for candidate in selected_candidates)
    ranking_label_counts = Counter(candidate["ranking"]["ranking_label"] for candidate in ranked_candidates)
    vlm_priority_counts = Counter(
        candidate["ranking"]["vlm_priority"] for candidate in selected_candidates if candidate["ranking"]["vlm_priority"] is not None
    )

    ranked_payload = {
        "status": "success",
        "source_file": source_file,
        "config": ranking_config,
        "summary": {
            "input_candidate_count": len(source_candidates),
            "ranked_candidate_count": len(ranked_candidates),
            "selected_top_k_count": len(selected_candidates),
            "rejected_candidate_count": max(0, len(ranked_candidates) - len(selected_candidates)),
            "temporal_cluster_count": temporal_cluster_count,
            "critical_event_count": len(critical_candidates),
            "ready_for_step13_vlm_input_generation": len(selected_candidates) > 0,
        },
        "ranked_candidates": ranked_candidates,
    }
    selected_payload = {
        "status": "success",
        "source_file": source_file,
        "top_k": top_k,
        "selected_count": len(selected_candidates),
        "selected_candidates": [
            {
                "selection_rank": candidate["selection"]["selection_rank"],
                "candidate_event_id": candidate["candidate_event_id"],
                "event_type": candidate["event_type"],
                "best_timestamp_text": candidate["best_timestamp_text"],
                "context_start_seconds": candidate["context_start_seconds"],
                "context_end_seconds": candidate["context_end_seconds"],
                "ranking_score": candidate["ranking"]["ranking_score"],
                "vlm_priority": candidate["ranking"]["vlm_priority"],
                "representative_frame_path": candidate.get("representative_frame", {}).get("image_path"),
                "full_frame_paths": list(candidate.get("full_frame_paths", [])),
                "needs_vlm_review": candidate.get("needs_vlm_review"),
                "selected_for_vlm": candidate["selection"]["selected_for_vlm"],
            }
            for candidate in selected_candidates
        ],
    }
    flat_selected = [_flat_selected_candidate(candidate) for candidate in selected_candidates]
    recommendation = "Proceed to Step 13 VLM Input Generation for selected event candidates."
    if not selected_candidates:
        recommendation = "Lower threshold or inspect Step 11 candidate generation."
    report_payload = {
        "status": "success",
        "source_file": source_file,
        "input_candidate_count": len(source_candidates),
        "ranked_candidate_count": len(ranked_candidates),
        "selected_top_k_count": len(selected_candidates),
        "temporal_cluster_count": temporal_cluster_count,
        "critical_event_count": len(critical_candidates),
        "selected_event_type_counts": dict(selected_event_type_counts),
        "selected_confidence_counts": dict(selected_confidence_counts),
        "selected_severity_counts": dict(selected_severity_counts),
        "ranking_label_counts": dict(ranking_label_counts),
        "vlm_priority_counts": dict(vlm_priority_counts),
        "top_selected_candidates": [
            {
                "selection_rank": candidate["selection"]["selection_rank"],
                "candidate_event_id": candidate["candidate_event_id"],
                "event_type": candidate["event_type"],
                "best_timestamp_text": candidate["best_timestamp_text"],
                "ranking_score": candidate["ranking"]["ranking_score"],
                "vlm_priority": candidate["ranking"]["vlm_priority"],
                "selection_reason": candidate["selection"]["selection_reason"],
            }
            for candidate in selected_candidates
        ],
        "suppression_summary": {
            "candidates_suppressed_by_temporal_cluster": suppressed_by_cluster,
            "candidates_suppressed_by_event_type_cap": suppressed_by_type,
            "candidates_below_min_ranking_score": suppressed_by_score,
        },
        "warnings": [],
        "recommendation": recommendation,
    }

    write_json(run_dir / "12_ranked_event_candidates.json", ranked_payload)
    write_json(run_dir / "12_selected_top_event_candidates.json", selected_payload)
    write_json(run_dir / "12_event_candidate_ranking_report.json", report_payload)
    if bool(ranking_config["save_flat"]):
        write_json_any(run_dir / "12_selected_event_candidates_flat.json", flat_selected)
    else:
        write_json_any(run_dir / "12_selected_event_candidates_flat.json", [])
    return ranked_payload, selected_payload, report_payload, flat_selected
