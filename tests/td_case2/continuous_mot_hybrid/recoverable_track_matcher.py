from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised in the isolated td_case2 venv
    from ultralytics.utils.ops import linear_sum_assignment

from .recoverable_track_store import RecoverableTrackSnapshot
from .recovery_scoring import RecoveryScoringConfig, score_recovery_candidate


@dataclass
class RecoveryMatchResult:
    accepted_matches: list[dict[str, Any]]
    possible_matches: list[dict[str, Any]]
    rejected_matches: list[dict[str, Any]]
    all_attempts: list[dict[str, Any]]


def match_recoverable_tracks(
    *,
    unmatched_detections: list[dict[str, Any]],
    candidate_entries_by_tracker_id: dict[str, list[RecoverableTrackSnapshot]],
    timestamp_seconds: float,
    scoring_config: RecoveryScoringConfig,
) -> RecoveryMatchResult:
    scored_candidates: list[dict[str, Any]] = []
    for detection in unmatched_detections:
        for entry in candidate_entries_by_tracker_id.get(str(detection["tracker_id"]), []):
            scored_candidates.append(
                score_recovery_candidate(
                    unmatched_detection=detection,
                    entry=entry,
                    timestamp_seconds=timestamp_seconds,
                    detection_histogram=detection.get("histogram_descriptor"),
                    config=scoring_config,
                )
            )
    detection_ids = [str(item["tracker_id"]) for item in unmatched_detections]
    local_object_ids = sorted({int(item["proposed_local_object_id"]) for item in scored_candidates if not item["hard_rejection_reasons"]})
    accepted: list[dict[str, Any]] = []
    possible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if detection_ids and local_object_ids:
        cost_matrix = np.full((len(detection_ids), len(local_object_ids)), 1e6, dtype=np.float32)
        attempt_lookup: dict[tuple[str, int], dict[str, Any]] = {}
        for attempt in scored_candidates:
            key = (str(attempt["new_tracker_id"]), int(attempt["proposed_local_object_id"]))
            attempt_lookup[key] = attempt
            if attempt["hard_rejection_reasons"]:
                continue
            detection_index = detection_ids.index(str(attempt["new_tracker_id"]))
            local_index = local_object_ids.index(int(attempt["proposed_local_object_id"]))
            cost_matrix[detection_index, local_index] = -float(attempt["total_score"])
        row_indexes, col_indexes = linear_sum_assignment(cost_matrix)
        assigned_detection_ids: set[str] = set()
        assigned_local_ids: set[int] = set()
        for row_index, col_index in zip(row_indexes, col_indexes):
            if cost_matrix[row_index, col_index] >= 1e5:
                continue
            detection_id = detection_ids[row_index]
            local_object_id = local_object_ids[col_index]
            matching_attempts = [
                item for item in scored_candidates if str(item["new_tracker_id"]) == detection_id and not item["hard_rejection_reasons"]
            ]
            matching_attempts.sort(key=lambda item: float(item["total_score"]), reverse=True)
            best = next(item for item in matching_attempts if int(item["proposed_local_object_id"]) == local_object_id)
            second_best_score = float(matching_attempts[1]["total_score"]) if len(matching_attempts) > 1 else 0.0
            score_margin = float(best["total_score"]) - second_best_score
            decision = "rejected"
            if float(best["total_score"]) >= scoring_config.auto_reactivate_score and score_margin >= scoring_config.minimum_score_margin:
                decision = "accepted"
                assigned_detection_ids.add(detection_id)
                assigned_local_ids.add(local_object_id)
                accepted.append({**best, "second_best_score": round(second_best_score, 6), "score_margin": round(score_margin, 6), "final_decision": decision})
            elif float(best["total_score"]) >= scoring_config.possible_reactivate_score:
                decision = "possible"
                possible.append({**best, "second_best_score": round(second_best_score, 6), "score_margin": round(score_margin, 6), "final_decision": decision})
            else:
                rejected.append({**best, "second_best_score": round(second_best_score, 6), "score_margin": round(score_margin, 6), "final_decision": decision})
        for attempt in scored_candidates:
            if attempt["hard_rejection_reasons"]:
                rejected.append({**attempt, "second_best_score": 0.0, "score_margin": 0.0, "final_decision": "rejected"})
    else:
        for attempt in scored_candidates:
            rejected.append({**attempt, "second_best_score": 0.0, "score_margin": 0.0, "final_decision": "rejected"})
    return RecoveryMatchResult(
        accepted_matches=accepted,
        possible_matches=possible,
        rejected_matches=rejected,
        all_attempts=[*accepted, *possible, *rejected],
    )
