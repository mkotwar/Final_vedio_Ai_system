from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CURVED_PATH_MAX_GAP_SECONDS = 2.0
CURVED_PATH_MAX_PREDICTION_DISTANCE = 0.06
CURVED_PATH_MIN_APPEARANCE_SIMILARITY = 0.80
CURVED_PATH_DIRECTION_MISMATCH_THRESHOLD = -0.2


def _bbox_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center_distance(a: list[float], b: list[float], diagonal: float) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diagonal if diagonal > 0 else 0.0


def _direction_vector(track: dict[str, Any]) -> tuple[float, float] | None:
    detections = list(track.get("detections", []))
    if len(detections) < 2:
        return None
    first_center = _bbox_center(detections[0]["bbox_xyxy"])
    last_center = _bbox_center(detections[-1]["bbox_xyxy"])
    dx = last_center[0] - first_center[0]
    dy = last_center[1] - first_center[1]
    magnitude = (dx * dx + dy * dy) ** 0.5
    if magnitude < 8.0:
        return None
    return (dx / magnitude, dy / magnitude)


def _direction_similarity(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    vector_a = _direction_vector(a)
    vector_b = _direction_vector(b)
    if vector_a is None or vector_b is None:
        return None
    return max(-1.0, min(1.0, (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1])))


def _predict_distance(source: dict[str, Any], target: dict[str, Any], diagonal: float) -> float:
    source_detections = list(source["detections"])
    target_detection = list(target["detections"])[0]
    if len(source_detections) >= 2:
        prev = source_detections[-2]
        last = source_detections[-1]
        dt = max(1e-6, float(last["timestamp_seconds"]) - float(prev["timestamp_seconds"]))
        pcx, pcy = _bbox_center(prev["bbox_xyxy"])
        lcx, lcy = _bbox_center(last["bbox_xyxy"])
        vx = (lcx - pcx) / dt
        vy = (lcy - pcy) / dt
        gap = max(0.0, float(target_detection["timestamp_seconds"]) - float(last["timestamp_seconds"]))
        predicted = (lcx + (vx * gap), lcy + (vy * gap))
    else:
        predicted = _bbox_center(source_detections[-1]["bbox_xyxy"])
    tcx, tcy = _bbox_center(target_detection["bbox_xyxy"])
    return (((predicted[0] - tcx) ** 2 + (predicted[1] - tcy) ** 2) ** 0.5) / diagonal if diagonal > 0 else 0.0


def _bbox_area_ratio(source: dict[str, Any], target: dict[str, Any]) -> float:
    area_a = _bbox_area(source["detections"][-1]["bbox_xyxy"])
    area_b = _bbox_area(target["detections"][0]["bbox_xyxy"])
    if area_a <= 0.0 or area_b <= 0.0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)


def _resolve_crop(run_dir: Path, path_value: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (run_dir / path).resolve()


def _appearance_similarity(run_dir: Path, source: dict[str, Any], target: dict[str, Any]) -> float | None:
    comparisons: list[float] = []
    for detection_a in list(source["detections"])[-3:]:
        path_a = _resolve_crop(run_dir, str(detection_a.get("crop_path", "")))
        if path_a is None or not path_a.exists():
            continue
        image_a = cv2.imread(str(path_a))
        if image_a is None:
            continue
        image_a = cv2.resize(image_a, (96, 96), interpolation=cv2.INTER_AREA)
        hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        for detection_b in list(target["detections"])[:3]:
            path_b = _resolve_crop(run_dir, str(detection_b.get("crop_path", "")))
            if path_b is None or not path_b.exists():
                continue
            image_b = cv2.imread(str(path_b))
            if image_b is None:
                continue
            image_b = cv2.resize(image_b, (96, 96), interpolation=cv2.INTER_AREA)
            hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
            hist_b = cv2.calcHist([hsv_b], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(hist_b, hist_b)
            correlation = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
            comparisons.append(max(0.0, min(1.0, (correlation + 1.0) / 2.0)))
    if not comparisons:
        return None
    return round(sum(comparisons) / len(comparisons), 6)


def _direction_penalty(direction_similarity: float | None) -> float:
    if direction_similarity is None:
        return 0.0
    if direction_similarity >= 0.0:
        return 0.0
    return round(min(0.35, abs(direction_similarity) * 0.15), 6)


def _base_merge_score(
    *,
    prediction_distance: float,
    area_ratio: float,
    max_area_ratio: float,
    appearance_similarity: float | None,
    direction_similarity: float | None,
) -> float:
    score = max(0.0, 1.0 - prediction_distance)
    score += max(0.0, 1.0 - min(area_ratio / max_area_ratio, 1.0))
    if appearance_similarity is not None:
        score += appearance_similarity
    if direction_similarity is not None:
        score += max(0.0, direction_similarity)
    return round(score / 4.0, 6)


def merge_track_fragments(
    *,
    run_dir: Path,
    raw_tracks: list[dict[str, Any]],
    image_diagonal: float,
    max_gap_seconds: float = 2.0,
    max_prediction_distance: float = 0.18,
    max_area_ratio: float = 4.0,
    min_appearance_similarity: float = 0.60,
    min_score_margin: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Conservatively merge raw track fragments when all safety checks pass."""

    working_tracks = [dict(track, detections=[dict(item) for item in track["detections"]]) for track in raw_tracks]
    merge_audit: list[dict[str, Any]] = []
    merge_count = 0

    while True:
        proposals_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        proposals_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source in working_tracks:
            source_end = float(source["detections"][-1]["timestamp_seconds"])
            for target in working_tracks:
                if source["track_id"] == target["track_id"]:
                    continue
                if source["track_type"] != target["track_type"]:
                    continue
                target_start = float(target["detections"][0]["timestamp_seconds"])
                gap = target_start - source_end
                prediction_distance = _predict_distance(source, target, image_diagonal)
                direction_similarity = _direction_similarity(source, target)
                area_ratio = _bbox_area_ratio(source, target)
                appearance_similarity = _appearance_similarity(run_dir, source, target)
                hard_rejection_reasons: list[str] = []
                direction_mismatch = bool(
                    direction_similarity is not None
                    and direction_similarity < CURVED_PATH_DIRECTION_MISMATCH_THRESHOLD
                )
                if gap < 0.0:
                    hard_rejection_reasons.append("timestamp_overlap")
                if gap > max_gap_seconds:
                    hard_rejection_reasons.append("time_gap_too_large")
                if prediction_distance > max_prediction_distance:
                    hard_rejection_reasons.append("predicted_position_mismatch")
                if area_ratio > max_area_ratio:
                    hard_rejection_reasons.append("bbox_area_ratio_too_large")
                if appearance_similarity is not None and appearance_similarity < min_appearance_similarity:
                    hard_rejection_reasons.append("appearance_similarity_too_low")

                base_score = _base_merge_score(
                    prediction_distance=prediction_distance,
                    area_ratio=area_ratio,
                    max_area_ratio=max_area_ratio,
                    appearance_similarity=appearance_similarity,
                    direction_similarity=direction_similarity,
                )
                penalty = _direction_penalty(direction_similarity)
                final_score = round(max(0.0, base_score - penalty), 6)
                curved_exception_checks = {
                    "same_compatible_class_group": source["track_type"] == target["track_type"],
                    "timestamps_do_not_overlap": gap >= 0.0,
                    "time_gap_within_2s": gap <= CURVED_PATH_MAX_GAP_SECONDS,
                    "predicted_position_distance_within_0_06": prediction_distance <= CURVED_PATH_MAX_PREDICTION_DISTANCE,
                    "appearance_similarity_at_least_0_80": appearance_similarity is not None and appearance_similarity >= CURVED_PATH_MIN_APPEARANCE_SIMILARITY,
                    "bbox_size_change_plausible": area_ratio <= max_area_ratio,
                    "no_competing_successor": False,
                    "mutual_best_match": False,
                }
                curved_exception_preeligible = direction_mismatch and all(
                    curved_exception_checks[key]
                    for key in (
                        "same_compatible_class_group",
                        "timestamps_do_not_overlap",
                        "time_gap_within_2s",
                        "predicted_position_distance_within_0_06",
                        "appearance_similarity_at_least_0_80",
                        "bbox_size_change_plausible",
                    )
                )
                accepted = not hard_rejection_reasons
                decision_reasons = list(hard_rejection_reasons)
                if direction_mismatch and not curved_exception_preeligible:
                    accepted = False
                    decision_reasons.append("direction_mismatch")
                proposal = {
                    "source_track_id": source["track_id"],
                    "candidate_track_id": target["track_id"],
                    "accepted": accepted,
                    "time_gap_seconds": round(gap, 6),
                    "predicted_position_distance": round(prediction_distance, 6),
                    "direction_mismatch": direction_mismatch,
                    "direction_similarity": None if direction_similarity is None else round(direction_similarity, 6),
                    "bbox_area_ratio": None if area_ratio == float("inf") else round(area_ratio, 6),
                    "appearance_similarity": appearance_similarity,
                    "curved_path_or_abrupt_turn_exception_eligible": curved_exception_preeligible,
                    "curved_path_or_abrupt_turn_exception_checks": curved_exception_checks,
                    "competing_candidate_count": 0,
                    "score_before_direction_penalty": base_score,
                    "direction_penalty": penalty,
                    "final_merge_score": final_score,
                    "final_merge_reason": "candidate_pending_competition_checks" if accepted else "rejected_initial_checks",
                    "decision_reasons": decision_reasons or ["candidate_passed_initial_checks"],
                }
                merge_audit.append(proposal)
                if accepted:
                    proposals_by_source[source["track_id"]].append(proposal)
                    proposals_by_target[target["track_id"]].append(proposal)

        accepted_pairs: list[tuple[str, str]] = []
        for source_id, proposals in proposals_by_source.items():
            ranked = sorted(proposals, key=lambda item: item["final_merge_score"], reverse=True)
            best = ranked[0]
            for proposal in ranked:
                proposal["competing_candidate_count"] = max(0, len(ranked) - 1)
                proposal["curved_path_or_abrupt_turn_exception_checks"]["no_competing_successor"] = len(ranked) == 1
            if len(ranked) > 1 and best["final_merge_score"] - ranked[1]["final_merge_score"] < min_score_margin:
                best["accepted"] = False
                best["final_merge_reason"] = "rejected_competing_successor"
                best["decision_reasons"] = ["insufficient_score_margin"]
                continue
            reverse_ranked = sorted(proposals_by_target[best["candidate_track_id"]], key=lambda item: item["final_merge_score"], reverse=True)
            is_mutual_best = bool(reverse_ranked and reverse_ranked[0]["source_track_id"] == source_id)
            best["curved_path_or_abrupt_turn_exception_checks"]["mutual_best_match"] = is_mutual_best
            curved_exception_passed = bool(
                best["curved_path_or_abrupt_turn_exception_eligible"]
                and best["curved_path_or_abrupt_turn_exception_checks"]["no_competing_successor"]
                and is_mutual_best
            )
            if reverse_ranked and reverse_ranked[0]["source_track_id"] == source_id:
                if best["direction_mismatch"] and not curved_exception_passed:
                    best["accepted"] = False
                    best["final_merge_reason"] = "rejected_direction_without_exception"
                    best["decision_reasons"] = ["direction_mismatch_without_exception"]
                    continue
                best["accepted"] = True
                best["final_merge_reason"] = (
                    "accepted_curved_path_or_abrupt_turn_exception"
                    if curved_exception_passed
                    else "accepted_standard_merge"
                )
                if curved_exception_passed:
                    best["decision_reasons"] = ["curved_path_or_abrupt_turn_exception"]
                else:
                    best["decision_reasons"] = ["standard_merge_checks_passed"]
                accepted_pairs.append((source_id, best["candidate_track_id"]))
            else:
                best["accepted"] = False
                best["final_merge_reason"] = "rejected_not_mutual_best_match"
                best["decision_reasons"] = ["not_mutual_best_match"]

        if not accepted_pairs:
            break

        merged_ids: set[str] = set()
        next_tracks: list[dict[str, Any]] = []
        pair_lookup = {source_id: target_id for source_id, target_id in accepted_pairs}
        for track in working_tracks:
            track_id = track["track_id"]
            if track_id in merged_ids:
                continue
            if track_id in pair_lookup:
                target_id = pair_lookup[track_id]
                target_track = next(item for item in working_tracks if item["track_id"] == target_id)
                merged_ids.add(track_id)
                merged_ids.add(target_id)
                merged_track = {
                    "track_id": track_id,
                    "track_type": track["track_type"],
                    "source": "post_merge",
                    "merged_from_track_ids": [track_id, target_id],
                    "detections": sorted(
                        [*track["detections"], *target_track["detections"]],
                        key=lambda item: (float(item["timestamp_seconds"]), int(item["frame_idx"])),
                    ),
                }
                next_tracks.append(merged_track)
                merge_count += 1
            elif track_id not in merged_ids:
                next_tracks.append(track)
        working_tracks = next_tracks

    return working_tracks, merge_audit, {
        "post_merge_track_count": len(working_tracks),
        "post_merge_operations": merge_count,
    }
