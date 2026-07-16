from __future__ import annotations

import time
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


def _representative_crop_paths(run_dir: Path, track: dict[str, Any], max_crops: int = 3) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for detection in list(track.get("detections", [])):
        crop_path = _resolve_crop(run_dir, str(detection.get("crop_path", "")))
        if crop_path is None or not crop_path.exists():
            continue
        key = str(crop_path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(crop_path)
    if len(candidates) <= max_crops:
        return candidates
    midpoint = len(candidates) // 2
    selected_indexes = sorted({0, midpoint, len(candidates) - 1})
    return [candidates[index] for index in selected_indexes[:max_crops]]


def _compute_hsv_histogram(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram


def _get_or_build_histogram(
    crop_path: Path,
    *,
    crop_hist_cache: dict[str, np.ndarray | None],
    metrics: dict[str, Any],
) -> np.ndarray | None:
    cache_key = str(crop_path)
    if cache_key in crop_hist_cache:
        metrics["appearance_cache_hits"] += 1
        return crop_hist_cache[cache_key]
    image = cv2.imread(str(crop_path))
    metrics["crop_images_loaded"] += 1
    if image is None:
        crop_hist_cache[cache_key] = None
        return None
    histogram = _compute_hsv_histogram(image)
    crop_hist_cache[cache_key] = histogram
    return histogram


def _get_track_descriptors(
    run_dir: Path,
    track: dict[str, Any],
    *,
    descriptor_cache: dict[str, list[np.ndarray]],
    crop_hist_cache: dict[str, np.ndarray | None],
    metrics: dict[str, Any],
) -> list[np.ndarray]:
    track_id = str(track["track_id"])
    if track_id in descriptor_cache:
        metrics["appearance_cache_hits"] += 1
        return descriptor_cache[track_id]
    histograms: list[np.ndarray] = []
    for crop_path in _representative_crop_paths(run_dir, track):
        histogram = _get_or_build_histogram(crop_path, crop_hist_cache=crop_hist_cache, metrics=metrics)
        if histogram is not None:
            histograms.append(histogram)
    descriptor_cache[track_id] = histograms
    return histograms


def _appearance_similarity_cached(
    run_dir: Path,
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    descriptor_cache: dict[str, list[np.ndarray]],
    crop_hist_cache: dict[str, np.ndarray | None],
    metrics: dict[str, Any],
) -> float | None:
    source_descriptors = _get_track_descriptors(
        run_dir,
        source,
        descriptor_cache=descriptor_cache,
        crop_hist_cache=crop_hist_cache,
        metrics=metrics,
    )
    target_descriptors = _get_track_descriptors(
        run_dir,
        target,
        descriptor_cache=descriptor_cache,
        crop_hist_cache=crop_hist_cache,
        metrics=metrics,
    )
    if not source_descriptors or not target_descriptors:
        return None
    metrics["appearance_comparisons"] += 1
    comparisons: list[float] = []
    for hist_a in source_descriptors:
        for hist_b in target_descriptors:
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


def _geometric_candidate_score(
    *,
    prediction_distance: float,
    area_ratio: float,
    max_area_ratio: float,
    direction_similarity: float | None,
) -> float:
    score = max(0.0, 1.0 - prediction_distance)
    score += max(0.0, 1.0 - min(area_ratio / max_area_ratio, 1.0))
    if direction_similarity is not None:
        score += max(0.0, direction_similarity)
    return round(score / 3.0, 6)


def _track_sort_key(track: dict[str, Any]) -> tuple[float, float, str]:
    detections = list(track.get("detections", []))
    start = float(detections[0]["timestamp_seconds"]) if detections else 0.0
    end = float(detections[-1]["timestamp_seconds"]) if detections else 0.0
    return (start, end, str(track.get("track_id", "")))


def _log_progress(message: str) -> None:
    print(message, flush=True)


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
    max_candidates_per_track: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Conservatively merge raw track fragments when all safety checks pass."""

    started = time.perf_counter()
    working_tracks = [dict(track, detections=[dict(item) for item in track["detections"]]) for track in raw_tracks]
    merge_audit: list[dict[str, Any]] = []
    merge_count = 0
    descriptor_cache: dict[str, list[np.ndarray]] = {}
    crop_hist_cache: dict[str, np.ndarray | None] = {}
    metrics: dict[str, Any] = {
        "total_possible_pairs": 0,
        "geometric_pairs_considered": 0,
        "pairs_rejected_before_appearance": 0,
        "appearance_comparisons": 0,
        "crop_images_loaded": 0,
        "appearance_cache_hits": 0,
        "merge_time_seconds": 0.0,
    }

    while True:
        proposals_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        proposals_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        appearance_candidates: list[dict[str, Any]] = []
        grouped_tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for track in working_tracks:
            grouped_tracks[str(track["track_type"])].append(track)

        for track_type in ("vehicle", "person"):
            group = sorted(grouped_tracks.get(track_type, []), key=_track_sort_key)
            metrics["total_possible_pairs"] += max(0, (len(group) * (len(group) - 1)) // 2)
            for source_index, source in enumerate(group):
                source_candidates: list[dict[str, Any]] = []
                source_detections = list(source["detections"])
                source_start = float(source_detections[0]["timestamp_seconds"])
                source_end = float(source_detections[-1]["timestamp_seconds"])
                for target in group[source_index + 1:]:
                    target_detections = list(target["detections"])
                    target_start = float(target_detections[0]["timestamp_seconds"])
                    target_end = float(target_detections[-1]["timestamp_seconds"])
                    metrics["geometric_pairs_considered"] += 1
                    gap = target_start - source_end
                    prediction_distance = _predict_distance(source, target, image_diagonal)
                    direction_similarity = _direction_similarity(source, target)
                    area_ratio = _bbox_area_ratio(source, target)
                    hard_rejection_reasons: list[str] = []
                    if source["track_type"] != target["track_type"]:
                        hard_rejection_reasons.append("incompatible_class_group")
                    if target_start < source_start or target_end < source_end:
                        hard_rejection_reasons.append("invalid_successor_ordering")
                    if gap < 0.0:
                        hard_rejection_reasons.append("timestamp_overlap")
                    if gap > max_gap_seconds:
                        hard_rejection_reasons.append("time_gap_too_large")
                    if prediction_distance > max_prediction_distance:
                        hard_rejection_reasons.append("predicted_position_mismatch")
                    if area_ratio > max_area_ratio:
                        hard_rejection_reasons.append("bbox_area_ratio_too_large")

                    geometric_score = _geometric_candidate_score(
                        prediction_distance=prediction_distance,
                        area_ratio=area_ratio,
                        max_area_ratio=max_area_ratio,
                        direction_similarity=direction_similarity,
                    )
                    if hard_rejection_reasons:
                        metrics["pairs_rejected_before_appearance"] += 1
                        merge_audit.append(
                            {
                                "source_track_id": source["track_id"],
                                "candidate_track_id": target["track_id"],
                                "accepted": False,
                                "time_gap_seconds": round(gap, 6),
                                "predicted_position_distance": round(prediction_distance, 6),
                                "direction_mismatch": False,
                                "direction_similarity": None if direction_similarity is None else round(direction_similarity, 6),
                                "bbox_area_ratio": None if area_ratio == float("inf") else round(area_ratio, 6),
                                "appearance_similarity": None,
                                "geometric_candidate_score": geometric_score,
                                "curved_path_or_abrupt_turn_exception_eligible": False,
                                "curved_path_or_abrupt_turn_exception_checks": {
                                    "same_compatible_class_group": source["track_type"] == target["track_type"],
                                    "timestamps_do_not_overlap": gap >= 0.0,
                                    "time_gap_within_2s": gap <= CURVED_PATH_MAX_GAP_SECONDS,
                                    "predicted_position_distance_within_0_06": prediction_distance <= CURVED_PATH_MAX_PREDICTION_DISTANCE,
                                    "appearance_similarity_at_least_0_80": False,
                                    "bbox_size_change_plausible": area_ratio <= max_area_ratio,
                                    "no_competing_successor": False,
                                    "mutual_best_match": False,
                                },
                                "competing_candidate_count": 0,
                                "score_before_direction_penalty": geometric_score,
                                "direction_penalty": 0.0,
                                "final_merge_score": 0.0,
                                "final_merge_reason": "rejected_initial_checks",
                                "decision_reasons": hard_rejection_reasons,
                            }
                        )
                        continue

                    source_candidates.append(
                        {
                            "source": source,
                            "target": target,
                            "gap": gap,
                            "prediction_distance": prediction_distance,
                            "direction_similarity": direction_similarity,
                            "area_ratio": area_ratio,
                            "geometric_candidate_score": geometric_score,
                        }
                    )

                ranked_candidates = sorted(
                    source_candidates,
                    key=lambda item: (item["geometric_candidate_score"], -item["prediction_distance"]),
                    reverse=True,
                )
                for dropped in ranked_candidates[max_candidates_per_track:]:
                    metrics["pairs_rejected_before_appearance"] += 1
                    merge_audit.append(
                        {
                            "source_track_id": dropped["source"]["track_id"],
                            "candidate_track_id": dropped["target"]["track_id"],
                            "accepted": False,
                            "time_gap_seconds": round(dropped["gap"], 6),
                            "predicted_position_distance": round(dropped["prediction_distance"], 6),
                            "direction_mismatch": False,
                            "direction_similarity": None if dropped["direction_similarity"] is None else round(dropped["direction_similarity"], 6),
                            "bbox_area_ratio": None if dropped["area_ratio"] == float("inf") else round(dropped["area_ratio"], 6),
                            "appearance_similarity": None,
                            "geometric_candidate_score": dropped["geometric_candidate_score"],
                            "curved_path_or_abrupt_turn_exception_eligible": False,
                            "curved_path_or_abrupt_turn_exception_checks": {
                                "same_compatible_class_group": True,
                                "timestamps_do_not_overlap": True,
                                "time_gap_within_2s": dropped["gap"] <= CURVED_PATH_MAX_GAP_SECONDS,
                                "predicted_position_distance_within_0_06": dropped["prediction_distance"] <= CURVED_PATH_MAX_PREDICTION_DISTANCE,
                                "appearance_similarity_at_least_0_80": False,
                                "bbox_size_change_plausible": dropped["area_ratio"] <= max_area_ratio,
                                "no_competing_successor": False,
                                "mutual_best_match": False,
                            },
                            "competing_candidate_count": 0,
                            "score_before_direction_penalty": dropped["geometric_candidate_score"],
                            "direction_penalty": 0.0,
                            "final_merge_score": 0.0,
                            "final_merge_reason": "rejected_candidate_limit",
                            "decision_reasons": ["candidate_limit_exceeded"],
                        }
                    )
                appearance_candidates.extend(ranked_candidates[:max_candidates_per_track])

        _log_progress("Merge preparation complete")
        _log_progress(f"Geometric pairs considered: {metrics['geometric_pairs_considered']}")
        _log_progress(f"Pairs rejected before appearance: {metrics['pairs_rejected_before_appearance']}")
        _log_progress(f"Appearance comparisons required: {len(appearance_candidates)}")

        for index, candidate in enumerate(appearance_candidates, start=1):
            source = candidate["source"]
            target = candidate["target"]
            gap = float(candidate["gap"])
            prediction_distance = float(candidate["prediction_distance"])
            direction_similarity = candidate["direction_similarity"]
            area_ratio = float(candidate["area_ratio"])
            appearance_similarity = _appearance_similarity_cached(
                run_dir,
                source,
                target,
                descriptor_cache=descriptor_cache,
                crop_hist_cache=crop_hist_cache,
                metrics=metrics,
            )
            direction_mismatch = bool(
                direction_similarity is not None and direction_similarity < CURVED_PATH_DIRECTION_MISMATCH_THRESHOLD
            )
            hard_rejection_reasons: list[str] = []
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
                "geometric_candidate_score": candidate["geometric_candidate_score"],
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
            if index == len(appearance_candidates) or index % 25 == 0:
                _log_progress(f"Current progress {index}/{len(appearance_candidates)}")

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
            reverse_ranked = sorted(
                proposals_by_target[best["candidate_track_id"]],
                key=lambda item: item["final_merge_score"],
                reverse=True,
            )
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
                best["decision_reasons"] = (
                    ["curved_path_or_abrupt_turn_exception"]
                    if curved_exception_passed
                    else ["standard_merge_checks_passed"]
                )
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
        _log_progress(f"Accepted merges: {merge_count}")

    metrics["merge_time_seconds"] = round(time.perf_counter() - started, 3)
    _log_progress(f"Appearance cache hits: {metrics['appearance_cache_hits']}")
    _log_progress(f"Accepted merges: {merge_count}")
    return working_tracks, merge_audit, {
        "post_merge_track_count": len(working_tracks),
        "post_merge_operations": merge_count,
        **metrics,
    }
