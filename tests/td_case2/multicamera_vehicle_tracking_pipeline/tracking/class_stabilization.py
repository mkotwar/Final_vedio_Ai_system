from __future__ import annotations

from dataclasses import asdict
from math import isclose

from .tracking_config import TrackingConfig
from .tracking_models import ClassObservation, LocalVehicleTrack, TrackClassDiagnostics


def normalize_track_class_name(value: str | None, config: TrackingConfig) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    alias_key = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    return config.class_aliases.get(alias_key, raw.strip().lower())


def build_class_diagnostics(
    *,
    history: list[ClassObservation],
    class_scores: dict[str, float],
    class_counts: dict[str, int],
    class_max_confidences: dict[str, float],
    config: TrackingConfig,
    previous_stable_class_name: str | None = None,
    previous_class_is_locked: bool = False,
) -> TrackClassDiagnostics:
    if config.is_standard_bytetrack:
        return _build_standard_class_diagnostics(
            history=history,
            class_scores=class_scores,
            class_counts=class_counts,
            class_max_confidences=class_max_confidences,
            config=config,
        )

    latest_class_name = history[-1].class_name if history else None
    total_observations = len(history)
    class_sequence = [item.class_name for item in history]
    ranked = sorted(
        (
            (
                class_name,
                int(class_counts.get(class_name, 0)),
                float(class_scores.get(class_name, 0.0)),
                float(class_max_confidences.get(class_name, 0.0)),
            )
            for class_name in class_counts
        ),
        key=lambda item: (item[1], item[2], item[3], item[0]),
        reverse=True,
    )

    score_winner = ranked[0][0] if ranked else latest_class_name
    provisional = score_winner
    winner_score = ranked[0][2] if ranked else 0.0
    runner_up_score = ranked[1][2] if len(ranked) > 1 else 0.0
    winner_margin = (
        (int(ranked[0][1]) / total_observations) - (int(ranked[1][1]) / total_observations)
        if len(ranked) > 1 and total_observations > 0
        else ((int(ranked[0][1]) / total_observations) if ranked and total_observations > 0 else None)
    )

    ranked_by_count = sorted(
        ((class_name, int(count), float(class_scores.get(class_name, 0.0))) for class_name, count in class_counts.items()),
        key=lambda item: (item[1], item[2], item[0]),
        reverse=True,
    )
    count_winner = ranked_by_count[0][0] if ranked_by_count else latest_class_name
    winning_class_count = int(class_counts.get(score_winner or "", 0)) if score_winner is not None else 0
    runner_up_class_name = ranked[1][0] if len(ranked) > 1 else None
    runner_up_class_count = int(class_counts.get(runner_up_class_name or "", 0)) if runner_up_class_name is not None else 0
    winning_class_ratio = (winning_class_count / total_observations) if total_observations > 0 else None
    runner_up_ratio = (runner_up_class_count / total_observations) if total_observations > 0 else None
    winner_count_margin = winning_class_count - runner_up_class_count
    winners_agree = (count_winner == score_winner) if (count_winner and score_winner) else True
    class_ratios = {
        class_name: (int(class_counts.get(class_name, 0)) / total_observations) if total_observations > 0 else 0.0
        for class_name in sorted(class_counts)
    }

    recent_window_size = min(total_observations, int(config.class_stabilization.recent_window_size))
    recent_history = history[-recent_window_size:] if recent_window_size > 0 else []
    recent_class_counts: dict[str, int] = {}
    for item in recent_history:
        recent_class_counts[item.class_name] = recent_class_counts.get(item.class_name, 0) + 1
    ranked_recent = sorted(
        recent_class_counts.items(),
        key=lambda item: (item[1], float(class_scores.get(item[0], 0.0)), item[0]),
        reverse=True,
    )
    recent_winning_class = ranked_recent[0][0] if ranked_recent else None
    recent_winning_count = ranked_recent[0][1] if ranked_recent else 0
    recent_winning_ratio = (recent_winning_count / len(recent_history)) if recent_history else None

    maximum_consecutive_winner_count = _max_consecutive_occurrences(class_sequence, provisional)
    recent_consecutive_winner_count = _trailing_consecutive_occurrences(class_sequence, provisional)
    class_transition_count, incompatible_class_transition_count = _transition_counts(history, config)

    stable = previous_stable_class_name
    locked = bool(previous_class_is_locked)
    class_confidence = winning_class_ratio if ranked else None

    meets_observation_threshold = total_observations >= config.class_stabilization.minimum_observations
    meets_consistency_ratio = (winning_class_ratio or 0.0) >= float(config.class_stabilization.minimum_consistency_ratio)
    meets_winner_margin = (winner_margin or 0.0) >= float(config.class_stabilization.minimum_winner_margin)
    meets_consecutive_threshold = maximum_consecutive_winner_count >= int(config.class_stabilization.minimum_consecutive_winner_observations)
    stable_candidate = provisional if (
        provisional is not None
        and meets_observation_threshold
        and meets_consistency_ratio
        and meets_winner_margin
        and meets_consecutive_threshold
    ) else None

    recent_conflict_minimum_observations = int(config.class_stabilization.recent_conflict_minimum_observations)
    recent_conflict_ratio = float(config.class_stabilization.recent_conflict_minimum_ratio)
    strong_conflict_detected = bool(
        previous_stable_class_name
        and previous_class_is_locked
        and recent_winning_class
        and recent_winning_class != previous_stable_class_name
        and len(recent_history) >= recent_conflict_minimum_observations
        and (recent_winning_ratio or 0.0) >= recent_conflict_ratio
        and _trailing_consecutive_occurrences(class_sequence, recent_winning_class)
        >= int(config.class_stabilization.strong_conflict_min_observations)
    )

    mixed_identity = _detect_mixed_identity(
        history=history,
        class_counts=class_counts,
        count_winner=count_winner,
        config=config,
    )

    if stable_candidate is not None:
        stable = stable_candidate
    elif not previous_class_is_locked:
        stable = None

    if mixed_identity["detected"]:
        stable = None
        locked = False

    if (
        not locked
        and stable is not None
        and total_observations >= config.class_stabilization.lock_after_observations
        and meets_winner_margin
        and meets_consistency_ratio
        and meets_consecutive_threshold
    ):
        locked = True

    if strong_conflict_detected and config.class_stabilization.allow_unlock_on_strong_conflict:
        locked = False

    if stable is None and not previous_class_is_locked:
        locked = False

    conflict_count = sum(count for name, count in class_counts.items() if stable is not None and name != stable)
    if total_observations < config.class_stabilization.minimum_observations:
        class_status = "INSUFFICIENT_OBSERVATIONS"
        final_class_reason = "INSUFFICIENT_OBSERVATIONS"
    elif mixed_identity["detected"]:
        class_status = "MIXED_IDENTITY"
        final_class_reason = "MIXED_IDENTITY"
    elif strong_conflict_detected:
        class_status = "STRONG_CONFLICT"
        final_class_reason = "STRONG_CONFLICT"
    elif stable is None:
        class_status = "AMBIGUOUS"
        final_class_reason = "NO_CLEAR_WINNER"
    elif locked:
        class_status = "LOCKED"
        final_class_reason = "COUNT_MAJORITY"
    else:
        class_status = "CONSISTENT"
        final_class_reason = "COUNT_MAJORITY"

    return TrackClassDiagnostics(
        provisional_class_name=provisional,
        stable_class_name=stable,
        class_status=class_status,
        final_class_reason=final_class_reason,
        class_is_locked=locked,
        class_confidence=class_confidence,
        class_winner_margin=winner_margin,
        class_observation_count=total_observations,
        class_conflict_count=conflict_count,
        winning_class_name=provisional,
        winning_class_count=winning_class_count,
        winning_class_ratio=winning_class_ratio,
        runner_up_class_name=runner_up_class_name,
        runner_up_class_count=runner_up_class_count,
        runner_up_ratio=runner_up_ratio,
        winner_count_margin=winner_count_margin,
        count_winner_class_name=count_winner,
        score_winner_class_name=score_winner,
        winners_agree=winners_agree,
        maximum_consecutive_winner_count=maximum_consecutive_winner_count,
        recent_consecutive_winner_count=recent_consecutive_winner_count,
        class_transition_count=class_transition_count,
        incompatible_class_transition_count=incompatible_class_transition_count,
        recent_class_counts=dict(sorted(recent_class_counts.items())),
        recent_winning_class_name=recent_winning_class,
        recent_winning_ratio=recent_winning_ratio,
        recent_observation_count=len(recent_history),
        strong_conflict_detected=strong_conflict_detected,
        split_recommended=strong_conflict_detected or bool(mixed_identity["detected"]),
        mixed_identity_detected=bool(mixed_identity["detected"]),
        mixed_identity_classes=tuple(mixed_identity["classes"]),
        mixed_identity_start_frame=mixed_identity["start_frame"],
        mixed_identity_confidence=mixed_identity["confidence"],
        final_class_blocked_due_to_mixed_identity=bool(mixed_identity["detected"]),
        class_scores={key: float(value) for key, value in sorted(class_scores.items())},
        class_ratios=class_ratios,
        class_observation_counts={key: int(value) for key, value in sorted(class_counts.items())},
        class_max_confidences={key: float(value) for key, value in sorted(class_max_confidences.items())},
        winner_confidence_sum=float(class_scores.get(score_winner or "", 0.0)) if score_winner is not None else None,
        raw_class_history=list(history),
        latest_observation_class_name=latest_class_name,
        possible_identity_switch=bool(mixed_identity["detected"]),
    )


def _build_standard_class_diagnostics(
    *,
    history: list[ClassObservation],
    class_scores: dict[str, float],
    class_counts: dict[str, int],
    class_max_confidences: dict[str, float],
    config: TrackingConfig,
) -> TrackClassDiagnostics:
    latest_class_name = history[-1].class_name if history else None
    total_observations = len(history)
    ranked = sorted(
        (
            (
                class_name,
                int(class_counts.get(class_name, 0)),
                float(class_scores.get(class_name, 0.0)),
                float(class_max_confidences.get(class_name, 0.0)),
            )
            for class_name in class_counts
        ),
        key=lambda item: (item[1], item[2], item[3], item[0]),
        reverse=True,
    )
    winning_class_name = ranked[0][0] if ranked else None
    winning_class_count = ranked[0][1] if ranked else 0
    winning_class_ratio = (winning_class_count / total_observations) if total_observations > 0 else None
    runner_up_class_name = ranked[1][0] if len(ranked) > 1 else None
    runner_up_class_count = ranked[1][1] if len(ranked) > 1 else 0
    runner_up_ratio = (runner_up_class_count / total_observations) if total_observations > 0 else None
    stable = None
    class_status = "UNKNOWN"
    final_class_reason = "NO_CLASS_OBSERVATIONS"
    if total_observations == 0:
        stable = None
    elif total_observations < int(config.track_class.minimum_observations):
        class_status = "INSUFFICIENT_OBSERVATIONS"
        final_class_reason = "INSUFFICIENT_OBSERVATIONS"
    elif (winning_class_ratio or 0.0) < float(config.track_class.minimum_winner_ratio):
        class_status = "AMBIGUOUS"
        final_class_reason = "NO_CLEAR_WINNER"
    else:
        stable = winning_class_name
        class_status = "CONSISTENT"
        final_class_reason = "COUNT_MAJORITY"
    class_ratios = {
        class_name: (int(class_counts.get(class_name, 0)) / total_observations) if total_observations > 0 else 0.0
        for class_name in sorted(class_counts)
    }
    count_winner = ranked[0][0] if ranked else latest_class_name
    possible_identity_switch = _detect_possible_identity_switch(history=history, config=config)
    class_transition_count, incompatible_class_transition_count = _transition_counts(history, config)
    sequence = [item.class_name for item in history]
    return TrackClassDiagnostics(
        provisional_class_name=winning_class_name,
        stable_class_name=stable,
        class_status=class_status,
        final_class_reason=final_class_reason,
        class_is_locked=False,
        class_confidence=winning_class_ratio,
        class_winner_margin=((winning_class_count - runner_up_class_count) / total_observations) if total_observations > 0 else None,
        class_observation_count=total_observations,
        class_conflict_count=sum(count for class_name, count in class_counts.items() if stable is not None and class_name != stable),
        winning_class_name=winning_class_name,
        winning_class_count=winning_class_count,
        winning_class_ratio=winning_class_ratio,
        runner_up_class_name=runner_up_class_name,
        runner_up_class_count=runner_up_class_count,
        runner_up_ratio=runner_up_ratio,
        winner_count_margin=winning_class_count - runner_up_class_count,
        count_winner_class_name=count_winner,
        score_winner_class_name=count_winner,
        winners_agree=True,
        maximum_consecutive_winner_count=_max_consecutive_occurrences(sequence, winning_class_name),
        recent_consecutive_winner_count=_trailing_consecutive_occurrences(sequence, winning_class_name),
        class_transition_count=class_transition_count,
        incompatible_class_transition_count=incompatible_class_transition_count,
        recent_class_counts={},
        recent_winning_class_name=None,
        recent_winning_ratio=None,
        recent_observation_count=0,
        strong_conflict_detected=False,
        split_recommended=False,
        mixed_identity_detected=False,
        mixed_identity_classes=(),
        mixed_identity_start_frame=None,
        mixed_identity_confidence=None,
        final_class_blocked_due_to_mixed_identity=False,
        class_scores={key: float(value) for key, value in sorted(class_scores.items())},
        class_ratios=class_ratios,
        class_observation_counts={key: int(value) for key, value in sorted(class_counts.items())},
        class_max_confidences={key: float(value) for key, value in sorted(class_max_confidences.items())},
        winner_confidence_sum=float(class_scores.get(winning_class_name or "", 0.0)) if winning_class_name is not None else None,
        raw_class_history=list(history),
        latest_observation_class_name=latest_class_name,
        possible_identity_switch=possible_identity_switch,
    )


def _max_consecutive_occurrences(class_sequence: list[str], target_class: str | None) -> int:
    if not target_class:
        return 0
    best = 0
    current = 0
    for item in class_sequence:
        if item == target_class:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _trailing_consecutive_occurrences(class_sequence: list[str], target_class: str | None) -> int:
    if not target_class:
        return 0
    count = 0
    for item in reversed(class_sequence):
        if item != target_class:
            break
        count += 1
    return count


def _class_compatibility(left: str | None, right: str | None, config: TrackingConfig) -> float:
    normalized_left = normalize_track_class_name(left, config)
    normalized_right = normalize_track_class_name(right, config)
    if normalized_left == normalized_right:
        return 1.0
    for family_members in config.class_families.values():
        if normalized_left in family_members and normalized_right in family_members:
            return 0.5
    return 0.0


def _transition_counts(history: list[ClassObservation], config: TrackingConfig) -> tuple[int, int]:
    transitions = 0
    incompatible_transitions = 0
    for previous, current in zip(history, history[1:]):
        if previous.class_name == current.class_name:
            continue
        transitions += 1
        if isclose(_class_compatibility(previous.class_name, current.class_name, config), 0.0):
            incompatible_transitions += 1
    return transitions, incompatible_transitions


def _detect_mixed_identity(
    *,
    history: list[ClassObservation],
    class_counts: dict[str, int],
    count_winner: str | None,
    config: TrackingConfig,
) -> dict[str, object]:
    if not history or not count_winner:
        return {"detected": False, "classes": (), "start_frame": None, "confidence": None}
    competing_classes = sorted(
        (
            class_name
            for class_name, count in class_counts.items()
            if class_name != count_winner and int(count) >= 2 and _class_compatibility(count_winner, class_name, config) == 0.0
        ),
        key=lambda class_name: (int(class_counts.get(class_name, 0)), class_name),
        reverse=True,
    )
    if not competing_classes:
        return {"detected": False, "classes": (), "start_frame": None, "confidence": None}
    competing_class = competing_classes[0]
    if int(class_counts.get(count_winner, 0)) < 2 or int(class_counts.get(competing_class, 0)) < 2:
        return {"detected": False, "classes": (), "start_frame": None, "confidence": None}

    segments = _build_class_segments(history)
    qualifying_pair: tuple[dict[str, object], dict[str, object]] | None = None
    for left_index, left in enumerate(segments):
        if left["class_name"] != count_winner or int(left["count"]) < 2:
            continue
        for right in segments[left_index + 1 :]:
            if right["class_name"] != competing_class or int(right["count"]) < 2:
                continue
            qualifying_pair = (left, right)
            break
        if qualifying_pair is not None:
            break
    if qualifying_pair is None:
        return {"detected": False, "classes": (), "start_frame": None, "confidence": None}

    mixed_classes = tuple(sorted({count_winner, competing_class}))
    combined_count = max(1, int(class_counts.get(count_winner, 0)) + int(class_counts.get(competing_class, 0)))
    return {
        "detected": True,
        "classes": mixed_classes,
        "start_frame": int(qualifying_pair[1]["start_frame"]),
        "confidence": combined_count / max(len(history), 1),
    }


def _build_class_segments(history: list[ClassObservation]) -> list[dict[str, object]]:
    if not history:
        return []
    segments: list[dict[str, object]] = []
    current_items: list[ClassObservation] = [history[0]]
    for item in history[1:]:
        if item.class_name == current_items[-1].class_name:
            current_items.append(item)
            continue
        segments.append(
            {
                "class_name": current_items[0].class_name,
                "start_frame": current_items[0].frame_number,
                "end_frame": current_items[-1].frame_number,
                "count": len(current_items),
                "average_confidence": sum(float(segment_item.confidence) for segment_item in current_items) / max(len(current_items), 1),
            }
        )
        current_items = [item]
    segments.append(
        {
            "class_name": current_items[0].class_name,
            "start_frame": current_items[0].frame_number,
            "end_frame": current_items[-1].frame_number,
            "count": len(current_items),
            "average_confidence": sum(float(segment_item.confidence) for segment_item in current_items) / max(len(current_items), 1),
        }
    )
    return segments


def _detect_possible_identity_switch(*, history: list[ClassObservation], config: TrackingConfig) -> bool:
    if len(history) < 4:
        return False
    segments = _build_class_segments(history)
    qualifying_segments = [segment for segment in segments if int(segment["count"]) >= 2]
    if len(qualifying_segments) < 2:
        return False
    for left_index, left in enumerate(qualifying_segments):
        for right in qualifying_segments[left_index + 1 :]:
            if _class_compatibility(str(left["class_name"]), str(right["class_name"]), config) > 0.0:
                continue
            left_bbox = history[_history_index_for_frame(history, int(left["end_frame"]))].bbox_xyxy
            right_bbox = history[_history_index_for_frame(history, int(right["start_frame"]))].bbox_xyxy
            center_jump = _bbox_center_distance(left_bbox, right_bbox)
            larger_size = max(_bbox_scale(left_bbox), _bbox_scale(right_bbox), 1.0)
            if (center_jump / larger_size) >= 0.50:
                return True
    return False


def _history_index_for_frame(history: list[ClassObservation], frame_number: int) -> int:
    for index, item in enumerate(history):
        if int(item.frame_number) == frame_number:
            return index
    return max(len(history) - 1, 0)


def _bbox_center_distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_center_x = (float(left[0]) + float(left[2])) / 2.0
    left_center_y = (float(left[1]) + float(left[3])) / 2.0
    right_center_x = (float(right[0]) + float(right[2])) / 2.0
    right_center_y = (float(right[1]) + float(right[3])) / 2.0
    return ((right_center_x - left_center_x) ** 2 + (right_center_y - left_center_y) ** 2) ** 0.5


def _bbox_scale(bbox: tuple[float, float, float, float]) -> float:
    width = max(float(bbox[2]) - float(bbox[0]), 1.0)
    height = max(float(bbox[3]) - float(bbox[1]), 1.0)
    return max(width, height)


def class_diagnostics_to_metadata(diagnostics: TrackClassDiagnostics) -> dict[str, object]:
    return {
        "provisional_class_name": diagnostics.provisional_class_name,
        "stable_class_name": diagnostics.stable_class_name,
        "class_status": diagnostics.class_status,
        "final_class_reason": diagnostics.final_class_reason,
        "class_is_locked": diagnostics.class_is_locked,
        "class_confidence": diagnostics.class_confidence,
        "class_winner_margin": diagnostics.class_winner_margin,
        "class_observation_count": diagnostics.class_observation_count,
        "class_conflict_count": diagnostics.class_conflict_count,
        "winning_class_name": diagnostics.winning_class_name,
        "winning_class_count": diagnostics.winning_class_count,
        "winning_class_ratio": diagnostics.winning_class_ratio,
        "runner_up_class_name": diagnostics.runner_up_class_name,
        "runner_up_class_count": diagnostics.runner_up_class_count,
        "runner_up_ratio": diagnostics.runner_up_ratio,
        "winner_count_margin": diagnostics.winner_count_margin,
        "count_winner_class_name": diagnostics.count_winner_class_name,
        "score_winner_class_name": diagnostics.score_winner_class_name,
        "winners_agree": diagnostics.winners_agree,
        "maximum_consecutive_winner_count": diagnostics.maximum_consecutive_winner_count,
        "recent_consecutive_winner_count": diagnostics.recent_consecutive_winner_count,
        "class_transition_count": diagnostics.class_transition_count,
        "incompatible_class_transition_count": diagnostics.incompatible_class_transition_count,
        "recent_class_counts": dict(diagnostics.recent_class_counts),
        "recent_winning_class_name": diagnostics.recent_winning_class_name,
        "recent_winning_ratio": diagnostics.recent_winning_ratio,
        "recent_observation_count": diagnostics.recent_observation_count,
        "strong_conflict_detected": diagnostics.strong_conflict_detected,
        "split_recommended": diagnostics.split_recommended,
        "mixed_identity_detected": diagnostics.mixed_identity_detected,
        "mixed_identity_classes": list(diagnostics.mixed_identity_classes),
        "mixed_identity_start_frame": diagnostics.mixed_identity_start_frame,
        "mixed_identity_confidence": diagnostics.mixed_identity_confidence,
        "final_class_blocked_due_to_mixed_identity": diagnostics.final_class_blocked_due_to_mixed_identity,
        "class_scores": dict(diagnostics.class_scores),
        "class_ratios": dict(diagnostics.class_ratios),
        "class_observation_counts": dict(diagnostics.class_observation_counts),
        "class_max_confidences": dict(diagnostics.class_max_confidences),
        "winner_confidence_sum": diagnostics.winner_confidence_sum,
        "latest_observation_class_name": diagnostics.latest_observation_class_name,
        "possible_identity_switch": diagnostics.possible_identity_switch,
        "raw_class_history": [
            {
                **asdict(item),
                "camera_timestamp": item.camera_timestamp.isoformat() if item.camera_timestamp is not None else None,
            }
            for item in diagnostics.raw_class_history
        ],
        "linked_track_group_id": diagnostics.linked_track_group_id,
        "fragment_candidate_track_uuids": list(diagnostics.fragment_candidate_track_uuids),
    }


def track_class_metadata(track: LocalVehicleTrack) -> dict[str, object]:
    return class_diagnostics_to_metadata(
        TrackClassDiagnostics(
            provisional_class_name=track.provisional_class_name,
            stable_class_name=track.stable_class_name,
            class_status=track.class_status,
            final_class_reason=track.final_class_reason,
            class_is_locked=track.class_is_locked,
            class_confidence=track.class_confidence,
            class_winner_margin=track.class_winner_margin,
            class_observation_count=track.class_observation_count,
            class_conflict_count=track.class_conflict_count,
            winning_class_name=track.winning_class_name,
            winning_class_count=track.winning_class_count,
            winning_class_ratio=track.winning_class_ratio,
            runner_up_class_name=track.runner_up_class_name,
            runner_up_class_count=track.runner_up_class_count,
            runner_up_ratio=track.runner_up_ratio,
            winner_count_margin=track.winner_count_margin,
            count_winner_class_name=track.count_winner_class_name,
            score_winner_class_name=track.score_winner_class_name,
            winners_agree=track.winners_agree,
            maximum_consecutive_winner_count=track.maximum_consecutive_winner_count,
            recent_consecutive_winner_count=track.recent_consecutive_winner_count,
            class_transition_count=track.class_transition_count,
            incompatible_class_transition_count=track.incompatible_class_transition_count,
            recent_class_counts=dict(track.recent_class_counts),
            recent_winning_class_name=track.recent_winning_class_name,
            recent_winning_ratio=track.recent_winning_ratio,
            recent_observation_count=track.recent_observation_count,
            strong_conflict_detected=track.strong_conflict_detected,
            split_recommended=track.split_recommended,
            mixed_identity_detected=track.mixed_identity_detected,
            mixed_identity_classes=tuple(track.mixed_identity_classes),
            mixed_identity_start_frame=track.mixed_identity_start_frame,
            mixed_identity_confidence=track.mixed_identity_confidence,
            final_class_blocked_due_to_mixed_identity=track.final_class_blocked_due_to_mixed_identity,
            class_scores=dict(track.class_scores),
            class_ratios=dict(track.class_ratios),
            class_observation_counts=dict(track.class_observation_counts),
            class_max_confidences=dict(track.class_max_confidences),
            winner_confidence_sum=track.winner_confidence_sum,
            raw_class_history=list(track.raw_class_history),
            latest_observation_class_name=track.latest_observation_class_name,
            linked_track_group_id=track.linked_track_group_id,
            fragment_candidate_track_uuids=list(track.fragment_candidate_track_uuids),
            possible_identity_switch=track.possible_identity_switch,
        )
    )
