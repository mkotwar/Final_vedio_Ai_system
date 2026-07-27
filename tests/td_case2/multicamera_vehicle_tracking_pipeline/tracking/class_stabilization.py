from __future__ import annotations

from dataclasses import asdict

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
    latest_class_name = history[-1].class_name if history else None
    total_observations = len(history)
    ranked = []
    for class_name, confidence_sum in class_scores.items():
        count = int(class_counts.get(class_name, 0))
        max_confidence = float(class_max_confidences.get(class_name, 0.0))
        score = float(confidence_sum)
        if config.class_stabilization.enabled:
            score += config.class_stabilization.observation_count_weight * count
            score += config.class_stabilization.max_confidence_weight * max_confidence
        ranked.append((class_name, score, count, max_confidence))
    ranked.sort(key=lambda item: (item[1], item[2], item[3], item[0]), reverse=True)

    provisional = ranked[0][0] if ranked else latest_class_name
    winner_score = ranked[0][1] if ranked else 0.0
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    winner_margin = winner_score - runner_up_score if ranked else None

    stable = previous_stable_class_name
    locked = bool(previous_class_is_locked)
    class_confidence = None
    if ranked:
        total_score = sum(max(item[1], 0.0) for item in ranked)
        class_confidence = (winner_score / total_score) if total_score > 0 else None

    if stable is None and total_observations >= config.class_stabilization.minimum_observations and provisional is not None:
        stable = provisional
    if (
        not locked
        and stable is not None
        and total_observations >= config.class_stabilization.lock_after_observations
        and winner_margin is not None
        and winner_margin >= config.class_stabilization.minimum_winner_margin
    ):
        locked = True

    if locked and provisional is not None and provisional != stable and config.class_stabilization.allow_unlock_on_strong_conflict:
        provisional_count = int(class_counts.get(provisional, 0))
        if provisional_count >= config.class_stabilization.strong_conflict_min_observations and (winner_margin or 0.0) >= config.class_stabilization.strong_conflict_margin:
            stable = provisional

    if not locked and provisional is not None:
        stable = stable or provisional

    conflict_count = sum(count for name, count in class_counts.items() if stable is not None and name != stable)
    return TrackClassDiagnostics(
        provisional_class_name=provisional,
        stable_class_name=stable,
        class_is_locked=locked,
        class_confidence=class_confidence,
        class_winner_margin=winner_margin,
        class_observation_count=total_observations,
        class_conflict_count=conflict_count,
        class_scores={key: float(value) for key, value in sorted(class_scores.items())},
        class_observation_counts={key: int(value) for key, value in sorted(class_counts.items())},
        class_max_confidences={key: float(value) for key, value in sorted(class_max_confidences.items())},
        raw_class_history=list(history),
        latest_observation_class_name=latest_class_name,
    )


def class_diagnostics_to_metadata(diagnostics: TrackClassDiagnostics) -> dict[str, object]:
    return {
        "provisional_class_name": diagnostics.provisional_class_name,
        "stable_class_name": diagnostics.stable_class_name,
        "class_is_locked": diagnostics.class_is_locked,
        "class_confidence": diagnostics.class_confidence,
        "class_winner_margin": diagnostics.class_winner_margin,
        "class_observation_count": diagnostics.class_observation_count,
        "class_conflict_count": diagnostics.class_conflict_count,
        "class_scores": dict(diagnostics.class_scores),
        "class_observation_counts": dict(diagnostics.class_observation_counts),
        "class_max_confidences": dict(diagnostics.class_max_confidences),
        "latest_observation_class_name": diagnostics.latest_observation_class_name,
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
            class_is_locked=track.class_is_locked,
            class_confidence=track.class_confidence,
            class_winner_margin=track.class_winner_margin,
            class_observation_count=track.class_observation_count,
            class_conflict_count=track.class_conflict_count,
            class_scores=dict(track.class_scores),
            class_observation_counts=dict(track.class_observation_counts),
            class_max_confidences=dict(track.class_max_confidences),
            raw_class_history=list(track.raw_class_history),
            latest_observation_class_name=track.latest_observation_class_name,
            linked_track_group_id=track.linked_track_group_id,
            fragment_candidate_track_uuids=list(track.fragment_candidate_track_uuids),
        )
    )
