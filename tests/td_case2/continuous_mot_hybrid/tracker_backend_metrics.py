from __future__ import annotations

import statistics
from typing import Any


def _safe_mean(values: list[float]) -> float | str:
    return round(sum(values) / len(values), 6) if values else "not_available"


def _safe_median(values: list[float]) -> float | str:
    return round(statistics.median(values), 6) if values else "not_available"


def _safe_max(values: list[float]) -> float | str:
    return round(max(values), 6) if values else "not_available"


def summarize_track_durations(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(item.get("duration", 0.0)) for item in tracks]
    return {
        "average_duration": _safe_mean(durations),
        "median_duration": _safe_median(durations),
        "maximum_duration": _safe_max(durations),
        "tracks_under_0_5_seconds": len([value for value in durations if value < 0.5]),
        "tracks_under_1_0_seconds": len([value for value in durations if value < 1.0]),
    }


def build_reid_metric_block(verification: dict[str, Any]) -> dict[str, Any]:
    actual = bool(verification.get("actual_with_reid"))
    return {
        "embedding_extractions": verification.get("feature_vector_count", "not_available"),
        "appearance_comparisons": verification.get("appearance_comparisons", "not_available"),
        "accepted_appearance_matches": verification.get("accepted_appearance_matches", "not_available"),
        "rejected_appearance_matches": verification.get("rejected_appearance_matches", "not_available"),
        "average_appearance_similarity": verification.get("average_appearance_similarity", "not_available"),
        "reid_runtime_overhead_seconds": verification.get("reid_runtime_overhead_seconds", "not_available"),
        "reid_gpu_memory_overhead_mb": verification.get("reid_gpu_memory_overhead_mb", "not_available"),
        "actual_with_reid": actual,
    }


def winner_for_metric(values: dict[str, float | int | str | None], *, lower_is_better: bool) -> str:
    comparable = {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}
    if not comparable:
        return "inconclusive"
    target = min(comparable.values()) if lower_is_better else max(comparable.values())
    winners = sorted([key for key, value in comparable.items() if value == target])
    if len(winners) != 1:
        return "approximately_equal"
    return winners[0]
