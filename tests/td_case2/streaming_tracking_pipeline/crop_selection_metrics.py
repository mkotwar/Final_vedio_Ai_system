from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .crop_selection import SelectedCrop, SelectedTrackCropSet


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _median(values: list[float]) -> float:
    return round(float(median(values)), 6) if values else 0.0


def _bucket_observation_count(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return "11+"


@dataclass
class BestCropSelectionMetricsAccumulator:
    """Aggregate Step 6 crop-selection results without claiming OCR readiness."""

    completed_track_bundles_processed: int = 0
    tracks_with_primary_crops: int = 0
    tracks_with_fallback_only: int = 0
    tracks_with_no_valid_crop: int = 0
    total_primary_crops_selected: int = 0
    total_fallback_crops_selected: int = 0
    tracks_reaching_primary_target: int = 0
    tracks_below_primary_target: int = 0
    primary_candidate_rejections: int = 0
    fallback_candidate_rejections: int = 0
    strict_diversity_selections: int = 0
    relaxed_diversity_selections: int = 0
    duplicate_candidates_removed: int = 0
    rejection_reason_counts: Counter[str] = field(default_factory=Counter)
    selection_status_counts: Counter[str] = field(default_factory=Counter)
    missing_metric_counts: Counter[str] = field(default_factory=Counter)
    completion_reason_counts: Counter[str] = field(default_factory=Counter)
    dominant_class_counts: Counter[str] = field(default_factory=Counter)
    observation_bucket_counts: Counter[str] = field(default_factory=Counter)
    primary_scores: list[float] = field(default_factory=list)
    fallback_scores: list[float] = field(default_factory=list)
    primary_counts_per_track: list[int] = field(default_factory=list)
    component_sums: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    component_counts: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, result: SelectedTrackCropSet, *, primary_target: int) -> None:
        self.completed_track_bundles_processed += 1
        self.selection_status_counts[result.selection_status] += 1
        if result.primary_crops:
            self.tracks_with_primary_crops += 1
        if result.selection_status == "fallback_only":
            self.tracks_with_fallback_only += 1
        if result.selection_status == "no_valid_crop":
            self.tracks_with_no_valid_crop += 1
        primary_count = len(result.primary_crops)
        self.primary_counts_per_track.append(primary_count)
        fallback_count = 1 if result.fallback_crop is not None else 0
        self.total_primary_crops_selected += primary_count
        self.total_fallback_crops_selected += fallback_count
        if primary_count >= primary_target:
            self.tracks_reaching_primary_target += 1
        else:
            self.tracks_below_primary_target += 1
        self.strict_diversity_selections += int(result.metadata.get("strict_diversity_selection_count", 0) or 0)
        self.relaxed_diversity_selections += int(result.metadata.get("relaxed_diversity_selection_count", 0) or 0)
        self.duplicate_candidates_removed += int(result.metadata.get("duplicate_candidates_removed", 0) or 0)
        completion_reason = (
            result.lifecycle_record.completion_reason.value if result.lifecycle_record.completion_reason is not None else "unknown"
        )
        self.completion_reason_counts[completion_reason] += 1
        self.dominant_class_counts[result.lifecycle_record.dominant_class or "unknown"] += 1
        self.observation_bucket_counts[_bucket_observation_count(result.lifecycle_record.observation_count)] += 1
        for reason, count in result.rejection_reason_counts.items():
            self.rejection_reason_counts[reason] += int(count)
            if reason.startswith("fallback_"):
                self.fallback_candidate_rejections += int(count)
            else:
                self.primary_candidate_rejections += int(count)
        for crop in result.primary_crops:
            self._record_crop(crop, primary=True)
        if result.fallback_crop is not None:
            self._record_crop(result.fallback_crop, primary=False)

    def to_dict(self) -> dict[str, Any]:
        average_primary = (
            round(self.total_primary_crops_selected / self.completed_track_bundles_processed, 6)
            if self.completed_track_bundles_processed
            else 0.0
        )
        return {
            "completed_track_bundles_processed": self.completed_track_bundles_processed,
            "tracks_with_primary_crops": self.tracks_with_primary_crops,
            "tracks_with_fallback_only": self.tracks_with_fallback_only,
            "tracks_with_no_valid_crop": self.tracks_with_no_valid_crop,
            "total_primary_crops_selected": self.total_primary_crops_selected,
            "total_fallback_crops_selected": self.total_fallback_crops_selected,
            "average_primary_crops_per_track": average_primary,
            "median_primary_crops_per_track": _median([float(value) for value in self.primary_counts_per_track]),
            "tracks_reaching_primary_target": self.tracks_reaching_primary_target,
            "tracks_below_primary_target": self.tracks_below_primary_target,
            "primary_candidate_rejections": self.primary_candidate_rejections,
            "fallback_candidate_rejections": self.fallback_candidate_rejections,
            "rejection_reason_counts": dict(sorted(self.rejection_reason_counts.items())),
            "selection_status_counts": dict(sorted(self.selection_status_counts.items())),
            "average_selected_primary_score": _average(self.primary_scores),
            "median_selected_primary_score": _median(self.primary_scores),
            "average_selected_fallback_score": _average(self.fallback_scores),
            "median_selected_fallback_score": _median(self.fallback_scores),
            "strict_diversity_selections": self.strict_diversity_selections,
            "relaxed_diversity_selections": self.relaxed_diversity_selections,
            "duplicate_candidates_removed": self.duplicate_candidates_removed,
            "missing_metric_counts": dict(sorted(self.missing_metric_counts.items())),
            "score_component_averages": self._component_averages(),
            "by_completion_reason": dict(sorted(self.completion_reason_counts.items())),
            "by_dominant_class": dict(sorted(self.dominant_class_counts.items())),
            "by_observation_count_bucket": dict(sorted(self.observation_bucket_counts.items())),
        }

    def _record_crop(self, crop: SelectedCrop, *, primary: bool) -> None:
        if primary:
            self.primary_scores.append(crop.final_score)
        else:
            self.fallback_scores.append(crop.final_score)
        for missing in crop.score_breakdown.missing_metric_names:
            self.missing_metric_counts[missing] += 1
        for name, value in crop.score_breakdown.to_dict().items():
            if name in {"final_score", "missing_metric_names"}:
                continue
            self.component_sums[name] += float(value)
            self.component_counts[name] += 1

    def _component_averages(self) -> dict[str, float]:
        return {
            name: round(self.component_sums[name] / max(self.component_counts[name], 1), 6)
            for name in sorted(self.component_sums)
        }


def build_selection_summary(results: list[SelectedTrackCropSet], *, primary_target: int) -> dict[str, Any]:
    accumulator = BestCropSelectionMetricsAccumulator()
    primary_counts: list[int] = []
    for result in results:
        accumulator.update(result, primary_target=primary_target)
        primary_counts.append(len(result.primary_crops))
    summary = accumulator.to_dict()
    summary["median_primary_crops_per_track"] = _median([float(value) for value in primary_counts])
    summary["average_primary_crops_per_track"] = _average([float(value) for value in primary_counts])
    return summary
