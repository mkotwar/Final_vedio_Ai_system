from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .plate_diagnostics import PlateBoxDisposition, TrackPlateDiagnosticResult


def score_bucket(value: float) -> str:
    if value >= 0.75:
        return "0.75-1.00"
    if value >= 0.50:
        return "0.50-0.74"
    if value >= 0.25:
        return "0.25-0.49"
    return "0.00-0.24"


def build_plate_diagnostic_metrics(
    results: Iterable[TrackPlateDiagnosticResult],
    *,
    processor_metrics: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(results)
    attempt_status_counts = Counter()
    final_status_counts = Counter(row.final_status for row in rows)
    rejection_reason_counts = Counter()
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_rank: dict[str, Counter[str]] = defaultdict(Counter)
    by_threshold: dict[str, Counter[str]] = defaultdict(Counter)
    by_generation: dict[str, Counter[str]] = defaultdict(Counter)
    by_score_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_size_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    dispositions = Counter()
    raw_boxes_by_threshold = Counter()
    metrics = Counter()

    for row in rows:
        if row.selected_plate_candidate is not None:
            metrics["tracks_with_plate_candidate"] += 1
        else:
            metrics["tracks_without_plate_candidate"] += 1
        if row.selected_ocr_result is not None and row.selected_ocr_result.normalized_text:
            metrics["tracks_with_non_empty_ocr"] += 1
        elif any(attempt.ocr_results for attempt in row.attempts):
            metrics["tracks_with_empty_ocr"] += 1
        if row.exhausted_selected_crops:
            metrics["tracks_exhausting_all_crops"] += 1
        by_generation[str(row.track_generation)][row.final_status] += 1
        for attempt in row.attempts:
            metrics["vehicle_crop_attempts"] += 1
            if attempt.vehicle_crop_role == "primary":
                metrics["primary_attempts"] += 1
            if attempt.vehicle_crop_role == "fallback":
                metrics["fallback_attempts"] += 1
            if attempt.stop_reason == "stop_after_first_valid_plate_candidate":
                metrics["attempts_stopped_after_plate"] += 1
            if attempt.stop_reason == "stop_after_first_non_empty_ocr_text":
                metrics["attempts_stopped_after_ocr"] += 1
            attempt_status_counts[attempt.attempt_status.value] += 1
            by_role[attempt.vehicle_crop_role][attempt.attempt_status.value] += 1
            by_rank[f"{attempt.vehicle_crop_role}:{attempt.vehicle_crop_rank}"][attempt.attempt_status.value] += 1
            selection_score = float(attempt.metadata.get("vehicle_crop", {}).get("selection_score", 0.0) or 0.0)
            by_score_bucket[score_bucket(selection_score)][attempt.attempt_status.value] += 1
            by_size_bucket[str(attempt.metadata.get("vehicle_crop_size_bucket", "unknown"))][attempt.attempt_status.value] += 1
            metrics["raw_detector_boxes"] += attempt.raw_box_count
            metrics["boxes_below_threshold"] += attempt.below_threshold_box_count
            metrics["boxes_invalid_geometry"] += attempt.invalid_geometry_count
            metrics["boxes_empty_after_clipping"] += attempt.empty_after_clipping_count
            metrics["boxes_too_small"] += attempt.too_small_count
            metrics["accepted_plate_candidates"] += attempt.accepted_plate_count
            metrics["ocr_calls"] += len(attempt.ocr_results)
            for ocr in attempt.ocr_results:
                if ocr.normalized_text:
                    metrics["ocr_non_empty_outputs"] += 1
                elif ocr.status == "inference_error":
                    metrics["ocr_failures"] += 1
                else:
                    metrics["ocr_empty_outputs"] += 1
            for threshold in attempt.diagnostic_thresholds_used:
                by_threshold[f"{threshold:.3f}"][attempt.attempt_status.value] += 1
            for raw_box in attempt.raw_boxes:
                dispositions[raw_box.disposition.value] += 1
                raw_boxes_by_threshold[f"{raw_box.diagnostic_threshold:.3f}"] += 1
                if raw_box.disposition == PlateBoxDisposition.WRONG_CLASS:
                    metrics["boxes_wrong_class"] += 1
                if raw_box.rejection_reason:
                    rejection_reason_counts[raw_box.rejection_reason] += 1
    payload = {
        "tracks_processed": len(rows),
        "tracks_with_plate_candidate": metrics["tracks_with_plate_candidate"],
        "tracks_without_plate_candidate": metrics["tracks_without_plate_candidate"],
        "tracks_with_non_empty_ocr": metrics["tracks_with_non_empty_ocr"],
        "tracks_with_empty_ocr": metrics["tracks_with_empty_ocr"],
        "vehicle_crop_attempts": metrics["vehicle_crop_attempts"],
        "primary_attempts": metrics["primary_attempts"],
        "fallback_attempts": metrics["fallback_attempts"],
        "plate_model_calls": int((processor_metrics or {}).get("plate_model_calls", 0)),
        "threshold_probe_calls": int((processor_metrics or {}).get("threshold_probe_calls", 0)),
        "raw_detector_boxes": metrics["raw_detector_boxes"],
        "raw_boxes_by_threshold": dict(raw_boxes_by_threshold),
        "boxes_below_threshold": metrics["boxes_below_threshold"],
        "boxes_wrong_class": metrics["boxes_wrong_class"],
        "boxes_invalid_geometry": metrics["boxes_invalid_geometry"],
        "boxes_empty_after_clipping": metrics["boxes_empty_after_clipping"],
        "boxes_too_small": metrics["boxes_too_small"],
        "accepted_plate_candidates": metrics["accepted_plate_candidates"],
        "attempts_stopped_after_plate": metrics["attempts_stopped_after_plate"],
        "attempts_stopped_after_ocr": metrics["attempts_stopped_after_ocr"],
        "tracks_exhausting_all_crops": metrics["tracks_exhausting_all_crops"],
        "vehicle_crop_read_failures": attempt_status_counts.get("vehicle_crop_missing", 0)
        + attempt_status_counts.get("vehicle_crop_unreadable", 0),
        "annotated_images_written": int((processor_metrics or {}).get("annotated_images_written", 0)),
        "annotation_write_failures": int((processor_metrics or {}).get("annotation_write_failures", 0)),
        "rejected_plate_crops_written": int((processor_metrics or {}).get("rejected_plate_crops_written", 0)),
        "accepted_plate_crops_written": int((processor_metrics or {}).get("accepted_plate_crops_written", 0)),
        "ocr_calls": metrics["ocr_calls"],
        "ocr_non_empty_outputs": metrics["ocr_non_empty_outputs"],
        "ocr_empty_outputs": metrics["ocr_empty_outputs"],
        "ocr_failures": metrics["ocr_failures"],
        "attempt_status_counts": dict(attempt_status_counts),
        "final_status_counts": dict(final_status_counts),
        "rejection_reason_counts": dict(rejection_reason_counts),
        "box_disposition_counts": dict(dispositions),
        "by_crop_role": {key: dict(value) for key, value in sorted(by_role.items())},
        "by_crop_rank": {key: dict(value) for key, value in sorted(by_rank.items())},
        "by_diagnostic_threshold": {key: dict(value) for key, value in sorted(by_threshold.items())},
        "by_track_generation": {key: dict(value) for key, value in sorted(by_generation.items())},
        "by_step6_selection_score_bucket": {key: dict(value) for key, value in sorted(by_score_bucket.items())},
        "by_vehicle_crop_size_bucket": {key: dict(value) for key, value in sorted(by_size_bucket.items())},
    }
    if model_metadata:
        payload["model_metadata"] = model_metadata
    return payload
