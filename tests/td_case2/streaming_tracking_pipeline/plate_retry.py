from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .config import PlateDiagnosticConfig
from .crop_selection import SelectedCropJob, SelectedTrackCropSet
from .florence_inference import FlorenceInferenceEngine
from .plate_detection import UltralyticsPlateDetectionStage
from .plate_diagnostics import PlateAttemptStatus, PlateDiagnosticProcessor, TrackPlateDiagnosticResult


def group_selected_jobs_by_track(jobs: Iterable[SelectedCropJob]) -> list[list[SelectedCropJob]]:
    groups: dict[tuple[str, int, int], list[SelectedCropJob]] = defaultdict(list)
    for job in jobs:
        groups[(job.source_id, job.track_id, job.track_generation)].append(job)
    return [groups[key] for key in sorted(groups)]


class BoundedPlateRetryController:
    def __init__(
        self,
        plate_detector: UltralyticsPlateDetectionStage,
        florence_engine: FlorenceInferenceEngine | None,
        config: PlateDiagnosticConfig,
    ) -> None:
        self.plate_detector = plate_detector
        self.florence_engine = florence_engine
        self.config = config
        self.processor = PlateDiagnosticProcessor(
            detector_stage=plate_detector,
            plate_config=plate_detector.config,
            diagnostic_config=config,
            output_dir=plate_detector.output_dir,
            florence_engine=florence_engine,
        )

    def process_track(self, selected_set: SelectedTrackCropSet) -> TrackPlateDiagnosticResult:
        return self.process_jobs(selected_set.to_crop_jobs())

    def process_many(self, selected_sets: Sequence[SelectedTrackCropSet]) -> list[TrackPlateDiagnosticResult]:
        return [self.process_track(item) for item in selected_sets]

    def process_job_groups(self, groups: Iterable[Iterable[SelectedCropJob]]) -> list[TrackPlateDiagnosticResult]:
        return [self.process_jobs(group) for group in groups]

    def process_jobs(self, jobs: Iterable[SelectedCropJob]) -> TrackPlateDiagnosticResult:
        job_list = list(jobs)
        ordered = self._retry_order(job_list)
        if not ordered:
            return TrackPlateDiagnosticResult(
                source_id="",
                track_id=0,
                track_generation=0,
                source_track_id=None,
                object_class=None,
                attempts=[],
                selected_attempt_number=None,
                selected_plate_candidate=None,
                selected_ocr_result=None,
                final_status="no_plate_candidate",
                final_failure_reasons=["no_selected_crop_jobs"],
                exhausted_selected_crops=True,
            )
        identity = ordered[0]
        attempts = []
        selected_candidate = None
        selected_ocr = None
        seen_paths: set[str] = set()
        for job in ordered:
            normalized_path = job.vehicle_crop_path.lower()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            if len(attempts) >= self.config.maximum_vehicle_crop_attempts_per_track:
                break
            attempt = self.processor.process_job(job, attempt_number=len(attempts) + 1)
            attempts.append(attempt)
            if attempt.accepted_candidates and selected_candidate is None:
                selected_candidate = attempt.accepted_candidates[0]
                if attempt.ocr_results:
                    selected_ocr = attempt.ocr_results[0]
            if self.config.stop_after_first_valid_plate_candidate and attempt.accepted_candidates:
                break
            if self.config.stop_after_first_non_empty_ocr_text:
                non_empty = next((result for result in attempt.ocr_results if result.normalized_text), None)
                if non_empty is not None:
                    selected_ocr = non_empty
                    selected_candidate = attempt.accepted_candidates[0] if attempt.accepted_candidates else selected_candidate
                    break
        final_status = self._final_status(attempts)
        return TrackPlateDiagnosticResult(
            source_id=identity.source_id,
            track_id=identity.track_id,
            track_generation=identity.track_generation,
            source_track_id=identity.source_track_id,
            object_class=identity.object_class,
            attempts=attempts,
            selected_attempt_number=next((attempt.attempt_number for attempt in attempts if attempt.accepted_candidates), None),
            selected_plate_candidate=selected_candidate,
            selected_ocr_result=selected_ocr,
            final_status=final_status,
            final_failure_reasons=self._failure_reasons(attempts),
            exhausted_selected_crops=len(attempts) >= min(len(self._retry_order(job_list)), self.config.maximum_vehicle_crop_attempts_per_track)
            and final_status == "no_plate_candidate",
            metadata={"configured_retry_order": list(self.config.retry_order), "available_selected_jobs": len(job_list)},
        )

    def _retry_order(self, jobs: list[SelectedCropJob]) -> list[SelectedCropJob]:
        by_role_rank: dict[tuple[str, int], list[SelectedCropJob]] = defaultdict(list)
        fallback: list[SelectedCropJob] = []
        for job in jobs:
            if job.crop_role == "primary":
                by_role_rank[(job.crop_role, job.crop_rank)].append(job)
            elif job.crop_role == "fallback":
                fallback.append(job)
        ordered: list[SelectedCropJob] = []
        for item in self.config.retry_order:
            if item.startswith("primary_rank_"):
                rank = int(item.rsplit("_", 1)[1])
                ordered.extend(sorted(by_role_rank.get(("primary", rank), []), key=lambda job: (job.frame_index, job.selection_score)))
            elif item == "fallback":
                ordered.extend(sorted(fallback, key=lambda job: (job.crop_rank, job.frame_index, job.selection_score)))
        return ordered

    def _final_status(self, attempts: list[object]) -> str:
        if not self.config.enabled:
            return "disabled"
        if not attempts:
            return "no_plate_candidate"
        if any(attempt.attempt_status in {PlateAttemptStatus.DETECTOR_DISABLED} for attempt in attempts):
            return "disabled"
        if any(attempt.attempt_status in {PlateAttemptStatus.DETECTOR_LOAD_ERROR, PlateAttemptStatus.DETECTOR_INFERENCE_ERROR} for attempt in attempts):
            return "detector_failure"
        if all(attempt.attempt_status in {PlateAttemptStatus.VEHICLE_CROP_MISSING, PlateAttemptStatus.VEHICLE_CROP_UNREADABLE} for attempt in attempts):
            return "input_failure"
        if any(result.normalized_text for attempt in attempts for result in attempt.ocr_results):
            return "plate_found_ocr_non_empty"
        if any(attempt.accepted_candidates for attempt in attempts):
            if self.config.run_ocr_on_valid_plate_candidates:
                return "plate_found_ocr_empty"
            return "plate_found_ocr_not_run"
        return "no_plate_candidate"

    def _failure_reasons(self, attempts: list[object]) -> list[str]:
        reasons: list[str] = []
        for attempt in attempts:
            if attempt.error_message:
                reasons.append(attempt.error_message)
            for raw_box in attempt.raw_boxes:
                if raw_box.rejection_reason and raw_box.rejection_reason not in reasons:
                    reasons.append(raw_box.rejection_reason)
            if not attempt.raw_boxes and attempt.attempt_status.value not in reasons:
                reasons.append(attempt.attempt_status.value)
        return reasons
