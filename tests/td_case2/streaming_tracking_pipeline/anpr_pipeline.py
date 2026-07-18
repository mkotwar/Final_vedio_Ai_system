from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .anpr_schemas import TrackAnprColourResult
from .config import Step7InferenceConfig
from .crop_selection import SelectedCropJob, SelectedTrackCropSet
from .florence_inference import FlorenceInferenceEngine
from .plate_detection import UltralyticsPlateDetectionStage


def track_key(job: SelectedCropJob) -> tuple[str, int, int]:
    return (job.source_id, job.track_id, job.track_generation)


def group_jobs_by_track(jobs: Iterable[SelectedCropJob]) -> list[list[SelectedCropJob]]:
    groups: dict[tuple[str, int, int], list[SelectedCropJob]] = defaultdict(list)
    for job in jobs:
        groups[track_key(job)].append(job)
    return [groups[key] for key in sorted(groups)]


class SequentialAnprColourPipeline:
    """Sequential Step 7 enrichment over Step 6 selected crop jobs."""

    def __init__(
        self,
        config: Step7InferenceConfig,
        *,
        plate_detector: UltralyticsPlateDetectionStage,
        florence_engine: FlorenceInferenceEngine,
    ) -> None:
        self.config = config
        self.plate_detector = plate_detector
        self.florence_engine = florence_engine

    def process_track(self, selected_set: SelectedTrackCropSet) -> TrackAnprColourResult:
        return self.process_jobs(selected_set.to_crop_jobs())

    def process_job_groups(self, groups: Iterable[Iterable[SelectedCropJob]]) -> list[TrackAnprColourResult]:
        return [self.process_jobs(group) for group in groups]

    def process_jobs(self, jobs: Iterable[SelectedCropJob]) -> TrackAnprColourResult:
        ordered_jobs = self._order_jobs(list(jobs))
        if not ordered_jobs:
            return TrackAnprColourResult(
                source_id="",
                track_id=0,
                track_generation=0,
                source_track_id=None,
                object_class=None,
                lifecycle_completion_reason=None,
                processing_status="no_jobs",
            )
        identity = ordered_jobs[0]
        selected_jobs = ordered_jobs[: self.config.maximum_vehicle_crops_per_track]
        plate_candidates = []
        ocr_results = []
        failures: list[str] = []

        colour_job = self._select_colour_job(selected_jobs)
        colour_result = self.florence_engine.run_colour(colour_job) if colour_job is not None else None
        if colour_result is not None and colour_result.status not in {"success", "empty_output"}:
            failures.append(f"colour:{colour_result.status}")

        for job in selected_jobs:
            try:
                candidates = self.plate_detector.detect(job)
            except FileNotFoundError as exc:
                failures.append(f"plate_load_error:{exc}")
                continue
            except Exception as exc:
                failures.append(f"plate_inference_error:{exc}")
                continue
            candidates = candidates[: self.config.maximum_plate_candidates_per_vehicle_crop]
            plate_candidates.extend(candidates)
            for candidate in candidates:
                result = self.florence_engine.run_ocr(candidate)
                ocr_results.append(result)
                if result.status not in {"success", "empty_output"}:
                    failures.append(f"ocr:{result.status}")
                if self.config.stop_after_first_raw_plate_text and result.normalized_text:
                    break
            if self.config.stop_after_first_raw_plate_text and any(result.normalized_text for result in ocr_results):
                break

        raw_texts = [result.raw_text for result in ocr_results if result.raw_text]
        normalized_texts = []
        for result in ocr_results:
            if result.normalized_text and result.normalized_text not in normalized_texts:
                normalized_texts.append(result.normalized_text)
        status = self._status(ordered_jobs, plate_candidates, ocr_results, colour_result, failures)
        return TrackAnprColourResult(
            source_id=identity.source_id,
            track_id=identity.track_id,
            track_generation=identity.track_generation,
            source_track_id=identity.source_track_id,
            object_class=identity.object_class,
            lifecycle_completion_reason=identity.lifecycle_completion_reason,
            processing_status=status,
            selected_crop_jobs=selected_jobs,
            plate_candidates=plate_candidates,
            ocr_results=ocr_results,
            colour_result=colour_result,
            raw_plate_texts=raw_texts,
            normalized_plate_texts=normalized_texts,
            normalized_colour=colour_result.normalized_colour if colour_result is not None else "unknown",
            failure_reasons=failures,
            metadata={"configured_job_count": len(ordered_jobs)},
        )

    def _order_jobs(self, jobs: list[SelectedCropJob]) -> list[SelectedCropJob]:
        filtered = [
            job
            for job in jobs
            if (job.crop_role == "primary" and self.config.process_primary_crops)
            or (job.crop_role == "fallback" and self.config.process_fallback_crops)
        ]
        role_order = {"primary": 0, "fallback": 1}
        return sorted(filtered, key=lambda job: (role_order.get(job.crop_role, 9), job.crop_rank, job.frame_index))

    def _select_colour_job(self, jobs: list[SelectedCropJob]) -> SelectedCropJob | None:
        if not jobs:
            return None
        if not self.config.run_colour_once_per_track:
            return jobs[0]
        if self.config.prefer_primary_for_colour:
            for job in jobs:
                if job.crop_role == "primary":
                    return job
        return jobs[0]

    def _status(self, jobs: list[SelectedCropJob], plate_candidates: list[object], ocr_results: list[object], colour_result: object | None, failures: list[str]) -> str:
        if not jobs:
            return "no_jobs"
        if failures and not plate_candidates and colour_result is None:
            if any(item.startswith("plate_load_error") for item in failures):
                return "load_error"
            return "inference_error"
        if not plate_candidates:
            return "no_plate_candidates"
        if failures:
            return "partial"
        if not ocr_results and colour_result is None:
            return "model_disabled"
        return "success"
