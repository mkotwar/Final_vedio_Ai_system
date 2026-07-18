from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import FlorenceColourResult, FlorenceOcrResult, PlateDetectionCandidate
from tests.td_case2.streaming_tracking_pipeline.anpr_pipeline import SequentialAnprColourPipeline, group_jobs_by_track
from tests.td_case2.streaming_tracking_pipeline.config import Step7InferenceConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob


def _job(role: str, rank: int) -> SelectedCropJob:
    return SelectedCropJob("cam", 1, 0, "raw", "car", "done", role, rank, rank, float(rank), f"{role}.jpg", None, 0.8)


class FakePlateStage:
    def __init__(self) -> None:
        self.seen_roles: list[str] = []

    def detect(self, job: SelectedCropJob):
        self.seen_roles.append(job.crop_role)
        return [
            PlateDetectionCandidate(
                job.source_id,
                job.track_id,
                job.track_generation,
                job.crop_role,
                job.crop_rank,
                job.frame_index,
                job.vehicle_crop_path,
                1,
                0.9,
                (1, 1, 20, 8),
                (0, 0, 21, 9),
                "plate.jpg",
            )
        ]


class FakeFlorence:
    def run_colour(self, job: SelectedCropJob):
        return FlorenceColourResult(job.source_id, job.track_id, job.track_generation, job.crop_role, job.crop_rank, job.frame_index, job.vehicle_crop_path, "white", "white", "success", "<VQA>")

    def run_ocr(self, candidate):
        return FlorenceOcrResult(candidate.source_id, candidate.track_id, candidate.track_generation, candidate.crop_role, candidate.crop_rank, candidate.frame_index, candidate.plate_rank, candidate.plate_crop_path, "MH12AB1234", "MH12AB1234", "success", "<OCR>")


class AnprPipelineTests(unittest.TestCase):
    def test_primary_order_and_stop_after_first_text(self) -> None:
        plate_stage = FakePlateStage()
        pipeline = SequentialAnprColourPipeline(Step7InferenceConfig(), plate_detector=plate_stage, florence_engine=FakeFlorence())
        result = pipeline.process_jobs([_job("fallback", 1), _job("primary", 2)])
        self.assertEqual(result.processing_status, "success")
        self.assertEqual(plate_stage.seen_roles, ["primary"])
        self.assertEqual(result.normalized_plate_texts, ["MH12AB1234"])

    def test_group_jobs_by_track(self) -> None:
        groups = group_jobs_by_track([_job("primary", 1), SelectedCropJob("cam", 2, 0, None, "car", "done", "primary", 1, 1, 1.0, "x.jpg", None, 0.8)])
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()
