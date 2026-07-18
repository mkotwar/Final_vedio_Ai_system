from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.anpr_job_eligibility import (
    AnprJobEligibilityConfig,
    filter_anpr_eligible_jobs,
)
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob


def _write_image(path: Path, width: int, height: int) -> None:
    import cv2
    import numpy as np

    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _job(path: Path, object_class: str | None = "car") -> SelectedCropJob:
    return SelectedCropJob(
        source_id="cam",
        track_id=1,
        track_generation=0,
        source_track_id="raw_1",
        object_class=object_class,
        lifecycle_completion_reason="video_ended",
        crop_role="primary",
        crop_rank=1,
        frame_index=10,
        timestamp_sec=1.0,
        vehicle_crop_path=str(path),
        full_frame_path=None,
        selection_score=0.8,
    )


class AnprJobEligibilityTests(unittest.TestCase):
    def test_keeps_vehicle_crop_that_passes_size_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "crop.jpg"
            _write_image(crop, 90, 40)
            eligible, records = filter_anpr_eligible_jobs([_job(crop)], AnprJobEligibilityConfig())
            self.assertEqual(len(eligible), 1)
            self.assertTrue(records[0].eligible)
            self.assertIsNone(records[0].exclusion_reason)
            self.assertEqual(records[0].crop_area, 3600)

    def test_excludes_person_and_too_small_vehicle_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            person_crop = Path(directory) / "person.jpg"
            small_crop = Path(directory) / "small.jpg"
            _write_image(person_crop, 90, 40)
            _write_image(small_crop, 30, 20)
            jobs = [_job(person_crop, "person"), _job(small_crop, "car")]
            eligible, records = filter_anpr_eligible_jobs(jobs, AnprJobEligibilityConfig())
            self.assertEqual(eligible, [])
            self.assertEqual([record.exclusion_reason for record in records], ["non_vehicle_class", "vehicle_crop_too_small"])

    def test_records_missing_and_unreadable_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jpg"
            unreadable = Path(directory) / "bad.jpg"
            unreadable.write_text("not an image", encoding="utf-8")
            _eligible, records = filter_anpr_eligible_jobs([_job(missing), _job(unreadable)], AnprJobEligibilityConfig())
            self.assertEqual([record.exclusion_reason for record in records], ["vehicle_crop_missing", "vehicle_crop_unreadable"])


if __name__ == "__main__":
    unittest.main()
