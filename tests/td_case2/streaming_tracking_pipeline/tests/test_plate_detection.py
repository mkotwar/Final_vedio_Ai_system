from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.config import PlateDetectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.plate_detection import UltralyticsPlateDetectionStage


class FakePlateModel:
    def __call__(self, image, **kwargs):
        return [
            {"bbox": [10, 10, 40, 25], "confidence": 0.90},
            {"bbox": [0, 0, 2, 2], "confidence": 0.99},
            {"bbox": [5, 5, 30, 20], "confidence": 0.10},
        ]


def _job(path: Path) -> SelectedCropJob:
    return SelectedCropJob(
        source_id="cam",
        track_id=7,
        track_generation=1,
        source_track_id="raw",
        object_class="car",
        lifecycle_completion_reason="done",
        crop_role="primary",
        crop_rank=1,
        frame_index=3,
        timestamp_sec=1.5,
        vehicle_crop_path=str(path),
        full_frame_path=None,
        selection_score=0.9,
    )


class PlateDetectionTests(unittest.TestCase):
    def test_filters_sorts_and_saves_plate_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 50), "white").save(crop)
            stage = UltralyticsPlateDetectionStage(
                PlateDetectionConfig(confidence_threshold=0.2, minimum_plate_width=5, minimum_plate_height=4),
                output_dir=directory,
                model=FakePlateModel(),
            )
            candidates = stage.detect(_job(crop))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].plate_rank, 1)
        self.assertTrue(candidates[0].plate_crop_path)
        self.assertEqual(stage.metrics["plate_candidates_rejected"], 2)

    def test_missing_crop_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=directory, model=FakePlateModel())
            self.assertEqual(stage.detect(_job(Path(directory) / "missing.jpg")), [])
            self.assertEqual(stage.metrics["vehicle_crops_missing"], 1)


if __name__ == "__main__":
    unittest.main()
