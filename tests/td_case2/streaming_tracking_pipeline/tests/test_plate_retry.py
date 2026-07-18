from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.config import PlateDetectionConfig, PlateDiagnosticConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.plate_detection import UltralyticsPlateDetectionStage
from tests.td_case2.streaming_tracking_pipeline.plate_retry import BoundedPlateRetryController


class PathDrivenModel:
    names = {0: "license_plate"}

    def predict(self, source, **kwargs):
        if "fail" in str(source):
            return {"boxes": []}
        return {"boxes": [{"bbox": [10, 10, 40, 24], "confidence": 0.9, "class_id": 0}]}


def _job(path: Path, role: str, rank: int, generation: int = 0) -> SelectedCropJob:
    return SelectedCropJob("cam", 1, generation, "raw", "car", "done", role, rank, rank, float(rank), str(path), None, 0.9 - rank * 0.1)


class PlateRetryTests(unittest.TestCase):
    def test_retry_order_second_primary_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fail = Path(directory) / "fail.jpg"
            ok = Path(directory) / "ok.jpg"
            Image.new("RGB", (80, 50), "white").save(fail)
            Image.new("RGB", (80, 50), "white").save(ok)
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=directory, model=PathDrivenModel())
            controller = BoundedPlateRetryController(stage, None, PlateDiagnosticConfig(stop_after_first_valid_plate_candidate=True, save_annotated_vehicle_crops=False))
            result = controller.process_jobs([_job(fail, "primary", 1), _job(ok, "primary", 2)])
        self.assertEqual(result.selected_attempt_number, 2)
        self.assertEqual(result.final_status, "plate_found_ocr_not_run")

    def test_fallback_succeeds_and_duplicate_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fail = Path(directory) / "fail.jpg"
            ok = Path(directory) / "ok.jpg"
            Image.new("RGB", (80, 50), "white").save(fail)
            Image.new("RGB", (80, 50), "white").save(ok)
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=directory, model=PathDrivenModel())
            controller = BoundedPlateRetryController(stage, None, PlateDiagnosticConfig(save_annotated_vehicle_crops=False))
            result = controller.process_jobs([_job(fail, "primary", 1), _job(ok, "fallback", 1), _job(ok, "fallback", 1)])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.selected_plate_candidate.crop_role, "fallback")

    def test_generation_identity_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ok = Path(directory) / "ok.jpg"
            Image.new("RGB", (80, 50), "white").save(ok)
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=directory, model=PathDrivenModel())
            controller = BoundedPlateRetryController(stage, None, PlateDiagnosticConfig(save_annotated_vehicle_crops=False))
            result = controller.process_jobs([_job(ok, "primary", 1, generation=2)])
        self.assertEqual(result.track_generation, 2)


if __name__ == "__main__":
    unittest.main()
