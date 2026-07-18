from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.config import PlateDetectionConfig, PlateDiagnosticConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.plate_detection import UltralyticsPlateDetectionStage
from tests.td_case2.streaming_tracking_pipeline.plate_diagnostics import (
    PlateBoxDisposition,
    PlateDiagnosticProcessor,
    extract_raw_plate_boxes,
)


class FakeModel:
    names = {0: "license_plate", 1: "vehicle"}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, source, **kwargs):
        return {"boxes": self.boxes}


def _image(path: Path) -> None:
    Image.new("RGB", (80, 50), "white").save(path)


def _job(path: Path) -> SelectedCropJob:
    return SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 10, 2.5, str(path), None, 0.8, metadata={"source_bbox": [5, 5, 50, 30]})


class PlateDiagnosticsTests(unittest.TestCase):
    def test_raw_box_preserves_class_and_confidence(self) -> None:
        boxes = extract_raw_plate_boxes({"boxes": [{"bbox": [1, 2, 3, 4], "confidence": 0.3, "class_id": 0}]}, model_names={0: "plate"})
        self.assertEqual(boxes[0].class_name, "plate")
        self.assertEqual(boxes[0].confidence, 0.3)

    def test_classifies_rejections_and_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.jpg"
            _image(path)
            model = FakeModel(
                [
                    {"bbox": [10, 10, 12, 20], "confidence": 0.9, "class_id": 0},
                    {"bbox": [15, 10, 35, 20], "confidence": 0.9, "class_id": 1},
                    {"bbox": [20, 10, 50, 24], "confidence": 0.12, "class_id": 0},
                ]
            )
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(confidence_threshold=0.2), output_dir=directory, model=model)
            processor = PlateDiagnosticProcessor(
                detector_stage=stage,
                plate_config=stage.config,
                diagnostic_config=PlateDiagnosticConfig(diagnostic_confidence_thresholds=(0.2, 0.1, 0.05), save_annotated_vehicle_crops=False),
                output_dir=directory,
            )
            attempt = processor.process_job(_job(path), attempt_number=1)
        self.assertEqual(attempt.raw_box_count, 3)
        self.assertEqual(attempt.below_threshold_box_count, 1)
        self.assertIn(PlateBoxDisposition.TOO_SMALL, [box.disposition for box in attempt.raw_boxes])
        self.assertIn(PlateBoxDisposition.WRONG_CLASS, [box.disposition for box in attempt.raw_boxes])
        self.assertEqual(attempt.accepted_plate_count, 1)
        self.assertEqual(attempt.metadata["vehicle_crop"]["crop_width"], 80)

    def test_raw_result_count_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.jpg"
            _image(path)
            model = FakeModel([{"bbox": [10, 10, 30, 20], "confidence": 0.9, "class_id": 0} for _ in range(5)])
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=directory, model=model)
            processor = PlateDiagnosticProcessor(
                detector_stage=stage,
                plate_config=stage.config,
                diagnostic_config=PlateDiagnosticConfig(maximum_raw_boxes_per_attempt=2, save_annotated_vehicle_crops=False),
                output_dir=directory,
            )
            attempt = processor.process_job(_job(path), attempt_number=1)
        self.assertEqual(attempt.raw_box_count, 2)


if __name__ == "__main__":
    unittest.main()
