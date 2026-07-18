from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import FlorenceOcrResult
from tests.td_case2.streaming_tracking_pipeline.config import PlateDetectionConfig
from tests.td_case2.streaming_tracking_pipeline.image_anpr_validation import (
    ImageAnprValidationConfig,
    ImageAnprValidator,
    discover_image_inputs,
)
from tests.td_case2.streaming_tracking_pipeline.plate_detection import UltralyticsPlateDetectionStage
from tests.td_case2.streaming_tracking_pipeline.serialization import to_json_safe


class FakeImagePlateModel:
    names = {0: "License_Plate", 1: "vehicle"}

    def predict(self, source: str, **kwargs: Any) -> dict[str, Any]:
        name = Path(source).name
        if "none" in name:
            return {"boxes": []}
        if "reject" in name:
            return {"boxes": [{"bbox": [10, 10, 12, 12], "confidence": 0.9, "class_id": 0}]}
        return {"boxes": [{"bbox": [10, 10, 50, 26], "confidence": 0.9, "class_id": 0}]}


class FakeOcr:
    def run_ocr(self, candidate: Any) -> FlorenceOcrResult:
        text = "MH12AB1234"
        return FlorenceOcrResult(
            source_id=candidate.source_id,
            track_id=candidate.track_id,
            track_generation=candidate.track_generation,
            crop_role=candidate.crop_role,
            crop_rank=candidate.crop_rank,
            frame_index=candidate.frame_index,
            plate_rank=candidate.plate_rank,
            plate_crop_path=candidate.plate_crop_path,
            raw_text=text,
            normalized_text=text,
            status="success",
            prompt="<OCR>",
        )


class ImageAnprValidationTests(unittest.TestCase):
    def test_discovery_order_extensions_and_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20)).save(root / "b.png")
            Image.new("RGB", (20, 20)).save(root / "a.jpg")
            (root / "c.txt").write_text("skip", encoding="utf-8")
            (root / "d.jpg").write_text("not image", encoding="utf-8")
            records = discover_image_inputs(root)
        self.assertEqual([record.filename for record in records], ["a.jpg", "b.png", "d.jpg"])
        self.assertFalse(records[-1].readable)

    def test_validation_accepts_rejects_and_runs_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            root.mkdir()
            Image.new("RGB", (100, 60), "white").save(root / "plate.jpg")
            Image.new("RGB", (100, 60), "white").save(root / "reject.jpg")
            Image.new("RGB", (100, 60), "white").save(root / "none.jpg")
            run_dir = Path(directory) / "out"
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(confidence_threshold=0.25), output_dir=run_dir, model=FakeImagePlateModel())
            validator = ImageAnprValidator(
                config=ImageAnprValidationConfig(input_dir=str(root), run_florence_ocr=True, direct_ocr_on_input=True, save_annotations=True),
                plate_detector=stage,
                florence_engine=FakeOcr(),
                output_dir=run_dir,
            )
            payload = validator.run()
            summary = payload["summary"]
            self.assertEqual(summary["images_discovered"], 3)
            self.assertEqual(summary["images_with_accepted_plates"], 1)
            self.assertEqual(summary["images_with_all_boxes_rejected"], 1)
            self.assertEqual(summary["ocr_non_empty_outputs"], 1)
            self.assertEqual(summary["direct_input_ocr_non_empty_outputs"], 3)
            json.dumps(to_json_safe(payload["results"][0]))
            self.assertTrue((run_dir / "image_results.jsonl").exists())

    def test_no_ocr_when_no_accepted_plate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            root.mkdir()
            Image.new("RGB", (100, 60), "white").save(root / "none.jpg")
            run_dir = Path(directory) / "out"
            stage = UltralyticsPlateDetectionStage(PlateDetectionConfig(), output_dir=run_dir, model=FakeImagePlateModel())
            validator = ImageAnprValidator(
                config=ImageAnprValidationConfig(input_dir=str(root), run_florence_ocr=True),
                plate_detector=stage,
                florence_engine=FakeOcr(),
                output_dir=run_dir,
            )
            payload = validator.run()
        self.assertEqual(payload["summary"]["ocr_calls"], 0)


if __name__ == "__main__":
    unittest.main()
