from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import PlateDetectionCandidate
from tests.td_case2.streaming_tracking_pipeline.config import FlorenceConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.florence_inference import FlorenceInferenceEngine, FlorenceModelBundle


class FakeProcessor:
    def __call__(self, text: str, images: Any, return_tensors: str) -> dict[str, Any]:
        return {"prompt": text}

    def batch_decode(self, generated_ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [generated_ids[0]]

    def post_process_generation(self, generated_text: str, task: str, image_size: tuple[int, int]) -> dict[str, str]:
        return {task: "gray" if "VQA" in task else "DL01AB1234"}


class FakeModel:
    def generate(self, **kwargs: Any) -> list[str]:
        return ["raw"]


def _bundle() -> FlorenceModelBundle:
    return FlorenceModelBundle(model=FakeModel(), processor=FakeProcessor(), device="cpu")


class FlorenceInferenceTests(unittest.TestCase):
    def test_ocr_and_colour_use_shared_fake_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.jpg"
            Image.new("RGB", (32, 24), "white").save(image)
            engine = FlorenceInferenceEngine(FlorenceConfig(base_model_path="unused"), bundle=_bundle())
            candidate = PlateDetectionCandidate(
                source_id="cam",
                track_id=1,
                track_generation=0,
                crop_role="primary",
                crop_rank=1,
                frame_index=1,
                vehicle_crop_path=str(image),
                plate_rank=1,
                confidence=0.9,
                bbox_xyxy=(1, 1, 10, 8),
                padded_bbox_xyxy=(0, 0, 11, 9),
                plate_crop_path=str(image),
            )
            job = SelectedCropJob("cam", 1, 0, None, "car", "done", "primary", 1, 1, 0.25, str(image), None, 0.8)
            ocr = engine.run_ocr(candidate)
            colour = engine.run_colour(job)
        self.assertEqual(ocr.normalized_text, "DL01AB1234")
        self.assertEqual(colour.normalized_colour, "gray")

    def test_disabled_engine_returns_model_disabled(self) -> None:
        engine = FlorenceInferenceEngine(FlorenceConfig(enabled=False))
        candidate = PlateDetectionCandidate("cam", 1, 0, "primary", 1, 1, "x.jpg", 1, 0.9, (1, 1, 10, 8), (0, 0, 11, 9), "x.jpg")
        self.assertEqual(engine.run_ocr(candidate).status, "model_disabled")


if __name__ == "__main__":
    unittest.main()
