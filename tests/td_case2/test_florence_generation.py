from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_04a_florence_model_audit import run_florence_generation


class _FakeProcessor:
    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def batch_decode(self, _generated_ids: Any, skip_special_tokens: bool) -> list[str]:
        self.skip_special_tokens = skip_special_tokens
        return ["A white car."]

    def post_process_generation(self, text: str, task: str, image_size: tuple[int, int]) -> str:
        return text


class _FakeModel:
    def generate(self, **kwargs: Any) -> np.ndarray:
        self.generate_kwargs = kwargs
        return np.zeros((1, 2), dtype=np.int64)


class FlorenceGenerationTests(unittest.TestCase):
    def test_generation_disables_incompatible_transformers_cache(self) -> None:
        processor = _FakeProcessor()
        model = _FakeModel()

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "vehicle.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((16, 24, 3), dtype=np.uint8)))

            result = run_florence_generation(
                image_path=image_path,
                processor=processor,
                model=model,
                device_used="cpu",
                task_prompt="<CAPTION>",
                max_new_tokens=16,
                num_beams=1,
            )

        self.assertIs(model.generate_kwargs["use_cache"], False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["raw_decoded_text"], "A white car.")


if __name__ == "__main__":
    unittest.main()
