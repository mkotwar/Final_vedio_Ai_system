from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import PlateDetectionCandidate
from tests.td_case2.streaming_tracking_pipeline.config import FlorenceConfig, GeminiConfig, VisionBackendConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.florence_inference import FlorenceModelBundle
from tests.td_case2.streaming_tracking_pipeline.vision_backends.factory import create_vision_backend


class _FakeProcessor:
    def __call__(self, text: str, images: Any, return_tensors: str) -> dict[str, Any]:
        return {"prompt": text}

    def batch_decode(self, generated_ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [generated_ids[0]]

    def post_process_generation(self, generated_text: str, task: str, image_size: tuple[int, int]) -> dict[str, str]:
        return {task: "white" if "VQA" in task else "MH12AB1234"}


class _FakeModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate(self, **kwargs: Any) -> list[str]:
        if self.fail:
            raise RuntimeError("forced failure")
        prompt = str(kwargs.get("prompt", ""))
        return ["white" if "VQA" in prompt else "MH12AB1234"]


class _FakeGeminiResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.parsed = payload


class _FakeGeminiModels:
    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> _FakeGeminiResponse:
        prompt = str(contents[-1]) if isinstance(contents, list) else ""
        if "normalized_colour" in prompt:
            return _FakeGeminiResponse({"raw_text": "white", "normalized_colour": "white", "confidence": 0.91, "notes": "ok"})
        return _FakeGeminiResponse({"raw_text": "MH12AB1234", "confidence": 0.93, "notes": "ok"})


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


class VisionBackendTests(unittest.TestCase):
    def test_auto_falls_back_to_gemini_when_florence_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="auto"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="test-key"),
                run_dir=directory,
                florence_bundle=FlorenceModelBundle(model=_FakeModel(fail=True), processor=_FakeProcessor(), device="cpu"),
                gemini_client_factory=_FakeGeminiClient,
            )
            job = SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 1, 0.1, str(image_path), None, 0.8)
            colour = backend.run_colour(job)
        self.assertEqual(colour.status, "success")
        self.assertEqual(colour.metadata.get("vision_backend"), "gemini")
        self.assertEqual(colour.metadata.get("vision_fallback_from"), "florence")

    def test_gemini_ocr_uses_cached_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "plate.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="gemini"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="test-key"),
                run_dir=directory,
                gemini_client_factory=_FakeGeminiClient,
            )
            candidate = PlateDetectionCandidate("cam", 1, 0, "primary", 1, 1, str(image_path), 1, 0.9, (1, 1, 10, 8), (0, 0, 11, 9), str(image_path))
            first = backend.run_ocr(candidate)
            second = backend.run_ocr(candidate)
        self.assertEqual(first.normalized_text, "MH12AB1234")
        self.assertEqual(second.metadata.get("cache_hit"), True)
        self.assertGreaterEqual(backend.metrics.get("gemini_cache_hits", 0), 1)


if __name__ == "__main__":
    unittest.main()
