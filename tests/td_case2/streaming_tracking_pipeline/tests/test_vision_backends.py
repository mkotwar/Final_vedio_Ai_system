from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import PlateDetectionCandidate
from tests.td_case2.streaming_tracking_pipeline.config import FlorenceConfig, GeminiConfig, VisionBackendConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob
from tests.td_case2.streaming_tracking_pipeline.florence_inference import FlorenceModelBundle
from tests.td_case2.streaming_tracking_pipeline.vision_backends.factory import create_vision_backend
from tests.td_case2.streaming_tracking_pipeline.vision_backends.gemini_backend import prepare_gemini_image


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


class _SequenceModels:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _SequenceClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _SequenceModels(responses)


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

    def test_gemini_mode_does_not_construct_florence_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            with patch(
                "tests.td_case2.streaming_tracking_pipeline.vision_backends.factory.FlorenceVisionBackend",
                side_effect=AssertionError("Florence should not be constructed in gemini mode"),
            ):
                backend = create_vision_backend(
                    vision_config=VisionBackendConfig(backend_mode="gemini"),
                    florence_config=FlorenceConfig(base_model_path="unused"),
                    gemini_config=GeminiConfig(api_key="test-key"),
                    run_dir=directory,
                    gemini_client_factory=_FakeGeminiClient,
                )
            job = SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 1, 0.1, str(image_path), None, 0.8)
            colour = backend.run_colour(job)
        self.assertEqual(backend.backend_name, "gemini")
        self.assertEqual(colour.metadata.get("vision_backend"), "gemini")

    def test_prepare_gemini_image_preserves_jpeg_and_mime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (96, 64), "white").save(image_path, format="JPEG", quality=85)
            prepared = prepare_gemini_image(image_path)
        self.assertEqual(prepared.mime_type, "image/jpeg")
        self.assertEqual((prepared.width, prepared.height), (96, 64))
        self.assertGreater(prepared.image_bytes, 0)

    def test_timeout_is_converted_to_milliseconds(self) -> None:
        captured: dict[str, Any] = {}

        def _client_factory(*, api_key: str, http_options: Any) -> Any:
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            return _FakeGeminiClient()

        backend = create_vision_backend(
            vision_config=VisionBackendConfig(backend_mode="gemini"),
            florence_config=FlorenceConfig(base_model_path="unused"),
            gemini_config=GeminiConfig(api_key="test-key", timeout_seconds=90),
            run_dir=".",
        )
        with patch("google.genai.Client", side_effect=_client_factory):
            backend.load()
        self.assertEqual(captured["http_options"].timeout, 90000)

    def test_transient_timeout_retries_once_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="gemini"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="test-key", max_retries=1, retry_backoff_seconds=0.0),
                run_dir=directory,
                gemini_client_factory=lambda: _SequenceClient([httpx.ReadTimeout("timed out"), _FakeGeminiResponse({"raw_text": "white", "normalized_colour": "white", "confidence": 0.95, "notes": "ok"})]),
            )
            job = SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 1, 0.1, str(image_path), None, 0.8)
            colour = backend.run_colour(job)
        self.assertEqual(colour.status, "success")
        self.assertEqual(backend.metrics.get("gemini_requests"), 2)
        self.assertEqual(backend.metrics.get("gemini_retries"), 1)

    def test_auth_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            auth_error = RuntimeError("401 invalid api key secret-key-value")
            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="gemini"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="secret-key-value", max_retries=3, retry_backoff_seconds=0.0),
                run_dir=directory,
                gemini_client_factory=lambda: _SequenceClient([auth_error]),
            )
            job = SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 1, 0.1, str(image_path), None, 0.8)
            colour = backend.run_colour(job)
        self.assertEqual(colour.status, "inference_error")
        self.assertEqual(backend.metrics.get("gemini_requests"), 1)
        self.assertEqual(colour.metadata.get("error_message").find("secret-key-value"), -1)

    def test_failed_request_records_latency_and_sanitized_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vehicle.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)

            class _SleepyFailure(Exception):
                pass

            def _raise() -> Any:
                time.sleep(0.01)
                raise _SleepyFailure("request failed for key secret-key-value")

            class _Client:
                def __init__(self) -> None:
                    self.models = MagicMock()
                    self.models.generate_content.side_effect = _raise

            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="gemini"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="secret-key-value", max_retries=0, retry_backoff_seconds=0.0),
                run_dir=directory,
                gemini_client_factory=_Client,
            )
            job = SelectedCropJob("cam", 1, 0, "raw", "car", "done", "primary", 1, 1, 0.1, str(image_path), None, 0.8)
            colour = backend.run_colour(job)
        self.assertEqual(colour.status, "inference_error")
        self.assertGreater(float(colour.metadata.get("elapsed_sec", 0.0) or 0.0), 0.0)
        self.assertGreater(float(backend.metrics.get("gemini_total_latency_sec", 0.0) or 0.0), 0.0)
        self.assertNotIn("secret-key-value", str(colour.metadata))

    def test_response_text_json_is_parsed_when_parsed_field_missing(self) -> None:
        class _TextOnlyResponse:
            def __init__(self) -> None:
                self.text = '{"raw_text":"MH12AB1234","confidence":0.93,"notes":"ok"}'

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "plate.jpg"
            Image.new("RGB", (80, 48), "white").save(image_path)
            backend = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="gemini"),
                florence_config=FlorenceConfig(base_model_path="unused"),
                gemini_config=GeminiConfig(api_key="test-key"),
                run_dir=directory,
                gemini_client_factory=lambda: _SequenceClient([_TextOnlyResponse()]),
            )
            candidate = PlateDetectionCandidate("cam", 1, 0, "primary", 1, 1, str(image_path), 1, 0.9, (1, 1, 10, 8), (0, 0, 11, 9), str(image_path))
            result = backend.run_ocr(candidate)
        self.assertEqual(result.normalized_text, "MH12AB1234")


if __name__ == "__main__":
    unittest.main()
