from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.td_case2.streaming_tracking_pipeline.config import PipelineConfig, QueueConfig, TrackingConfig


STREAM_ENV_KEYS = [key for key in os.environ if key.startswith("TD_CASE2_STREAM_")]


class ConfigTests(unittest.TestCase):
    def test_default_config_creation(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.source.source_id, "default_source")
        self.assertEqual(config.tracking.backend, "ultralytics_bytetrack")
        self.assertTrue(config.florence.local_files_only)

    def test_invalid_confidence_threshold(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig().with_overrides({"detection": {"confidence_threshold": 1.5}})

    def test_invalid_queue_size(self) -> None:
        with self.assertRaises(ValueError):
            QueueConfig(frame_queue_size=0)

    def test_unsupported_tracking_backend(self) -> None:
        with self.assertRaises(ValueError):
            TrackingConfig(backend="unknown")

    def test_environment_integer_float_and_boolean_parsing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TD_CASE2_STREAM_TRACKING_BUFFER": "44",
                "TD_CASE2_STREAM_DETECTION_CONFIDENCE": "0.55",
                "TD_CASE2_STREAM_PLATE_ENABLED": "false",
            },
            clear=False,
        ):
            for key in STREAM_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["TD_CASE2_STREAM_TRACKING_BUFFER"] = "44"
            os.environ["TD_CASE2_STREAM_DETECTION_CONFIDENCE"] = "0.55"
            os.environ["TD_CASE2_STREAM_PLATE_ENABLED"] = "false"
            config = PipelineConfig.from_env()
        self.assertEqual(config.tracking.lost_track_buffer, 44)
        self.assertEqual(config.detection.confidence_threshold, 0.55)
        self.assertFalse(config.plate_detection.enabled)

    def test_invalid_boolean_parsing(self) -> None:
        with patch.dict(os.environ, {"TD_CASE2_STREAM_PLATE_ENABLED": "sometimes"}, clear=True):
            with self.assertRaises(ValueError):
                PipelineConfig.from_env()

    def test_json_config_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"source": {"source_id": "json_cam"}, "queue": {"frame_queue_size": 3}}),
                encoding="utf-8",
            )
            config = PipelineConfig.from_json(path)
        self.assertEqual(config.source.source_id, "json_cam")
        self.assertEqual(config.queue.frame_queue_size, 3)

    def test_environment_overrides_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"source": {"source_id": "json_cam"}}), encoding="utf-8")
            with patch.dict(os.environ, {"TD_CASE2_STREAM_SOURCE_ID": "env_cam"}, clear=True):
                config = PipelineConfig.from_env(path)
        self.assertEqual(config.source.source_id, "env_cam")

    def test_to_dict_json_safety(self) -> None:
        payload = PipelineConfig().to_dict()
        self.assertIsInstance(payload["detection"]["allowed_class_names"], list)
        json.dumps(payload)

    def test_google_api_key_is_used_when_gemini_api_key_is_absent(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-only-key"}, clear=True):
            config = PipelineConfig.from_env()
        self.assertEqual(config.gemini.api_key, "google-only-key")

    def test_gemini_api_key_takes_priority_over_google_api_key(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"}, clear=True):
            config = PipelineConfig.from_env()
        self.assertEqual(config.gemini.api_key, "gemini-key")

    def test_new_gemini_timeout_and_backoff_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TD_CASE2_GEMINI_TIMEOUT_SEC": "90",
                "TD_CASE2_GEMINI_MAX_RETRIES": "1",
                "TD_CASE2_GEMINI_RETRY_BACKOFF_SEC": "2.5",
            },
            clear=True,
        ):
            config = PipelineConfig.from_env()
        self.assertEqual(config.gemini.timeout_seconds, 90)
        self.assertEqual(config.gemini.max_retries, 1)
        self.assertEqual(config.gemini.retry_backoff_seconds, 2.5)


if __name__ == "__main__":
    unittest.main()
