from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import qwen_4bit
from step_11_5_lightweight_vlm_filter import run_lightweight_vlm_filter
from step_14_vlm_event_review import run_vlm_event_review
from stage_checks import write_json


class TestQwen4Bit(unittest.TestCase):
    def _make_model_dir(self, temp_dir: Path, *, quantization_config: dict | None) -> Path:
        model_dir = temp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "quantization_config": quantization_config,
        }
        (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        return model_dir

    def test_normal_checkpoint_builds_runtime_nf4_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            model_dir = self._make_model_dir(temp_dir, quantization_config=None)
            with patch.object(qwen_4bit, "_verify_processor_offline", return_value={"processor_class": "MockProcessor", "processor_offline_ready": True}):
                with patch.dict(os.environ, {}, clear=False):
                    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=True):
                        load_config = qwen_4bit.build_qwen_4bit_load_config(model_dir, torch_module=torch)
        self.assertEqual(load_config["checkpoint_type"], "normal_runtime_quantized")
        self.assertIsNotNone(load_config["quantization_config"])
        quantization = load_config["quantization_config"]
        self.assertTrue(quantization.load_in_4bit)
        self.assertEqual(quantization.bnb_4bit_quant_type, "nf4")
        self.assertTrue(quantization.bnb_4bit_use_double_quant)
        self.assertIs(load_config["compute_dtype"], torch.bfloat16)
        self.assertEqual(load_config["precision_label"], "4bit_nf4_runtime_bfloat16")

    def test_compute_dtype_falls_back_to_float16_when_bf16_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            model_dir = self._make_model_dir(temp_dir, quantization_config=None)
            with patch.object(qwen_4bit, "_verify_processor_offline", return_value={"processor_class": "MockProcessor", "processor_offline_ready": True}):
                with patch.dict(os.environ, {qwen_4bit.ENV_QWEN_4BIT_COMPUTE_DTYPE: "auto"}, clear=False):
                    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=False):
                        load_config = qwen_4bit.build_qwen_4bit_load_config(model_dir, torch_module=torch)
        self.assertIs(load_config["compute_dtype"], torch.float16)
        self.assertEqual(load_config["compute_dtype_name"], "float16")

    def test_prequantized_checkpoint_is_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            model_dir = self._make_model_dir(
                temp_dir,
                quantization_config={
                    "load_in_4bit": True,
                    "quant_method": "bitsandbytes",
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                },
            )
            with patch.object(qwen_4bit, "_verify_processor_offline", return_value={"processor_class": "MockProcessor", "processor_offline_ready": True}):
                with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=True):
                    load_config = qwen_4bit.build_qwen_4bit_load_config(model_dir, torch_module=torch)
        self.assertEqual(load_config["checkpoint_type"], "prequantized_nf4")
        self.assertIsNone(load_config["quantization_config"])

    def test_missing_model_directory_raises_clear_error(self) -> None:
        missing = Path("Z:/definitely_missing_qwen_dir")
        with self.assertRaises(FileNotFoundError) as exc:
            qwen_4bit.build_qwen_4bit_load_config(missing, torch_module=torch)
        self.assertIn("directory not found", str(exc.exception).lower())

    def test_missing_config_json_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            model_dir = Path(temp_dir_str) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(FileNotFoundError) as exc:
                qwen_4bit.build_qwen_4bit_load_config(model_dir, torch_module=torch)
        self.assertIn("config", str(exc.exception).lower())

    def test_full_precision_silent_fallback_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            model_dir = self._make_model_dir(temp_dir, quantization_config=None)
            with patch.object(qwen_4bit, "_verify_processor_offline", return_value={"processor_class": "MockProcessor", "processor_offline_ready": True}):
                with patch.dict(os.environ, {qwen_4bit.ENV_QWEN_LOAD_IN_4BIT: "0"}, clear=False):
                    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=True):
                        with self.assertRaises(RuntimeError) as exc:
                            qwen_4bit.build_qwen_4bit_load_config(model_dir, torch_module=torch)
        self.assertIn("full-precision fallback is disabled", str(exc.exception).lower())

    def test_disabled_backend_skips_local_qwen_loading_in_step11_5(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            run_dir = Path(temp_dir_str)
            write_json(run_dir / "11_full_scene_event_candidates.json", {"candidate_events": []})
            result, report, flat = run_lightweight_vlm_filter(
                run_dir=run_dir,
                filter_config={
                    "vlm_backend": "disabled",
                    "api_provider": "openrouter",
                    "api_model": "unused",
                    "model_path": str(run_dir / "unused"),
                    "max_candidates_to_check": 5,
                    "min_filtered_events": 1,
                    "max_filtered_events": 3,
                    "allow_uncertain_backfill": True,
                    "allow_normal_context_backfill": True,
                    "max_new_tokens": 32,
                    "use_cache": True,
                    "device": "cpu",
                },
            )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(report["vlm_backend"], "disabled")
        self.assertEqual(flat, [])

    def test_api_backend_skips_local_qwen_loader_in_step11_5(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            run_dir = Path(temp_dir_str)
            frame_dir = run_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / "frame_000001.jpg").write_bytes(b"fake")
            write_json(
                run_dir / "11_full_scene_event_candidates.json",
                {
                    "candidate_events": [
                        {
                            "candidate_event_id": "evt_1",
                            "event_type": "possible_collision_or_near_miss",
                            "candidate_score": 0.9,
                            "best_timestamp_seconds": 1.0,
                            "best_timestamp_text": "00:01",
                            "representative_frame": {"image_path": "frames/frame_000001.jpg"},
                            "full_frame_paths": ["frames/frame_000001.jpg"],
                        }
                    ]
                },
            )
            with patch("step_11_5_lightweight_vlm_filter.call_qwen_api_with_image", return_value={"status": "success", "assistant_text": "{\"decision\":\"yes\",\"event_likelihood\":0.91,\"visible_event_type\":\"collision\",\"short_reason\":\"Visible collision.\",\"should_keep\":true}", "latency_seconds": 0.01, "provider": "openrouter", "model": "qwen/qwen3-vl-8b-instruct", "request_metadata": {}, "raw_response_text": "", "error_message": None}):
                result, report, flat = run_lightweight_vlm_filter(
                    run_dir=run_dir,
                    filter_config={
                        "vlm_backend": "api_qwen",
                        "api_provider": "openrouter",
                        "api_model": "qwen/qwen3-vl-8b-instruct",
                        "model_path": str(run_dir / "unused"),
                        "max_candidates_to_check": 5,
                        "min_filtered_events": 1,
                        "max_filtered_events": 3,
                        "allow_uncertain_backfill": True,
                        "allow_normal_context_backfill": True,
                        "max_new_tokens": 32,
                        "use_cache": False,
                        "device": "cpu",
                    },
                )
        self.assertEqual(result["status"], "success")
        self.assertEqual(report["vlm_backend"], "api_qwen")
        self.assertEqual(len(flat), 1)

    def test_disabled_backend_skips_local_qwen_loading_in_step14(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            run_dir = Path(temp_dir_str)
            write_json(run_dir / "13_vlm_event_inputs.json", {"vlm_inputs": []})
            write_json(run_dir / "13_vlm_event_input_report.json", {"inputs_ready_for_vlm": 0})
            write_json(run_dir / "12_selected_top_event_candidates.json", {"selected_count": 0})
            write_json(run_dir / "01_video_info.json", {"video_name": "demo.mp4", "duration_text": "00:10"})
            output_payload, flat_reviews, final_summary, report_payload = run_vlm_event_review(
                run_dir=run_dir,
                review_config={
                    "vlm_backend": "disabled",
                    "api_provider": "openrouter",
                    "api_model": "unused",
                    "model_path": str(run_dir / "unused"),
                    "max_inputs": 5,
                    "max_new_tokens": 64,
                    "use_cache": True,
                    "device": "cpu",
                    "require_strip": False,
                },
            )
        self.assertEqual(output_payload["status"], "skipped")
        self.assertEqual(report_payload["vlm_backend"], "disabled")
        self.assertEqual(flat_reviews, [])
        self.assertEqual(final_summary["overall_status"], "vlm_skipped")


if __name__ == "__main__":
    unittest.main()
