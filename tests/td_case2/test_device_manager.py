from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from device_manager import DeviceDecision, RuntimeInfo, record_stage_device, resolve_device
from stage_checks import read_json


class DeviceManagerTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self) -> None:
        runtime = RuntimeInfo(
            torch_available=True,
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="GPU",
            cuda_total_vram_mb=1024.0,
            cuda_bf16_supported=True,
            opencv_cuda_available=False,
            opencv_cuda_device_count=0,
            onnxruntime_cuda_available=False,
            tensorrt_available=False,
        )
        with patch("device_manager.get_runtime_info", return_value=runtime):
            with patch.dict("os.environ", {}, clear=False):
                decision = resolve_device(component_name="test_component")
        self.assertIsInstance(decision, DeviceDecision)
        self.assertEqual(decision.selected, "cuda")
        self.assertEqual(decision.torch_device, "cuda:0")

    def test_auto_falls_back_to_cpu_when_cuda_missing(self) -> None:
        runtime = RuntimeInfo(
            torch_available=True,
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name=None,
            cuda_total_vram_mb=None,
            cuda_bf16_supported=False,
            opencv_cuda_available=False,
            opencv_cuda_device_count=0,
            onnxruntime_cuda_available=False,
            tensorrt_available=False,
        )
        with patch("device_manager.get_runtime_info", return_value=runtime):
            with patch.dict("os.environ", {}, clear=False):
                decision = resolve_device(component_name="test_component")
        self.assertEqual(decision.selected, "cpu")
        self.assertEqual(decision.torch_device, "cpu")

    def test_record_stage_device_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            record_stage_device(
                run_dir=run_dir,
                stage_name="stage_a",
                component_name="component_x",
                supports_gpu=True,
                selected_device="cuda:0",
                actual_device="cuda:0",
                reason="test",
            )
            payload = read_json(run_dir / "gpu_utilization_report.json")
        self.assertIn("runtime", payload)
        self.assertIn("stage_a", payload["stages"])


if __name__ == "__main__":
    unittest.main()
