from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_config import FlorenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.florence_runtime_factory import FlorenceRuntimeFactory


class _FakeRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.loaded = False

    def load(self) -> None:
        self.loaded = True


class FlorenceRuntimeFactoryTests(unittest.TestCase):
    def test_auto_device_resolves_to_cpu_or_cuda(self) -> None:
        resolved = FlorenceRuntimeFactory._resolve_device("auto")
        self.assertIn(resolved, {"cpu", "cuda"})

    def test_override_device_is_normalized(self) -> None:
        resolved = FlorenceRuntimeFactory._resolve_device("CPU")
        self.assertEqual(resolved, "cpu")

    def test_runtime_uses_resolved_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "florence"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            factory = FlorenceRuntimeFactory(project_root=root)
            original_runtime_cls = factory.get_runtime.__globals__["FlorenceRuntime"]
            try:
                factory.get_runtime.__globals__["FlorenceRuntime"] = _FakeRuntime
                runtime = factory.get_runtime(
                    config=FlorenceConfig(
                        enabled=True,
                        model_path=str(model_dir),
                        adapter_path=str(adapter_dir),
                        device="auto",
                    )
                )
            finally:
                factory.get_runtime.__globals__["FlorenceRuntime"] = original_runtime_cls
            self.assertIsNotNone(runtime)
            self.assertTrue(runtime.loaded)
            self.assertIn(runtime.kwargs["device"], {"cpu", "cuda"})


if __name__ == "__main__":
    unittest.main()
