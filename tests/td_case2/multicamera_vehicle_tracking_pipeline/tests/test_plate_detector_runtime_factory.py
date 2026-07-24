from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.anpr_config import AnprConfig, PlateDetectorConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.plate_detector_runtime_factory import (
    DEFAULT_PLATE_DETECTOR_MODEL_PATH,
    PlateDetectorRuntimeFactory,
)


class _FakeRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.loaded = False

    def load(self) -> None:
        self.loaded = True


class PlateDetectorRuntimeFactoryTests(unittest.TestCase):
    def test_runtime_uses_explicit_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "license_plate_weights.pt"
            model_path.write_bytes(b"plate")
            factory = PlateDetectorRuntimeFactory(project_root=root)
            original_runtime_cls = factory.get_runtime.__globals__["PlateDetectorRuntime"]
            try:
                factory.get_runtime.__globals__["PlateDetectorRuntime"] = _FakeRuntime
                runtime = factory.get_runtime(
                    config=AnprConfig(enabled=True, plate_detector=PlateDetectorConfig(model_path=str(model_path))),
                    device_override="cpu",
                )
            finally:
                factory.get_runtime.__globals__["PlateDetectorRuntime"] = original_runtime_cls
            self.assertIsNotNone(runtime)
            self.assertTrue(runtime.loaded)
            self.assertEqual(runtime.kwargs["model_path"], model_path.resolve())
            self.assertEqual(runtime.kwargs["device"], "cpu")

    def test_default_path_points_to_root_model_assets(self) -> None:
        self.assertEqual(tuple(DEFAULT_PLATE_DETECTOR_MODEL_PATH.parts)[-3:], ("models", "plate_detection", "license_plate_weights.pt"))


if __name__ == "__main__":
    unittest.main()
