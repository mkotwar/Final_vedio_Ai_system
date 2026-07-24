from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_config import (
    GlobalMatchConfigError,
    load_global_match_config,
)


class GlobalMatchConfigTests(unittest.TestCase):
    def test_loads_default_example(self) -> None:
        config = load_global_match_config("tests/td_case2/multicamera_vehicle_tracking_pipeline/config/global_matching.yaml")
        self.assertEqual(config.rule_version, "global_match_v1")
        self.assertTrue(config.create_single_track_global_objects)
        self.assertEqual(config.time_matching.mode, "disabled")
        self.assertIsNotNone(config.route_for("CAM_001", "CAM_002"))

    def test_rejects_invalid_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "global_matching.yaml"
            path.write_text(
                "matching:\n  thresholds:\n    confirmed: 0.50\n    possible: 0.60\n",
                encoding="utf-8",
            )
            with self.assertRaises(GlobalMatchConfigError):
                load_global_match_config(path)


if __name__ == "__main__":
    unittest.main()
