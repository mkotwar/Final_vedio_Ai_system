from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_config import EvidenceConfig, EvidenceConfigError, load_evidence_config


class EvidenceConfigTests(unittest.TestCase):
    def test_defaults_are_disabled(self) -> None:
        self.assertFalse(EvidenceConfig().enabled)

    def test_invalid_quality_is_rejected(self) -> None:
        with self.assertRaises(EvidenceConfigError):
            EvidenceConfig(jpeg_quality=0)

    def test_yaml_loads_expected_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.yaml"
            path.write_text(
                "evidence:\n"
                "  enabled: true\n"
                "  output_root: artifacts\n"
                "  minimum_crop_width: 20\n",
                encoding="utf-8",
            )
            config = load_evidence_config(path)
            self.assertTrue(config.enabled)
            self.assertEqual(config.output_root, "artifacts")
            self.assertEqual(config.minimum_crop_width, 20)


if __name__ == "__main__":
    unittest.main()
