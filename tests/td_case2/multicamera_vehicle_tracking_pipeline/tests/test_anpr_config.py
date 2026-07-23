from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.anpr_config import AnprConfigError, load_anpr_config


class AnprConfigTests(unittest.TestCase):
    def test_loads_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anpr.yaml"
            path.write_text(
                "anpr:\n"
                "  enabled: true\n"
                "  vehicle_evidence_roles: [BEST_OVERALL, SHARPEST]\n"
                "  plate_detector:\n"
                "    confidence_threshold: 0.25\n",
                encoding="utf-8",
            )
            config = load_anpr_config(path)
            self.assertTrue(config.enabled)
            self.assertEqual(config.vehicle_evidence_roles, ("BEST_OVERALL", "SHARPEST"))

    def test_rejects_invalid_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anpr.yaml"
            path.write_text("anpr:\n  vehicle_evidence_roles: [BAD_ROLE]\n", encoding="utf-8")
            with self.assertRaises(AnprConfigError):
                load_anpr_config(path)


if __name__ == "__main__":
    unittest.main()
