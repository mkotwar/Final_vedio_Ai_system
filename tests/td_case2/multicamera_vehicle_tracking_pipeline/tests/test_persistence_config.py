from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig, PersistenceConfigError, load_persistence_config


class PersistenceConfigTests(unittest.TestCase):
    def test_default_persistence_is_disabled(self) -> None:
        self.assertFalse(PersistenceConfig().enabled)

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaises(PersistenceConfigError):
            PersistenceConfig(observation_batch_size=0)

    def test_invalid_observation_mode_is_rejected(self) -> None:
        with self.assertRaises(PersistenceConfigError):
            PersistenceConfig(observation_mode="weird")

    def test_invalid_sample_interval_is_rejected(self) -> None:
        with self.assertRaises(PersistenceConfigError):
            PersistenceConfig(observation_sample_every_n=0)

    def test_yaml_loads_expected_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "persistence.yaml"
            path.write_text(
                "persistence:\n"
                "  enabled: true\n"
                "  observation_mode: sampled\n"
                "  observation_batch_size: 25\n"
                "  observation_sample_every_n: 3\n",
                encoding="utf-8",
            )
            config = load_persistence_config(path)
            self.assertTrue(config.enabled)
            self.assertEqual(config.observation_mode, "sampled")
            self.assertEqual(config.observation_batch_size, 25)
            self.assertEqual(config.observation_sample_every_n, 3)


if __name__ == "__main__":
    unittest.main()
