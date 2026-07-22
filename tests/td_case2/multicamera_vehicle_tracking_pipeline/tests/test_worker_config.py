from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig, WorkerConfigError, load_worker_config


class WorkerConfigTests(unittest.TestCase):
    def test_default_disabled(self) -> None:
        self.assertFalse(WorkerConfig().enabled)

    def test_invalid_queue_size_rejected(self) -> None:
        with self.assertRaises(WorkerConfigError):
            WorkerConfig(frame_queue_size=0)

    def test_invalid_timeout_rejected(self) -> None:
        with self.assertRaises(WorkerConfigError):
            WorkerConfig(queue_put_timeout_seconds=0.0)

    def test_yaml_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workers.yaml"
            path.write_text("workers:\n  enabled: true\n  frame_queue_size: 5\n", encoding="utf-8")
            config = load_worker_config(path)
            self.assertTrue(config.enabled)
            self.assertEqual(config.frame_queue_size, 5)


if __name__ == "__main__":
    unittest.main()
