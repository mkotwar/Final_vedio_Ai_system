from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.report_tracking_configuration import generate_report


class _Args:
    def __init__(self, *, root: Path) -> None:
        base = root / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / "config"
        self.camera_config = None
        self.detection_config = str(base / "detection.yaml")
        self.tracking_config = str(base / "tracking.yaml")
        self.worker_config = str(base / "workers.yaml")
        self.persistence_config = str(base / "persistence.yaml")
        self.evidence_config = str(base / "evidence.yaml")
        self.anpr_config = None
        self.camera_code = None
        self.camera_codes = None
        self.camera_limit = None
        self.persist_to_supabase = False
        self.dry_run_persistence = True
        self.json_output = None


class ReportTrackingConfigurationTests(unittest.TestCase):
    def test_generate_report_merges_effective_runtime_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config_dir = root / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "detection.yaml").write_text(
                json.dumps(
                    {
                        "vehicle_detector": {
                            "model_path": "yolov8n.pt",
                            "allow_fallback": True,
                            "device": "cpu",
                            "confidence_threshold": 0.33,
                            "iou_threshold": 0.44,
                            "image_size": 640,
                            "allowed_classes": ["car", "truck"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "tracking.yaml").write_text(
                json.dumps(
                    {
                        "tracking": {
                            "backend": "supervision_bytetrack",
                            "track_activation_threshold": 0.21,
                            "lost_track_buffer": 12,
                            "minimum_matching_threshold": 0.77,
                            "frame_rate": 20,
                            "min_confirmed_observations": 3,
                            "max_lost_frames": 9,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "workers.yaml").write_text(
                json.dumps({"workers": {"enabled": False, "frame_queue_size": 8, "enable_persistence_worker": False}}),
                encoding="utf-8",
            )
            (config_dir / "persistence.yaml").write_text(
                json.dumps({"persistence": {"backend": "disabled", "enabled": False, "observation_mode": "all"}}),
                encoding="utf-8",
            )
            (config_dir / "evidence.yaml").write_text(
                json.dumps({"evidence": {"enabled": True, "output_root": "artifacts"}}),
                encoding="utf-8",
            )

            report = generate_report(_Args(root=root))

        self.assertEqual(report["effective_runtime"]["tracking"]["track_activation_threshold"], 0.21)
        self.assertEqual(report["effective_runtime"]["worker"]["enabled"], True)
        self.assertEqual(report["effective_runtime"]["worker"]["enable_persistence_worker"], True)
        self.assertEqual(report["effective_runtime"]["persistence"]["backend"], "dry_run")
        self.assertEqual(report["precedence"]["tracking"], "environment override -> YAML -> code default")


if __name__ == "__main__":
    unittest.main()
