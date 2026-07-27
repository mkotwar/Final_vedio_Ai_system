from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import recalculate_track_classes


class RecalculateTrackClassesTests(unittest.TestCase):
    def test_generate_report_marks_artifact_only_run_as_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            report_dir = root / "debug_runs" / "multicamera_vehicle_tracking_pipeline" / "sample_run"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_dir.joinpath("report.json").write_text(
                json.dumps(
                    {
                        "run_id": "RUN_SAMPLE",
                        "completed_tracks": [
                            {
                                "track_uuid": "RUN_SAMPLE:CAM_001:TRACK_2",
                                "camera_code": "CAM_001",
                                "local_track_id": 2,
                                "class_name": "bus",
                                "canonical_class_name": "BUS",
                                "observation_count": 17,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(recalculate_track_classes, "AnalyticsDatabaseClient", side_effect=RuntimeError("db unavailable")):
                previous = Path.cwd()
                try:
                    import os

                    os.chdir(root)
                    report = recalculate_track_classes.generate_report(
                        run_code="RUN_SAMPLE",
                        camera_code="CAM_001",
                        track_uuid=None,
                        tracking_config_path="C:/Mukul K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/config/tracking.yaml",
                        persist=False,
                    )
                finally:
                    os.chdir(previous)
            self.assertEqual(report["source"], "artifact_report_only")
            self.assertTrue(report["tracks"][0]["insufficient_history"])

    def test_print_report_includes_old_and_new_class(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            recalculate_track_classes.print_report(
                {
                    "run_code": "RUN_SAMPLE",
                    "camera_code": "CAM_001",
                    "track_uuid": None,
                    "source": "analytics",
                    "tracks": [
                        {
                            "track_uuid": "RUN_SAMPLE:CAM_001:TRACK_2",
                            "old_final_class": "BUS",
                            "new_final_class": "3wheeler",
                            "class_counts": {"3wheeler": 4, "bus": 1},
                            "class_confidence_sums": {"3wheeler": 2.91, "bus": 0.87},
                            "latest_observation_class": "bus",
                            "persisted": False,
                        }
                    ],
                    "fragment_candidates": [],
                    "notes": [],
                }
            )
        rendered = output.getvalue()
        self.assertIn("old_final_class: BUS", rendered)
        self.assertIn("new_final_class: 3wheeler", rendered)


if __name__ == "__main__":
    unittest.main()
