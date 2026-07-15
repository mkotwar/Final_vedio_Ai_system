from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_td_case2_search_ready_pipeline as pipeline


class SearchReadyPipelineTests(unittest.TestCase):
    def test_fresh_run_directory_replaces_stale_environment_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            stale_video_path = temp_root / "input.mp4"
            stale_video_path.touch()
            fresh_run_dir = temp_root / "debug_run"
            fresh_run_dir.mkdir()

            downstream_stages = (
                "step03_main",
                "step04a_main",
                "step04b_main",
                "step05_main",
                "step06_main",
                "step07b_main",
                "step08b_main",
                "step09b_main",
                "step10b_main",
            )

            with patch.dict(os.environ, {pipeline.ENV_RUN_DIR: str(stale_video_path)}):
                with patch.object(pipeline, "step01_main", return_value=fresh_run_dir):
                    with patch.multiple(
                        pipeline,
                        **{stage: DEFAULT for stage in downstream_stages},
                    ) as stage_mocks:
                        pipeline.main()

                self.assertEqual(os.environ[pipeline.ENV_RUN_DIR], str(fresh_run_dir.resolve()))
                self.assertTrue(all(mock.call_count == 1 for mock in stage_mocks.values()))

            status = json.loads((fresh_run_dir / "pipeline_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["latest_completed_step"], "step10b")
            self.assertEqual(status["object_search_status"], "ready")
            self.assertIsNone(status["error_message"])

    def test_resume_skips_completed_ingestion_and_yolo_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "03_yolo_detections.json").write_text("{}", encoding="utf-8")
            (run_dir / "03_yolo_object_crops").mkdir()

            remaining_stages = (
                "step04a_main",
                "step04b_main",
                "step05_main",
                "step06_main",
                "step07b_main",
                "step08b_main",
                "step09b_main",
                "step10b_main",
            )

            with patch.object(pipeline, "step01_main") as step01_mock:
                with patch.object(pipeline, "step03_main") as step03_mock:
                    with patch.multiple(
                        pipeline,
                        **{stage: DEFAULT for stage in remaining_stages},
                    ) as stage_mocks:
                        pipeline.main(run_dir)

            step01_mock.assert_not_called()
            step03_mock.assert_not_called()
            self.assertTrue(all(mock.call_count == 1 for mock in stage_mocks.values()))
            self.assertEqual(os.environ[pipeline.ENV_RUN_DIR], str(run_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
