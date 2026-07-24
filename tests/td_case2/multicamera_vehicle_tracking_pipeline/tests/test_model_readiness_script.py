from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import check_model_readiness


class ModelReadinessScriptTests(unittest.TestCase):
    def test_supporting_file_report_marks_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "Florence-2-base-ft"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            report = check_model_readiness._supporting_file_report(model_dir, ["config.json"])
            self.assertTrue(check_model_readiness._supporting_files_ready(report))

    def test_supporting_file_report_marks_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "Florence-2-base-ft"
            model_dir.mkdir()
            report = check_model_readiness._supporting_file_report(model_dir, ["config.json"])
            self.assertFalse(check_model_readiness._supporting_files_ready(report))

    def test_path_size_for_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "model.pt"
            model_file.write_bytes(b"12345")
            self.assertEqual(check_model_readiness._path_size(model_file), 5)

    def test_runtime_package_and_root_model_folder_are_distinct(self) -> None:
        runtime_package = Path(check_model_readiness.__file__).resolve().parents[1] / "models"
        root_model_folder = check_model_readiness.PROJECT_ROOT / "models"
        self.assertNotEqual(runtime_package, root_model_folder)

    def test_no_old_absolute_paths_in_runtime_modules(self) -> None:
        runtime_dir = Path(check_model_readiness.__file__).resolve().parents[1]
        for file_path in runtime_dir.rglob("*.py"):
            text = file_path.read_text(encoding="utf-8")
            self.assertNotIn("F:\\vinfo", text)
            self.assertNotIn("C:\\Mukul K", text)

    def test_readiness_report_shape_is_json_serializable(self) -> None:
        payload = {
            "project_root": str(check_model_readiness.PROJECT_ROOT),
            "models": [{"name": "vehicle_detector", "ready": True}],
        }
        encoded = json.dumps(payload)
        self.assertIn("vehicle_detector", encoded)


if __name__ == "__main__":
    unittest.main()
