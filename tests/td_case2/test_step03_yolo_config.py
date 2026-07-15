from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import case_root, repo_root, resolve_case_path
from run_td_case2_step03_yolo import _resolve_model_path


class Step03YoloConfigTests(unittest.TestCase):
    def test_relative_model_path_is_resolved_from_td_case2_directory(self) -> None:
        resolved_path = _resolve_model_path("../../yolo11m.pt", None)

        self.assertEqual(resolved_path, (case_root() / "../../yolo11m.pt").resolve())

    def test_absolute_model_path_is_preserved(self) -> None:
        absolute_path = (case_root() / "model.pt").resolve()

        self.assertEqual(_resolve_model_path(str(absolute_path), None), absolute_path)

    def test_existing_working_directory_path_takes_precedence(self) -> None:
        self.assertEqual(resolve_case_path("yolo11m.pt"), (repo_root() / "yolo11m.pt").resolve())


if __name__ == "__main__":
    unittest.main()
