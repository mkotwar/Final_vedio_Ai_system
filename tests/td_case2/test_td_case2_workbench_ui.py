from __future__ import annotations

import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from td_case2_workbench_ui import parse_created_run_dir, parse_extra_env_json


class WorkbenchUiTests(unittest.TestCase):
    def test_parse_created_run_dir_prefers_final_export_line(self) -> None:
        log_text = (
            "Created run directory: C:\\temp\\run_a\n"
            "Search-ready pipeline complete. TD_CASE2_RUN_DIR=C:\\temp\\run_b\n"
        )
        self.assertEqual(parse_created_run_dir(log_text), "C:\\temp\\run_b")

    def test_parse_extra_env_json_returns_string_values(self) -> None:
        payload = parse_extra_env_json('{"TD_CASE2_STEP12_TOP_K": 5, "TD_CASE2_VLM_BACKEND": "local_qwen"}')
        self.assertEqual(payload["TD_CASE2_STEP12_TOP_K"], "5")
        self.assertEqual(payload["TD_CASE2_VLM_BACKEND"], "local_qwen")


if __name__ == "__main__":
    unittest.main()
