from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.td_case2.streaming_tracking_pipeline.run_image_anpr_validation import build_parser, parse_thresholds, run


class RunImageAnprValidationTests(unittest.TestCase):
    def test_cli_defaults_and_threshold_parsing(self) -> None:
        args = build_parser().parse_args([])
        self.assertIn("debug_runs", args.input_dir)
        self.assertEqual(parse_thresholds("0.25,0.1"), (0.25, 0.1))

    def test_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            parse_thresholds("0.2,1.2")
        with self.assertRaises(ValueError):
            parse_thresholds("0.2,0.2")

    def test_invalid_directory(self) -> None:
        args = build_parser().parse_args(["--input-dir", "does-not-exist"])
        with self.assertRaises(FileNotFoundError):
            run(args)

    def test_max_image_limit_with_disabled_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            root.mkdir()
            Image.new("RGB", (20, 20), "white").save(root / "a.jpg")
            Image.new("RGB", (20, 20), "white").save(root / "b.jpg")
            args = build_parser().parse_args(
                [
                    "--input-dir",
                    str(root),
                    "--output-root",
                    str(Path(directory) / "out"),
                    "--max-images",
                    "1",
                    "--plate-detector-model",
                    "missing.pt",
                ]
            )
            result = run(args)
            self.assertEqual(result["summary"]["images_discovered"], 1)
            self.assertTrue(Path(result["run_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
