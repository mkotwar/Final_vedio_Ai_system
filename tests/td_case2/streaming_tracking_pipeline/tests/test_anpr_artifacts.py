from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.anpr_artifacts import AnprArtifactSink, read_selected_crop_jobs
from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import TrackAnprColourResult
from tests.td_case2.streaming_tracking_pipeline.crop_selection import SelectedCropJob


class AnprArtifactTests(unittest.TestCase):
    def test_writes_track_and_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = SelectedCropJob("cam", 1, 0, None, "car", "done", "primary", 1, 1, 0.1, "crop.jpg", None, 0.8)
            result = TrackAnprColourResult("cam", 1, 0, None, "car", "done", "no_plate_candidates", selected_crop_jobs=[job])
            sink = AnprArtifactSink(directory)
            sink.write_result(result)
            sink.write_summary({"tracks_processed": 1})
            sink.close()
            self.assertTrue((Path(directory) / "07_anpr" / "track_anpr_colour_results.jsonl").exists())
            self.assertTrue((Path(directory) / "reports" / "step7_anpr_colour_report.json").exists())
            self.assertEqual(len(read_selected_crop_jobs(Path(directory) / "07_anpr" / "step7_selected_crop_jobs.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
