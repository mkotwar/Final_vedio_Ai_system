from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.plate_diagnostic_artifacts import PlateDiagnosticArtifactSink
from tests.td_case2.streaming_tracking_pipeline.plate_diagnostics import TrackPlateDiagnosticResult


class PlateDiagnosticArtifactTests(unittest.TestCase):
    def test_jsonl_and_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = TrackPlateDiagnosticResult(
                source_id="cam",
                track_id=1,
                track_generation=0,
                source_track_id=None,
                object_class="car",
                attempts=[],
                selected_attempt_number=None,
                selected_plate_candidate=None,
                selected_ocr_result=None,
                final_status="no_plate_candidate",
                final_failure_reasons=["no_raw_detector_boxes"],
                exhausted_selected_crops=True,
            )
            sink = PlateDiagnosticArtifactSink(directory)
            sink.write_result(result)
            sink.write_summary({"tracks_processed": 1})
            sink.close()
            sink.close()
            self.assertTrue((Path(directory) / "07_5_plate_diagnostics" / "track_plate_diagnostic_results.jsonl").exists())
            self.assertTrue((Path(directory) / "reports" / "step75_plate_diagnostic_report.json").exists())


if __name__ == "__main__":
    unittest.main()
