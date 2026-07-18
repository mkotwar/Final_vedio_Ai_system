from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.plate_validation_artifacts import (
    PlateValidationArtifactSink,
    validate_required_step8_inputs,
)
from tests.td_case2.streaming_tracking_pipeline.plate_validation_schemas import FinalTrackAnprResult
from tests.td_case2.streaming_tracking_pipeline.serialization import read_jsonl


class PlateValidationArtifactTests(unittest.TestCase):
    def test_missing_artifact_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                validate_required_step8_inputs(temp_dir)

    def test_json_safe_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = FinalTrackAnprResult(
                source_id="s",
                track_id=1,
                track_generation=0,
                object_class="car",
                final_plate_text=None,
                plate_status="no_plate_detected",
                confidence=0.0,
                support_count=0,
                selected_candidate=None,
                all_candidates=[],
                agreement=None,
                normalized_colour="white",
                raw_colour="white",
                representative_frame_index=None,
                representative_timestamp_sec=None,
                representative_vehicle_crop_path=None,
                representative_plate_crop_path=None,
            )
            paths = PlateValidationArtifactSink(temp_dir).write([], [], [result], {"x": 1}, {"y": 2})
            self.assertTrue(Path(paths["final_track_anpr_results"]).exists())
            self.assertEqual(read_jsonl(paths["final_track_anpr_results"])[0]["plate_status"], "no_plate_detected")

    def test_colour_join_via_finalizer(self) -> None:
        from tests.td_case2.streaming_tracking_pipeline.run_step8_plate_validation import _finalize_track
        from tests.td_case2.streaming_tracking_pipeline.plate_validation_schemas import PlateValidationConfig

        final = _finalize_track(
            ("s", 1, 0),
            [],
            None,
            {"normalized_colour": "white", "colour_result": {"raw_text": "White"}},
            {"selected_plate_candidate": None, "object_class": "car"},
            {},
            PlateValidationConfig(),
        )
        self.assertEqual(final.normalized_colour, "white")
        self.assertEqual(final.raw_colour, "White")


if __name__ == "__main__":
    unittest.main()
