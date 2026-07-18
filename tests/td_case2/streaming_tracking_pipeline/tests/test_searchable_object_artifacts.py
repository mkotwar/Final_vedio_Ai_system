from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.searchable_object_artifacts import (
    SearchableObjectArtifactSink,
    validate_required_step9_inputs,
)
from tests.td_case2.streaming_tracking_pipeline.searchable_object_schemas import SearchableVehicleRecord
from tests.td_case2.streaming_tracking_pipeline.serialization import read_jsonl


class SearchableObjectArtifactTests(unittest.TestCase):
    def test_missing_artifact_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                validate_required_step9_inputs(temp_dir)

    def test_writes_json_safe_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record = SearchableVehicleRecord(
                record_id="s:track_000001:gen_000",
                source_id="s",
                video_path=None,
                track_id=1,
                track_generation=0,
                object_class="car",
                first_frame_index=1,
                last_frame_index=2,
                first_seen_sec=0.1,
                last_seen_sec=0.2,
                duration_sec=0.1,
                plate_text=None,
                plate_status="no_plate_detected",
                plate_confidence=0.0,
                plate_support_count=0,
                normalized_colour="white",
                raw_colour="white",
                representative_frame_index=1,
                representative_timestamp_sec=0.1,
                representative_vehicle_crop_path="vehicle.jpg",
                representative_plate_crop_path=None,
            )
            paths = SearchableObjectArtifactSink(temp_dir).write([record], {"x": 1}, {"y": 2})
            self.assertTrue(Path(paths["searchable_vehicle_records"]).exists())
            self.assertTrue(Path(paths["searchable_vehicle_records_flat"]).exists())
            self.assertEqual(read_jsonl(paths["no_plate_vehicle_records"])[0]["record_id"], record.record_id)


if __name__ == "__main__":
    unittest.main()
