from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.crop_artifacts import CompletedTrackCropBundle, CropArtifactSink
from tests.td_case2.streaming_tracking_pipeline.schemas import TrackRecord, TrackStatus


class CropArtifactTest(unittest.TestCase):
    def test_sink_writes_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = CropArtifactSink(temp_dir)
            track = TrackRecord(
                source_id="source/a",
                track_id=7,
                track_generation=2,
                source_track_id="raw",
                status=TrackStatus.COMPLETED,
                first_seen_frame=0,
                last_seen_frame=0,
                first_seen_sec=0.0,
                last_seen_sec=0.0,
                observation_count=1,
                missed_frame_count=0,
            )
            bundle = CompletedTrackCropBundle.from_track(track, [])
            sink.write_completed_bundle(bundle)
            sink.write_summary({"ok": True})
            sink.close()

            completed_path = Path(temp_dir) / "05_crops" / "completed_track_crop_bundles.jsonl"
            self.assertEqual(len(completed_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((Path(temp_dir) / "reports" / "step5_crop_collection_report.json").exists())


if __name__ == "__main__":
    unittest.main()
