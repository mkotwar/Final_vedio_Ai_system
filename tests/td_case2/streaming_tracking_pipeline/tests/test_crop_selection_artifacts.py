from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.config import BestCropSelectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import FinalBestCropSelector
from tests.td_case2.streaming_tracking_pipeline.crop_selection_artifacts import (
    CropSelectionArtifactSink,
    bundle_from_dict,
    read_completed_crop_bundles,
)
from tests.td_case2.streaming_tracking_pipeline.crop_selection_metrics import build_selection_summary
from tests.td_case2.streaming_tracking_pipeline.tests.test_crop_selection import bundle, candidate


class CropSelectionArtifactTest(unittest.TestCase):
    def test_jsonl_outputs_summary_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=1))
            result = selector.select(bundle([candidate(0), candidate(4, path=None)]))
            sink = CropSelectionArtifactSink(temp_dir, create_previews=False)
            sink.write_result(result)
            sink.write_summary(build_selection_summary([result], primary_target=1))
            sink.close()
            sink.close()

            base = Path(temp_dir) / "06_selected_crops"
            self.assertEqual(len((base / "selected_track_crop_sets.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertGreaterEqual(len((base / "selected_primary_crops.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((base / "crop_selection_rejections.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "reports" / "crop_selection_summary.json").exists())
            payload = json.loads((base / "selected_track_crop_sets.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn('"frame":', json.dumps(payload).lower())
            self.assertNotIn("ndarray", json.dumps(payload).lower())

    def test_bundle_reader_round_trips_step5_shape(self) -> None:
        source_bundle = bundle([candidate(0)])
        payload = source_bundle.to_dict()

        parsed = bundle_from_dict(payload)

        self.assertEqual(parsed.track_id, source_bundle.track_id)
        self.assertEqual(parsed.candidates[0].frame_index, 0)

    def test_read_completed_crop_bundles_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            crop_dir = run_dir / "05_crops"
            crop_dir.mkdir(parents=True)
            (crop_dir / "completed_track_crop_bundles.jsonl").write_text(
                json.dumps(bundle([candidate(0)]).to_dict()) + "\n",
                encoding="utf-8",
            )

            parsed = read_completed_crop_bundles(run_dir)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0].retained_candidate_count, 1)

    def test_preview_failure_preserves_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=1))
            result = selector.select(bundle([candidate(0, path="missing.jpg")]))
            sink = CropSelectionArtifactSink(temp_dir, create_previews=True)
            sink.write_result(result)
            sink.write_summary(build_selection_summary([result], primary_target=1))
            sink.close()

            self.assertTrue((Path(temp_dir) / "06_selected_crops" / "selected_track_crop_sets.jsonl").exists())
            self.assertGreaterEqual(sink.counts["preview_failures"], 1)


if __name__ == "__main__":
    unittest.main()
