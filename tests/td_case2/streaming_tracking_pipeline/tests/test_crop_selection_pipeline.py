from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.config import BestCropSelectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import FinalBestCropSelector
from tests.td_case2.streaming_tracking_pipeline.crop_selection_artifacts import CropSelectionArtifactSink
from tests.td_case2.streaming_tracking_pipeline.crop_selection_pipeline import (
    finalize_step6_artifacts,
    run_selection_for_existing_bundles,
    select_completed_crop_bundles,
)
from tests.td_case2.streaming_tracking_pipeline.tests.test_crop_selection import bundle, candidate


class RecordingSelector(FinalBestCropSelector):
    def __init__(self) -> None:
        super().__init__(BestCropSelectionConfig(primary_crop_count=1))
        self.calls: list[tuple[int, int]] = []

    def select(self, crop_bundle):  # type: ignore[no-untyped-def]
        self.calls.append((crop_bundle.track_id, crop_bundle.track_generation))
        return super().select(crop_bundle)


class FailingSelector(FinalBestCropSelector):
    def select(self, crop_bundle):  # type: ignore[no-untyped-def]
        raise RuntimeError("selector failed for test")


class CropSelectionPipelineTest(unittest.TestCase):
    def test_selects_all_bundles_in_deterministic_order(self) -> None:
        selector = RecordingSelector()
        bundles = [
            bundle([candidate(0, track_id=2)], track_id=2),
            bundle([candidate(0, track_id=1, generation=1)], track_id=1, generation=1),
            bundle([candidate(0, track_id=1)], track_id=1),
        ]

        results = select_completed_crop_bundles(bundles, selector=selector)

        self.assertEqual(selector.calls, [(1, 0), (1, 1), (2, 0)])
        self.assertEqual(len(results), 3)

    def test_existing_bundle_runner_writes_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = CropSelectionArtifactSink(temp_dir, create_previews=False)
            report = run_selection_for_existing_bundles(
                run_id="test_run",
                mode="existing_step5_artifacts",
                bundles=[bundle([candidate(0), candidate(4)])],
                selector=FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=1)),
                sink=sink,
            )
            finalize_step6_artifacts(temp_dir, report)

            self.assertTrue(report.sink_closed)
            self.assertEqual(report.completed_crop_bundles, 1)
            self.assertTrue((Path(temp_dir) / "reports" / "step6_best_crop_pipeline_report.json").exists())

    def test_sink_closes_after_selector_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = CropSelectionArtifactSink(temp_dir, create_previews=False)
            with self.assertRaises(RuntimeError):
                run_selection_for_existing_bundles(
                    run_id="test_run",
                    mode="existing_step5_artifacts",
                    bundles=[bundle([candidate(0)])],
                    selector=FailingSelector(),
                    sink=sink,
                )

            self.assertTrue(sink.closed)

    def test_deterministic_rerun(self) -> None:
        bundles = [bundle([candidate(0), candidate(4), candidate(8)])]
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=2))

        left = [item.to_dict() for item in select_completed_crop_bundles(bundles, selector=selector)]
        right = [item.to_dict() for item in select_completed_crop_bundles(bundles, selector=selector)]

        self.assertEqual(left, right)

    def test_pipeline_module_uses_no_queue_or_threading_primitives(self) -> None:
        import tests.td_case2.streaming_tracking_pipeline.crop_selection_pipeline as module

        names = set(module.__dict__)
        self.assertNotIn("Queue", names)
        self.assertNotIn("Thread", names)
        self.assertNotIn("Process", names)


if __name__ == "__main__":
    unittest.main()
