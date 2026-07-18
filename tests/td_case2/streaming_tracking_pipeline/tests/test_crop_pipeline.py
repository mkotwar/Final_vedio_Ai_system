from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.td_case2.streaming_tracking_pipeline.config import CropCollectionConfig, TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.crop_artifacts import CropArtifactSink
from tests.td_case2.streaming_tracking_pipeline.crop_collector import CropCandidateCollector
from tests.td_case2.streaming_tracking_pipeline.crop_pipeline import SequentialCropCollectionPipeline, finalize_step5_artifacts
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.lifecycle_pipeline import LifecycleArtifactSink
from tests.td_case2.streaming_tracking_pipeline.mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionRecord, TrackedObject
from tests.td_case2.streaming_tracking_pipeline.sources import SyntheticFrameSource


def _frame(_: int) -> np.ndarray:
    image = np.full((64, 80, 3), 90, dtype=np.uint8)
    image[10:40, 10:45, :] = 230
    return image


def _tracks(packet: object) -> list[TrackedObject]:
    frame_index = getattr(packet, "frame_index")
    timestamp_sec = getattr(packet, "timestamp_sec")
    if frame_index > 2:
        return []
    return [
        TrackedObject(
            track_id=1,
            source_track_id="raw_1",
            bbox=BoundingBox(10 + frame_index, 10, 45 + frame_index, 40),
            confidence=0.8,
            class_id=2,
            class_name="car",
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
    ]


class CropPipelineTest(unittest.TestCase):
    def test_pipeline_emits_crop_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = SyntheticFrameSource(
                source_id="pipe_source",
                total_frames=5,
                source_fps=2.0,
                frame_width=80,
                frame_height=64,
                use_source_fps=True,
                frame_factory=_frame,
            )
            detector = DeterministicMockDetectionStage(
                detection_factory=lambda packet: [
                    DetectionRecord(
                        bbox=track.bbox,
                        confidence=track.confidence,
                        class_id=track.class_id,
                        class_name=track.class_name,
                    )
                    for track in _tracks(packet)
                ]
            )
            tracker = DeterministicMockTrackingStage(track_factory=_tracks)
            pipeline = SequentialCropCollectionPipeline(
                run_id="pipe_run",
                source=source,
                detection_stage=detector,
                tracking_stage=tracker,
                lifecycle_manager=TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1)),
                crop_collector=CropCandidateCollector(CropCollectionConfig(save_crop_images=False, max_candidates_per_track=2)),
                lifecycle_sink=LifecycleArtifactSink(run_dir),
                crop_sink=CropArtifactSink(run_dir),
                source_path="synthetic",
                tracking_backend="mock",
            )

            report = pipeline.run()
            finalize_step5_artifacts(run_dir, report, {"source_type": "synthetic"})

            self.assertEqual(report.selected_frames_processed, 5)
            self.assertGreater(report.crop_collection_summary["total_candidates_created"], 0)
            self.assertTrue((run_dir / "05_crops" / "crop_candidates.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
