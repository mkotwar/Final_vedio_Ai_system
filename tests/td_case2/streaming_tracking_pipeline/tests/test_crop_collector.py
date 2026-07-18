from __future__ import annotations

import unittest

import numpy as np

from tests.td_case2.streaming_tracking_pipeline.config import CropCollectionConfig, TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.crop_collector import CropCandidateCollector
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, TrackCompletionReason, TrackedFramePacket, TrackedObject


def _packet(frame_index: int, tracks: list[TrackedObject]) -> TrackedFramePacket:
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    frame[10:50, 10:55, :] = 220
    return TrackedFramePacket(
        source_id="crop_source",
        frame_index=frame_index,
        timestamp_sec=frame_index / 2.0,
        frame_width=100,
        frame_height=80,
        tracks=tracks,
        frame=frame,
    )


def _track(frame_index: int, *, x_offset: int = 0) -> TrackedObject:
    return TrackedObject(
        track_id=1,
        source_track_id="raw_1",
        bbox=BoundingBox(10 + x_offset, 10, 50 + x_offset, 50),
        confidence=0.75 + frame_index * 0.02,
        class_id=2,
        class_name="car",
        frame_index=frame_index,
        timestamp_sec=frame_index / 2.0,
    )


class CropCandidateCollectorTest(unittest.TestCase):
    def test_collects_and_bounds_candidates(self) -> None:
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        collector = CropCandidateCollector(CropCollectionConfig(max_candidates_per_track=2, save_crop_images=False))
        for index in range(4):
            packet = _packet(index, [_track(index, x_offset=index)])
            collector.update(packet, manager.update(packet))

        retained = next(iter(collector._candidates_by_identity.values()))
        self.assertEqual(len(retained), 2)
        self.assertEqual(collector.total_candidates_created, 4)

    def test_completed_bundle_emitted_with_zero_candidates(self) -> None:
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        collector = CropCandidateCollector(CropCollectionConfig(minimum_bbox_area_ratio=0.5, save_crop_images=False))
        packet = _packet(0, [_track(0)])
        collector.update(packet, manager.update(packet))
        flush = manager.flush(frame_index=0, timestamp_sec=0.0, reason=TrackCompletionReason.VIDEO_ENDED)

        bundles = collector.complete_tracks(flush.newly_completed_tracks)

        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].retained_candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
