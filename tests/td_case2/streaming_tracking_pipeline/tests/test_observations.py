from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.config import CropCollectionConfig, TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.observations import TrackIdentity, TrackObservationCollector
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, TrackedFramePacket, TrackedObject


def _track(frame_index: int, timestamp_sec: float, *, track_id: int = 1) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        source_track_id=f"raw_{track_id}",
        bbox=BoundingBox(10, 10, 40, 40),
        confidence=0.8,
        class_id=2,
        class_name="car",
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def _packet(frame_index: int, tracks: list[TrackedObject]) -> TrackedFramePacket:
    timestamp = frame_index / 2.0
    return TrackedFramePacket(
        source_id="obs_source",
        frame_index=frame_index,
        timestamp_sec=timestamp,
        frame_width=100,
        frame_height=80,
        tracks=tracks,
        frame=object(),
    )


class TrackObservationCollectorTest(unittest.TestCase):
    def test_collects_identity_with_generation(self) -> None:
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        collector = TrackObservationCollector(CropCollectionConfig(max_observations_per_track=2))
        packet = _packet(0, [_track(0, 0.0)])
        result = manager.update(packet)
        collected = collector.collect(packet, result)

        self.assertEqual(len(collected.observations), 1)
        observation = collected.observations[0].observation
        self.assertEqual(observation.identity, TrackIdentity("obs_source", 1, 0))
        self.assertEqual(observation.lifecycle_status.value, "confirmed")

    def test_rejects_duplicate_observation_for_same_generation(self) -> None:
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        collector = TrackObservationCollector(CropCollectionConfig())
        packet = _packet(0, [_track(0, 0.0)])
        result = manager.update(packet)
        collector.collect(packet, result)
        duplicate = collector.collect(packet, result)

        self.assertEqual(duplicate.dropped_count, 1)
        self.assertEqual(duplicate.drop_reasons["duplicate_or_regressed_observation"], 1)


if __name__ == "__main__":
    unittest.main()
