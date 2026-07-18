import unittest

from tests.td_case2.streaming_tracking_pipeline.config import TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.schemas import (
    BoundingBox,
    TrackCompletionReason,
    TrackLifecycleEventType,
    TrackStatus,
    TrackedFramePacket,
    TrackedObject,
)


def track(track_id=1, frame_index=0, timestamp=0.0, class_name="car", source_track_id=None):
    return TrackedObject(
        track_id=track_id,
        source_track_id=source_track_id if source_track_id is not None else f"native_{track_id}",
        bbox=BoundingBox(1, 1, 20, 20),
        confidence=0.8,
        class_id=2 if class_name != "person" else 0,
        class_name=class_name,
        frame_index=frame_index,
        timestamp_sec=timestamp,
    )


def packet(frame_index, tracks, timestamp=None, source_id="cam_a"):
    ts = frame_index * 0.5 if timestamp is None else timestamp
    return TrackedFramePacket(source_id, frame_index, ts, 100, 80, tracks=list(tracks))


class TrackLifecycleManagerTest(unittest.TestCase):
    def test_creation_confirmation_and_threshold_one(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2))
        result = manager.update(packet(0, [track(1, 0, 0.0, source_track_id="vehicle_track_1")]))
        self.assertEqual([event.event_type for event in result.events], [TrackLifecycleEventType.CREATED, TrackLifecycleEventType.OBSERVED])
        self.assertEqual(result.active_tracks[0].status, TrackStatus.TENTATIVE)
        result = manager.update(packet(1, [track(1, 1, 0.5)]))
        self.assertIn(TrackLifecycleEventType.CONFIRMED, [event.event_type for event in result.events])
        self.assertEqual(result.active_tracks[0].status, TrackStatus.CONFIRMED)

        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        result = manager.update(packet(0, [track(2, 0, 0.0)]))
        self.assertEqual([event.event_type for event in result.events], [TrackLifecycleEventType.CREATED, TrackLifecycleEventType.OBSERVED, TrackLifecycleEventType.CONFIRMED])

    def test_tentative_missing_invalid_completion(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig(maximum_tentative_missed_frames=0))
        manager.update(packet(0, [track(1, 0, 0.0)]))
        result = manager.update(packet(1, []))
        self.assertEqual(result.newly_completed_tracks[0].completion_reason, TrackCompletionReason.INVALID_TRACK)
        self.assertEqual(result.newly_completed_tracks[0].status, TrackStatus.COMPLETED)

    def test_confirmed_missing_lost_once_recovery_and_expiry(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2, maximum_lost_processed_frames=1))
        manager.update(packet(0, [track(1, 0, 0.0)]))
        manager.update(packet(1, [track(1, 1, 0.5)]))
        lost = manager.update(packet(2, []))
        self.assertEqual([event.event_type for event in lost.events], [TrackLifecycleEventType.TEMPORARILY_LOST])
        still_lost = manager.update(packet(3, []))
        self.assertNotIn(TrackLifecycleEventType.TEMPORARILY_LOST, [event.event_type for event in still_lost.events])
        self.assertEqual(still_lost.newly_completed_tracks[0].completion_reason, TrackCompletionReason.LOST_BUFFER_EXPIRED)

        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2, maximum_lost_processed_frames=3))
        manager.update(packet(0, [track(1, 0, 0.0)]))
        manager.update(packet(1, [track(1, 1, 0.5)]))
        manager.update(packet(2, []))
        recovered = manager.update(packet(3, [track(1, 3, 1.5)]))
        self.assertIn(TrackLifecycleEventType.RECOVERED, [event.event_type for event in recovered.events])
        self.assertEqual(recovered.active_tracks[0].missed_frame_count, 0)

    def test_time_based_and_combined_expiry_use_exceeded_policy(self):
        manager = TrackLifecycleManager(
            TrackLifecycleConfig(minimum_confirmation_observations=1, maximum_lost_processed_frames=99, maximum_lost_seconds=0.5)
        )
        manager.update(packet(0, [track(1, 0, 0.0)], timestamp=0.0))
        not_expired = manager.update(packet(1, [], timestamp=0.5))
        self.assertEqual(not_expired.newly_completed_tracks, [])
        expired = manager.update(packet(2, [], timestamp=0.6))
        self.assertEqual(expired.newly_completed_tracks[0].completion_reason, TrackCompletionReason.LOST_BUFFER_EXPIRED)

        manager = TrackLifecycleManager(
            TrackLifecycleConfig(minimum_confirmation_observations=1, maximum_lost_processed_frames=0, maximum_lost_seconds=99.0)
        )
        manager.update(packet(0, [track(1, 0, 0.0)]))
        expired = manager.update(packet(1, []))
        self.assertEqual(expired.newly_completed_tracks[0].completion_reason, TrackCompletionReason.LOST_BUFFER_EXPIRED)

    def test_flush_reset_rejections_and_duplicate_validation(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=1))
        manager.update(packet(0, [track(1, 0, 0.0)]))
        flushed = manager.flush(reason=TrackCompletionReason.VIDEO_ENDED)
        self.assertEqual(flushed.newly_completed_tracks[0].completion_reason, TrackCompletionReason.VIDEO_ENDED)
        self.assertIn(TrackLifecycleEventType.FLUSHED, [event.event_type for event in flushed.events])
        manager.reset()
        manager.update(packet(0, [track(1, 0, 0.0)], source_id="cam_b"))
        with self.assertRaisesRegex(ValueError, "second source"):
            manager.update(packet(1, [], source_id="cam_c"))
        manager.reset()
        manager.update(packet(1, []))
        with self.assertRaisesRegex(ValueError, "frame-index"):
            manager.update(packet(1, []))
        manager.reset()
        manager.update(packet(0, [], timestamp=1.0))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            manager.update(packet(1, [], timestamp=0.9))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TrackLifecycleManager(TrackLifecycleConfig()).update(packet(0, [track(1, 0, 0.0), track(1, 0, 0.0)]))
        with self.assertRaisesRegex(ValueError, "TrackedObject metadata mismatch"):
            TrackLifecycleManager(TrackLifecycleConfig()).update(packet(0, [track(1, 1, 0.0)]))

    def test_completed_id_reuse_generation_class_voting_and_snapshot_immutability(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2, maximum_lost_processed_frames=0))
        first = manager.update(packet(0, [track(7, 0, 0.0, "car", source_track_id="source7")]))
        snapshot = first.active_tracks[0]
        manager.update(packet(1, [track(7, 1, 0.5, "truck", source_track_id="source7")]))
        manager.update(packet(2, []))
        manager.update(packet(3, [track(7, 3, 1.5, "bus", source_track_id="source7")]))
        records = [record for record in manager.get_completed_tracks() if record.track_id == 7]
        self.assertEqual(records[0].track_generation, 0)
        active = manager.get_track(7)
        self.assertEqual(active.track_generation, 1)
        self.assertEqual(active.source_track_id, "source7")
        self.assertEqual(snapshot.observation_count, 1)

        manager = TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=3))
        manager.update(packet(0, [track(1, 0, 0.0, "truck")]))
        changed = manager.update(packet(1, [track(1, 1, 0.5, "car")]))
        manager.update(packet(2, [track(1, 2, 1.0, "car")]))
        self.assertEqual(manager.get_track(1).dominant_class, "car")
        self.assertIn(TrackLifecycleEventType.CLASS_UPDATED, [event.event_type for event in changed.events])
        tie = manager.get_track(1)
        self.assertEqual(tie.class_votes["car"], 2)

    def test_deterministic_ordering(self):
        manager = TrackLifecycleManager(TrackLifecycleConfig())
        result = manager.update(packet(0, [track(3, 0, 0.0), track(1, 0, 0.0)]))
        self.assertEqual([event.track_id for event in result.events if event.event_type == TrackLifecycleEventType.CREATED], [1, 3])
        self.assertEqual([record.track_id for record in result.active_tracks], [1, 3])


if __name__ == "__main__":
    unittest.main()
