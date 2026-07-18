import unittest

from tests.td_case2.streaming_tracking_pipeline.config import TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.lifecycle_metrics import LifecycleMetricsAccumulator
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, TrackCompletionReason, TrackedFramePacket, TrackedObject


def tr(track_id, frame_index):
    return TrackedObject(track_id, BoundingBox(1, 1, 20, 20), 0.8, 2, "car", frame_index, frame_index * 0.5, source_track_id=track_id)


def pkt(frame_index, tracks):
    return TrackedFramePacket("cam_a", frame_index, frame_index * 0.5, 100, 80, list(tracks))


class LifecycleMetricsAccumulatorTest(unittest.TestCase):
    def test_empty_and_flush_only_runs(self):
        self.assertEqual(LifecycleMetricsAccumulator().to_dict()["tracks_created"], 0)
        manager = TrackLifecycleManager(TrackLifecycleConfig())
        metrics = LifecycleMetricsAccumulator()
        metrics.update(manager.flush())
        payload = metrics.to_dict()
        self.assertEqual(payload["tracks_completed"], 0)
        self.assertEqual(payload["completed_tracks_at_end"], 0)

    def test_counts_recovery_expiry_completion_and_generations(self):
        manager = TrackLifecycleManager(
            TrackLifecycleConfig(minimum_confirmation_observations=2, maximum_tentative_missed_frames=0, maximum_lost_processed_frames=1)
        )
        metrics = LifecycleMetricsAccumulator()
        for result in [
            manager.update(pkt(0, [tr(1, 0), tr(2, 0)])),
            manager.update(pkt(1, [tr(1, 1)])),
            manager.update(pkt(2, [])),
            manager.update(pkt(3, [tr(1, 3)])),
            manager.update(pkt(4, [])),
            manager.update(pkt(5, [])),
            manager.update(pkt(6, [tr(1, 6)])),
            manager.flush(reason=TrackCompletionReason.VIDEO_ENDED),
        ]:
            metrics.update(result)

        payload = metrics.to_dict()

        self.assertEqual(payload["tracks_created"], 3)
        self.assertEqual(payload["tracks_confirmed"], 1)
        self.assertGreaterEqual(payload["tracks_completed"], 3)
        self.assertEqual(payload["tracks_completed_invalid"], 1)
        self.assertEqual(payload["tracks_completed_lost_buffer"], 1)
        self.assertEqual(payload["tracks_completed_video_end"], 1)
        self.assertEqual(payload["tracks_temporarily_lost"], 2)
        self.assertEqual(payload["tracks_recovered"], 1)
        self.assertEqual(payload["recovery_attempts"], 2)
        self.assertEqual(payload["successful_same_id_recoveries"], 1)
        self.assertEqual(payload["expired_lost_tracks"], 1)
        self.assertGreaterEqual(payload["active_tracks_at_peak"], 1)
        self.assertEqual(payload["active_tracks_at_end"], 0)
        self.assertEqual(payload["completed_tracks_at_end"], 3)
        self.assertGreater(payload["average_observations_per_completed_track"], 0)
        self.assertGreater(payload["median_observations_per_completed_track"], 0)
        self.assertEqual(payload["tracks_with_one_observation"], 2)
        self.assertEqual(payload["tracks_with_less_than_three_observations"], 2)
        self.assertIn("lost_buffer_expired", payload["completion_reason_counts"])
        self.assertIn("confirmed->temporarily_lost", payload["status_transition_counts"])
        self.assertEqual(payload["generation_count"], 3)
        self.assertEqual(payload["reused_completed_track_ids"], 1)


if __name__ == "__main__":
    unittest.main()
