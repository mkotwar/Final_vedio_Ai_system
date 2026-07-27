from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.track_lifecycle import LocalTrackLifecycle
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import FragmentLinkingConfig, IdentityContinuityConfig, TrackingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import TrackObservation


def _packet(frame_number: int, camera_code: str = "CAM_001") -> DetectionPacket:
    return DetectionPacket(
        camera_code=camera_code,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        frame_width=128,
        frame_height=72,
        detections=[],
        inference_time_ms=0.0,
        detector_model="fake.pt",
        detector_device="cpu",
    )


def _observation(frame_number: int, local_track_id: int = 1, camera_code: str = "CAM_001") -> TrackObservation:
    return TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
    )


class LocalTrackLifecycleTests(unittest.TestCase):
    def test_tentative_to_active_transition(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=2))
        first = lifecycle.update(_packet(0), [_observation(0)])
        second = lifecycle.update(_packet(1), [_observation(1)])
        self.assertEqual(first.observations[0].state, "tentative")
        self.assertEqual(second.observations[0].state, "active")

    def test_active_to_temporarily_lost_transition(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=5))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [])
        self.assertEqual(lifecycle.get_active_tracks("CAM_001")[0].state, "temporarily_lost")

    def test_temporarily_lost_to_active_recovery(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=5))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [])
        recovered = lifecycle.update(_packet(2), [_observation(2)])
        self.assertEqual(recovered.observations[0].state, "active")

    def test_lost_timeout_completes_active_track(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=0))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.update(_packet(1), [])
        self.assertEqual(result.completed_tracks[0].state, "completed")

    def test_short_tentative_track_is_discarded(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=2, max_lost_frames=0))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.update(_packet(1), [])
        self.assertEqual(result.completed_tracks[0].state, "discarded")

    def test_flush_completes_valid_tracks(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.flush_camera("CAM_001")
        self.assertEqual(result.completed_tracks[0].state, "completed")

    def test_dominant_class_is_not_overwritten_by_latest_observation(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [_observation(1)])
        conflicting = TrackObservation(
            camera_code="CAM_001",
            local_track_id=1,
            frame_number=2,
            video_time_seconds=2.0,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 2),
            class_name="bus",
            confidence=0.87,
            bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        )
        lifecycle.update(_packet(2), [conflicting])
        completed = lifecycle.flush_camera("CAM_001").completed_tracks[0]
        self.assertEqual(completed.class_name, "car")
        self.assertEqual(completed.stable_class_name, "car")
        self.assertEqual(completed.latest_observation_class_name, "bus")
        self.assertEqual(completed.class_observation_counts["car"], 2)
        self.assertEqual(completed.class_observation_counts["bus"], 1)

    def test_aliases_normalize_to_canonical_class(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1))
        alias = TrackObservation(
            camera_code="CAM_001",
            local_track_id=1,
            frame_number=0,
            video_time_seconds=0.0,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 0),
            class_name="auto_rickshaw",
            confidence=0.91,
            bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        )
        result = lifecycle.update(_packet(0), [alias])
        self.assertEqual(result.observations[0].class_name, "3wheeler")
        completed = lifecycle.flush_camera("CAM_001").completed_tracks[0]
        self.assertEqual(completed.class_name, "3wheeler")

    def test_repeated_strong_conflict_can_change_final_class(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [_observation(1)])
        lifecycle.update(_packet(2), [_observation(2)])
        for frame in (3, 4, 5, 6, 7):
            lifecycle.update(
                _packet(frame),
                [
                    TrackObservation(
                        camera_code="CAM_001",
                        local_track_id=1,
                        frame_number=frame,
                        video_time_seconds=float(frame),
                        camera_timestamp=datetime(2026, 7, 22, 10, 0, frame),
                        class_name="bus",
                        confidence=0.99,
                        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                    )
                ],
            )
        completed = lifecycle.flush_camera("CAM_001").completed_tracks[0]
        self.assertEqual(completed.class_name, "bus")
        self.assertEqual(completed.latest_observation_class_name, "bus")

    def test_same_camera_fragment_reuses_original_track_identity_when_linking_enabled(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            max_lost_frames=5,
            fragment_linking=FragmentLinkingConfig(enabled=True, maximum_gap_seconds=3.0),
        )
        lifecycle = LocalTrackLifecycle(config)
        lifecycle.update(_packet(0), [_observation(0, local_track_id=1)])
        lifecycle.update(_packet(1), [])
        relinked = lifecycle.update(_packet(2), [_observation(2, local_track_id=7)])

        self.assertEqual(relinked.observations[0].track_uuid, "CAM_001:TRACK_1")
        self.assertEqual(relinked.observations[0].local_track_id, 1)
        self.assertEqual(len(relinked.completed_tracks), 0)

        completed = lifecycle.flush_camera("CAM_001").completed_tracks
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].track_uuid, "CAM_001:TRACK_1")
        self.assertIn("CAM_001:TRACK_7", completed[0].fragment_candidate_track_uuids)

    def test_same_tracker_id_is_split_when_identity_break_is_strong(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            identity_continuity=IdentityContinuityConfig(
                enabled=True,
                minimum_spatial_score=0.20,
                minimum_class_compatibility=0.50,
                maximum_area_ratio=3.50,
                hard_split_spatial_score=0.08,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        lifecycle.update(_packet(0), [_observation(0, local_track_id=8)])
        switched = TrackObservation(
            camera_code="CAM_001",
            local_track_id=8,
            frame_number=1,
            video_time_seconds=1.0,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 1),
            class_name="truck",
            confidence=0.92,
            bbox_xyxy=(90.0, 40.0, 120.0, 70.0),
        )
        result = lifecycle.update(_packet(1), [switched])
        self.assertEqual(len(result.completed_tracks), 1)
        self.assertEqual(result.completed_tracks[0].track_uuid, "CAM_001:TRACK_8")
        self.assertEqual(result.observations[0].track_uuid, "CAM_001:TRACK_9")
        self.assertEqual(result.observations[0].local_track_id, 9)


if __name__ == "__main__":
    unittest.main()
