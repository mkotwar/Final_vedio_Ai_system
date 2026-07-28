from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.track_lifecycle import LocalTrackLifecycle
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import (
    ClassConflictSplitConfig,
    ClassStabilizationConfig,
    FragmentLinkingConfig,
    IdentityContinuityConfig,
    TrackingConfig,
)
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


def _class_observation(
    frame_number: int,
    *,
    class_name: str,
    bbox_xyxy: tuple[float, float, float, float],
    confidence: float = 0.9,
    local_track_id: int = 1,
    camera_code: str = "CAM_001",
) -> TrackObservation:
    return TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
    )


class LocalTrackLifecycleTests(unittest.TestCase):
    def test_standard_mode_uses_native_tracker_id_as_local_track_id(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(behavior_mode="standard_bytetrack", min_confirmed_observations=1))
        result = lifecycle.update(_packet(0), [_observation(0, local_track_id=7)])

        self.assertEqual(result.observations[0].local_track_id, 7)
        self.assertEqual(result.observations[0].track_uuid, "CAM_001:TRACK_7")

    def test_standard_mode_does_not_fragment_link_new_native_id(self) -> None:
        config = TrackingConfig(
            behavior_mode="standard_bytetrack",
            min_confirmed_observations=1,
            max_lost_frames=5,
            fragment_linking=FragmentLinkingConfig(enabled=True, maximum_gap_seconds=3.0),
        )
        lifecycle = LocalTrackLifecycle(config)
        lifecycle.update(_packet(0), [_observation(0, local_track_id=1)])
        lifecycle.update(_packet(1), [])
        relinked = lifecycle.update(_packet(2), [_observation(2, local_track_id=7)])

        self.assertEqual(relinked.observations[0].track_uuid, "CAM_001:TRACK_7")
        self.assertEqual(relinked.observations[0].local_track_id, 7)

    def test_standard_mode_does_not_split_same_native_id_on_class_conflict(self) -> None:
        config = TrackingConfig(
            behavior_mode="standard_bytetrack",
            min_confirmed_observations=1,
            class_conflict_split=ClassConflictSplitConfig(enabled=True),
            identity_continuity=IdentityContinuityConfig(enabled=True),
        )
        lifecycle = LocalTrackLifecycle(config)
        lifecycle.update(_packet(0), [_class_observation(0, class_name="car", bbox_xyxy=(1.0, 2.0, 10.0, 12.0), local_track_id=8)])
        result = lifecycle.update(_packet(1), [_class_observation(1, class_name="truck", bbox_xyxy=(80.0, 40.0, 120.0, 70.0), local_track_id=8)])

        self.assertEqual(len(result.completed_tracks), 0)
        self.assertEqual(result.observations[0].track_uuid, "CAM_001:TRACK_8")

    def test_standard_mode_preserves_raw_class_per_frame(self) -> None:
        lifecycle = LocalTrackLifecycle(
            TrackingConfig(
                behavior_mode="standard_bytetrack",
                min_confirmed_observations=1,
                track_class=TrackingConfig().track_class.__class__(minimum_observations=1, minimum_winner_ratio=0.50),
            )
        )
        lifecycle.update(_packet(0), [_class_observation(0, class_name="car", confidence=0.95, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        result = lifecycle.update(_packet(1), [_class_observation(1, class_name="truck", confidence=0.40, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])

        self.assertEqual(result.observations[0].class_name, "truck")
        completed = lifecycle.flush_camera("CAM_001").completed_tracks[0]
        self.assertEqual(completed.stable_class_name, "car")
        self.assertEqual(completed.raw_class_history[-1].class_name, "truck")

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
        self.assertEqual(result.completed_tracks[0].completion_reason, "identity_split")
        self.assertEqual(result.observations[0].track_uuid, "CAM_001:TRACK_9")

        completed = lifecycle.flush_camera("CAM_001").completed_tracks
        self.assertEqual(completed[0].split_from_track_uuid, "CAM_001:TRACK_8")

    def test_conflict_alone_does_not_split_without_spatial_discontinuity(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            class_stabilization=ClassStabilizationConfig(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            ),
            class_conflict_split=ClassConflictSplitConfig(
                enabled=True,
                minimum_consecutive_conflicting_observations=3,
                minimum_conflict_confidence=0.50,
                require_spatial_discontinuity=True,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        for frame in range(5):
            lifecycle.update(_packet(frame), [_observation(frame, local_track_id=1)])
        for frame in range(5, 8):
            result = lifecycle.update(
                _packet(frame),
                [
                    TrackObservation(
                        camera_code="CAM_001",
                        local_track_id=1,
                        frame_number=frame,
                        video_time_seconds=float(frame),
                        camera_timestamp=datetime(2026, 7, 22, 10, 0, frame),
                        class_name="truck",
                        confidence=0.82,
                        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                    )
                ],
            )
            self.assertEqual(len(result.completed_tracks), 0)

    def test_conflict_plus_spatial_discontinuity_can_split_logical_track(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            class_stabilization=ClassStabilizationConfig(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            ),
            identity_continuity=IdentityContinuityConfig(
                enabled=True,
                minimum_spatial_score=0.20,
                minimum_class_compatibility=0.50,
                maximum_area_ratio=3.50,
                hard_split_spatial_score=0.08,
            ),
            class_conflict_split=ClassConflictSplitConfig(
                enabled=True,
                minimum_consecutive_conflicting_observations=3,
                minimum_conflict_confidence=0.50,
                require_spatial_discontinuity=True,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        for frame in range(5):
            lifecycle.update(_packet(frame), [_observation(frame, local_track_id=1)])
        lifecycle.update(
            _packet(5),
            [
                TrackObservation(
                    camera_code="CAM_001",
                    local_track_id=1,
                    frame_number=5,
                    video_time_seconds=5.0,
                    camera_timestamp=datetime(2026, 7, 22, 10, 0, 5),
                    class_name="truck",
                    confidence=0.82,
                    bbox_xyxy=(1.0, 2.0, 12.0, 14.0),
                )
            ],
        )
        lifecycle.update(
            _packet(6),
            [
                TrackObservation(
                    camera_code="CAM_001",
                    local_track_id=1,
                    frame_number=6,
                    video_time_seconds=6.0,
                    camera_timestamp=datetime(2026, 7, 22, 10, 0, 6),
                    class_name="truck",
                    confidence=0.84,
                    bbox_xyxy=(1.0, 2.0, 12.0, 14.0),
                )
            ],
        )
        result = lifecycle.update(
            _packet(7),
            [
                TrackObservation(
                    camera_code="CAM_001",
                    local_track_id=1,
                    frame_number=7,
                    video_time_seconds=7.0,
                    camera_timestamp=datetime(2026, 7, 22, 10, 0, 7),
                    class_name="truck",
                    confidence=0.86,
                    bbox_xyxy=(1.0, 2.0, 40.0, 52.0),
                )
            ],
        )
        self.assertEqual(len(result.completed_tracks), 1)
        self.assertEqual(result.completed_tracks[0].completion_reason, "identity_split")

    def test_split_reassigns_pending_conflict_observations_without_loss_or_duplication(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            class_stabilization=ClassStabilizationConfig(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                lock_after_observations=3,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            ),
            class_conflict_split=ClassConflictSplitConfig(
                enabled=True,
                minimum_consecutive_conflicting_observations=3,
                minimum_conflict_confidence=0.50,
                minimum_average_conflict_confidence=0.50,
                require_spatial_discontinuity=True,
                maximum_iou_for_split=0.15,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        for frame in range(5):
            lifecycle.update(_packet(frame), [_class_observation(frame, class_name="car", bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        lifecycle.update(_packet(5), [_class_observation(5, class_name="truck", confidence=0.82, bbox_xyxy=(1.0, 2.0, 12.0, 14.0))])
        lifecycle.update(_packet(6), [_class_observation(6, class_name="truck", confidence=0.84, bbox_xyxy=(1.0, 2.0, 12.0, 14.0))])
        result = lifecycle.update(_packet(7), [_class_observation(7, class_name="truck", confidence=0.86, bbox_xyxy=(1.0, 2.0, 40.0, 52.0))])

        self.assertEqual(len(result.completed_tracks), 1)
        old_track = result.completed_tracks[0]
        new_track = lifecycle.get_active_tracks("CAM_001")[0]
        self.assertEqual([item.frame_number for item in old_track.observations], [0, 1, 2, 3, 4])
        self.assertEqual([item.frame_number for item in new_track.observations], [5, 6, 7])
        self.assertEqual(sorted(item.frame_number for item in old_track.observations + new_track.observations), list(range(8)))
        self.assertEqual(len({(item.frame_number, item.track_uuid) for item in old_track.observations + new_track.observations}), 8)

    def test_mixed_identity_can_execute_split_even_without_strong_conflict_lock(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            class_stabilization=ClassStabilizationConfig(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                lock_after_observations=5,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            ),
            class_conflict_split=ClassConflictSplitConfig(
                enabled=True,
                minimum_consecutive_conflicting_observations=3,
                minimum_conflict_confidence=0.50,
                minimum_average_conflict_confidence=0.50,
                require_spatial_discontinuity=True,
                maximum_iou_for_split=0.15,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        lifecycle.update(_packet(0), [_class_observation(0, class_name="car", confidence=0.62, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        lifecycle.update(_packet(1), [_class_observation(1, class_name="car", confidence=0.60, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        lifecycle.update(_packet(2), [_class_observation(2, class_name="car", confidence=0.58, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        lifecycle.update(_packet(3), [_class_observation(3, class_name="car", confidence=0.56, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        lifecycle.update(_packet(4), [_class_observation(4, class_name="truck", confidence=0.94, bbox_xyxy=(1.0, 2.0, 12.0, 14.0))])
        lifecycle.update(_packet(5), [_class_observation(5, class_name="truck", confidence=0.92, bbox_xyxy=(1.0, 2.0, 12.0, 14.0))])
        result = lifecycle.update(_packet(6), [_class_observation(6, class_name="truck", confidence=0.90, bbox_xyxy=(1.0, 2.0, 40.0, 52.0))])

        self.assertEqual(len(result.completed_tracks), 1)
        self.assertEqual(result.completed_tracks[0].completion_reason, "identity_split")
        self.assertFalse(result.completed_tracks[0].strong_conflict_detected)

    def test_split_metadata_keeps_same_native_tracker_but_new_logical_identity(self) -> None:
        config = TrackingConfig(
            min_confirmed_observations=1,
            class_stabilization=ClassStabilizationConfig(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                lock_after_observations=3,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            ),
            class_conflict_split=ClassConflictSplitConfig(
                enabled=True,
                minimum_consecutive_conflicting_observations=3,
                minimum_conflict_confidence=0.50,
                minimum_average_conflict_confidence=0.50,
                require_spatial_discontinuity=True,
                maximum_iou_for_split=0.15,
            ),
        )
        lifecycle = LocalTrackLifecycle(config)
        for frame in range(5):
            lifecycle.update(_packet(frame), [_class_observation(frame, class_name="car", bbox_xyxy=(1.0, 2.0, 10.0, 12.0), local_track_id=8)])
        lifecycle.update(_packet(5), [_class_observation(5, class_name="truck", confidence=0.82, bbox_xyxy=(1.0, 2.0, 12.0, 14.0), local_track_id=8)])
        lifecycle.update(_packet(6), [_class_observation(6, class_name="truck", confidence=0.84, bbox_xyxy=(1.0, 2.0, 12.0, 14.0), local_track_id=8)])
        result = lifecycle.update(_packet(7), [_class_observation(7, class_name="truck", confidence=0.86, bbox_xyxy=(1.0, 2.0, 40.0, 52.0), local_track_id=8)])

        old_track = result.completed_tracks[0]
        new_track = lifecycle.get_active_tracks("CAM_001")[0]
        self.assertTrue(old_track.split_executed)
        self.assertEqual(old_track.split_frame, 5)
        self.assertEqual(old_track.new_logical_track_id, new_track.local_track_id)
        self.assertNotEqual(old_track.track_uuid, new_track.track_uuid)
        self.assertEqual(old_track.split_native_tracker_id, 8)
        self.assertEqual(new_track.split_native_tracker_id, 8)
        self.assertEqual(new_track.split_from_track_uuid, old_track.track_uuid)
        self.assertEqual(new_track.stable_class_before_split, "car")


if __name__ == "__main__":
    unittest.main()
