from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.class_recalculation import recalculate_track_class
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.class_stabilization import build_class_diagnostics
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import ClassObservation
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig, load_tracking_config
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import TrackObservation


def _observation(frame_number: int, class_name: str, confidence: float) -> TrackObservation:
    return TrackObservation(
        camera_code="CAM_002",
        local_track_id=1,
        frame_number=frame_number,
        video_time_seconds=float(frame_number) / 30.0,
        camera_timestamp=datetime(2026, 7, 28, 12, 0, min(frame_number, 59)),
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(10.0, 10.0, 30.0, 20.0),
        track_uuid="CAM_002:TRACK_1",
        state="active",
        raw_class_name=class_name,
    )


class ClassStabilizationTests(unittest.TestCase):
    def test_standard_mode_uses_count_majority(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.40),
                _observation(1, "car", 0.41),
                _observation(2, "truck", 0.90),
                _observation(3, "truck", 0.91),
                _observation(4, "truck", 0.92),
            ],
            TrackingConfig(behavior_mode="standard_bytetrack"),
        )

        self.assertEqual(diagnostics.stable_class_name, "truck")
        self.assertEqual(diagnostics.final_class_reason, "COUNT_MAJORITY")
        self.assertAlmostEqual(diagnostics.winner_confidence_sum or 0.0, 2.73)

    def test_standard_mode_returns_unknown_when_winner_ratio_is_too_low(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
                _observation(2, "truck", 0.87),
                _observation(3, "truck", 0.86),
            ],
            TrackingConfig(
                behavior_mode="standard_bytetrack",
                track_class=TrackingConfig().track_class.__class__(minimum_observations=3, minimum_winner_ratio=0.60),
            ),
        )

        self.assertIsNone(diagnostics.stable_class_name)
        self.assertEqual(diagnostics.final_class_reason, "NO_CLEAR_WINNER")

    def test_standard_mode_reports_possible_identity_switch_without_mixed_identity_state(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
                TrackObservation(
                    camera_code="CAM_002",
                    local_track_id=1,
                    frame_number=2,
                    video_time_seconds=2.0 / 30.0,
                    camera_timestamp=datetime(2026, 7, 28, 12, 0, 2),
                    class_name="truck",
                    confidence=0.86,
                    bbox_xyxy=(80.0, 10.0, 120.0, 30.0),
                    track_uuid="CAM_002:TRACK_1",
                    state="active",
                    raw_class_name="truck",
                ),
                TrackObservation(
                    camera_code="CAM_002",
                    local_track_id=1,
                    frame_number=3,
                    video_time_seconds=3.0 / 30.0,
                    camera_timestamp=datetime(2026, 7, 28, 12, 0, 3),
                    class_name="truck",
                    confidence=0.84,
                    bbox_xyxy=(82.0, 10.0, 122.0, 30.0),
                    track_uuid="CAM_002:TRACK_1",
                    state="active",
                    raw_class_name="truck",
                ),
            ],
            TrackingConfig(behavior_mode="standard_bytetrack"),
        )

        self.assertTrue(diagnostics.possible_identity_switch)
        self.assertFalse(diagnostics.mixed_identity_detected)

    def test_winning_and_runner_up_ratios_are_calculated(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.92),
                _observation(1, "car", 0.88),
                _observation(2, "bus", 0.31),
                _observation(3, "car", 0.84),
            ],
            TrackingConfig(),
        )

        self.assertEqual(diagnostics.winning_class_name, "car")
        self.assertEqual(diagnostics.winning_class_count, 3)
        self.assertAlmostEqual(diagnostics.winning_class_ratio or 0.0, 0.75)
        self.assertEqual(diagnostics.runner_up_class_name, "bus")
        self.assertAlmostEqual(diagnostics.runner_up_ratio or 0.0, 0.25)

    def test_maximum_consecutive_winner_and_recent_window_are_calculated(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.8),
                _observation(1, "car", 0.8),
                _observation(2, "car", 0.8),
                _observation(3, "bus", 0.8),
                _observation(4, "car", 0.8),
                _observation(5, "car", 0.8),
            ],
            TrackingConfig(),
        )

        self.assertEqual(diagnostics.maximum_consecutive_winner_count, 3)
        self.assertEqual(diagnostics.recent_winning_class_name, "car")
        self.assertAlmostEqual(diagnostics.recent_winning_ratio or 0.0, 0.8)
        self.assertEqual(diagnostics.recent_consecutive_winner_count, 2)

    def test_count_and_score_winner_now_match_under_count_based_logic(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.32),
                _observation(1, "car", 0.30),
                _observation(2, "truck", 0.94),
                _observation(3, "truck", 0.91),
                _observation(4, "car", 0.20),
            ],
            TrackingConfig(class_stabilization=TrackingConfig().class_stabilization),
        )

        self.assertEqual(diagnostics.count_winner_class_name, "car")
        self.assertEqual(diagnostics.score_winner_class_name, "car")
        self.assertTrue(diagnostics.winners_agree)

    def test_stable_class_is_accepted_when_thresholds_pass(self) -> None:
        config = TrackingConfig(
            class_stabilization=TrackingConfig().class_stabilization.__class__(
                minimum_observations=4,
                minimum_consistency_ratio=0.70,
                minimum_consecutive_winner_observations=3,
            )
        )
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
                _observation(2, "car", 0.84),
                _observation(3, "bus", 0.31),
                _observation(4, "car", 0.82),
            ],
            config,
        )

        self.assertEqual(diagnostics.stable_class_name, "car")
        self.assertEqual(diagnostics.class_status, "LOCKED")

    def test_stable_class_is_rejected_when_ratio_is_too_low(self) -> None:
        config = TrackingConfig(
            class_stabilization=TrackingConfig().class_stabilization.__class__(
                minimum_observations=4,
                minimum_consistency_ratio=0.80,
                minimum_consecutive_winner_observations=2,
            )
        )
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
                _observation(2, "truck", 0.84),
                _observation(3, "truck", 0.83),
            ],
            config,
        )

        self.assertIsNone(diagnostics.stable_class_name)
        self.assertEqual(diagnostics.class_status, "MIXED_IDENTITY")

    def test_stable_class_is_rejected_when_observations_are_insufficient(self) -> None:
        config = TrackingConfig(
            class_stabilization=TrackingConfig().class_stabilization.__class__(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
            )
        )
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
            ],
            config,
        )

        self.assertIsNone(diagnostics.stable_class_name)
        self.assertEqual(diagnostics.class_status, "INSUFFICIENT_OBSERVATIONS")

    def test_stable_class_is_rejected_when_consecutive_count_is_too_low(self) -> None:
        config = TrackingConfig(
            class_stabilization=TrackingConfig().class_stabilization.__class__(
                minimum_observations=5,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=3,
            )
        )
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "truck", 0.40),
                _observation(2, "car", 0.88),
                _observation(3, "bus", 0.30),
                _observation(4, "car", 0.84),
            ],
            config,
        )

        self.assertIsNone(diagnostics.stable_class_name)
        self.assertEqual(diagnostics.class_status, "AMBIGUOUS")

    def test_one_noisy_bus_frame_does_not_flip_car_track(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.89),
                _observation(2, "bus", 0.31),
                _observation(3, "car", 0.87),
                _observation(4, "car", 0.90),
            ],
            TrackingConfig(
                class_stabilization=TrackingConfig().class_stabilization.__class__(
                    minimum_observations=4,
                    minimum_consistency_ratio=0.70,
                    minimum_consecutive_winner_observations=2,
                )
            ),
        )

        self.assertEqual(diagnostics.stable_class_name, "car")
        self.assertFalse(diagnostics.strong_conflict_detected)

    def test_repeated_truck_observations_after_locked_car_trigger_strong_conflict(self) -> None:
        config = TrackingConfig(
            class_stabilization=TrackingConfig().class_stabilization.__class__(
                minimum_observations=3,
                minimum_consistency_ratio=0.60,
                minimum_consecutive_winner_observations=2,
                strong_conflict_min_observations=3,
                recent_window_size=5,
                recent_conflict_minimum_ratio=0.60,
                recent_conflict_minimum_observations=3,
            )
        )
        history = [
            ClassObservation(0, 0.0, datetime(2026, 7, 28, 12, 0, 0), "car", 0.92, (10.0, 10.0, 30.0, 20.0), "car"),
            ClassObservation(1, 1.0 / 30.0, datetime(2026, 7, 28, 12, 0, 1), "car", 0.91, (10.0, 10.0, 30.0, 20.0), "car"),
            ClassObservation(2, 2.0 / 30.0, datetime(2026, 7, 28, 12, 0, 2), "car", 0.90, (10.0, 10.0, 30.0, 20.0), "car"),
            ClassObservation(3, 3.0 / 30.0, datetime(2026, 7, 28, 12, 0, 3), "truck", 0.71, (10.0, 10.0, 30.0, 20.0), "truck"),
            ClassObservation(4, 4.0 / 30.0, datetime(2026, 7, 28, 12, 0, 4), "truck", 0.75, (10.0, 10.0, 30.0, 20.0), "truck"),
            ClassObservation(5, 5.0 / 30.0, datetime(2026, 7, 28, 12, 0, 5), "truck", 0.78, (10.0, 10.0, 30.0, 20.0), "truck"),
        ]
        diagnostics = build_class_diagnostics(
            history=history,
            class_scores={"car": 2.73, "truck": 2.24},
            class_counts={"car": 3, "truck": 3},
            class_max_confidences={"car": 0.92, "truck": 0.78},
            config=config,
            previous_stable_class_name="car",
            previous_class_is_locked=True,
        )

        self.assertTrue(diagnostics.strong_conflict_detected)
        self.assertTrue(diagnostics.split_recommended)

    def test_consecutive_incompatible_segments_mark_mixed_identity_and_block_final_class(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.62),
                _observation(1, "car", 0.60),
                _observation(2, "car", 0.58),
                _observation(3, "car", 0.56),
                _observation(4, "truck", 0.94),
                _observation(5, "truck", 0.92),
                _observation(6, "truck", 0.90),
            ],
            TrackingConfig(
                class_stabilization=TrackingConfig().class_stabilization.__class__(
                    minimum_observations=3,
                    minimum_consistency_ratio=0.60,
                    minimum_consecutive_winner_observations=2,
                )
            ),
        )

        self.assertTrue(diagnostics.mixed_identity_detected)
        self.assertEqual(diagnostics.mixed_identity_classes, ("car", "truck"))
        self.assertEqual(diagnostics.class_status, "MIXED_IDENTITY")
        self.assertTrue(diagnostics.final_class_blocked_due_to_mixed_identity)
        self.assertIsNone(diagnostics.stable_class_name)

    def test_one_noisy_incompatible_frame_does_not_mark_mixed_identity(self) -> None:
        diagnostics = recalculate_track_class(
            [
                _observation(0, "car", 0.91),
                _observation(1, "car", 0.88),
                _observation(2, "truck", 0.44),
                _observation(3, "car", 0.86),
                _observation(4, "car", 0.83),
            ],
            TrackingConfig(),
        )

        self.assertFalse(diagnostics.mixed_identity_detected)
        self.assertFalse(diagnostics.final_class_blocked_due_to_mixed_identity)

    def test_raw_class_history_is_preserved(self) -> None:
        diagnostics = recalculate_track_class(
            [_observation(0, "car", 0.91), _observation(1, "bus", 0.44)],
            TrackingConfig(),
        )

        self.assertEqual(len(diagnostics.raw_class_history), 2)
        self.assertEqual(diagnostics.raw_class_history[1].raw_class_name, "bus")

    def test_existing_configs_without_new_fields_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "tracking.yaml"
            path.write_text(
                json.dumps(
                    {
                        "tracking": {
                            "backend": "supervision_bytetrack",
                            "track_activation_threshold": 0.15,
                            "lost_track_buffer": 30,
                            "minimum_matching_threshold": 0.80,
                            "class_stabilization": {
                                "enabled": True,
                                "minimum_observations": 2,
                                "minimum_winner_margin": 0.20,
                                "lock_after_observations": 5,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_tracking_config(path)

        self.assertEqual(config.class_stabilization.minimum_consistency_ratio, 0.0)
        self.assertEqual(config.class_stabilization.minimum_consecutive_winner_observations, 1)
        self.assertEqual(config.behavior_mode, "experimental_custom")


if __name__ == "__main__":
    unittest.main()
