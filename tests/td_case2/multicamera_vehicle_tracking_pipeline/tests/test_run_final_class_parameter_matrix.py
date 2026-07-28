from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.run_final_class_parameter_matrix import (
    _class_stabilization_override,
    _class_threshold_override,
    _split_override,
)


class RunFinalClassParameterMatrixTests(unittest.TestCase):
    def test_class_threshold_override_builds_per_class_payload(self) -> None:
        payload = _class_threshold_override(
            default_threshold=0.40,
            classes={"car": 0.40, "truck": 0.75, "bus": 0.75, "motorcycle": 0.32, "3wheeler": 0.40},
            inference_floor=0.32,
        )

        self.assertEqual(payload["confidence_threshold"], 0.32)
        self.assertTrue(payload["class_confidence_thresholds"]["enabled"])
        self.assertEqual(payload["class_confidence_thresholds"]["classes"]["truck"], 0.75)

    def test_class_stabilization_override_builds_requested_thresholds(self) -> None:
        payload = _class_stabilization_override(
            minimum_observations=4,
            minimum_consistency_ratio=0.70,
            minimum_consecutive_winner_observations=3,
            minimum_winner_margin=0.20,
        )

        self.assertEqual(payload["class_stabilization"]["minimum_observations"], 4)
        self.assertEqual(payload["class_stabilization"]["minimum_consistency_ratio"], 0.70)
        self.assertEqual(payload["class_stabilization"]["minimum_consecutive_winner_observations"], 3)

    def test_split_override_can_disable_or_tune_split(self) -> None:
        disabled = _split_override(enabled=False)
        tuned = _split_override(
            enabled=True,
            minimum_consecutive_conflicting_observations=2,
            minimum_average_conflict_confidence=0.60,
            maximum_iou_for_split=0.15,
            minimum_normalized_center_distance_for_split=0.40,
            maximum_width_ratio_for_split=2.25,
            maximum_height_ratio_for_split=2.25,
        )

        self.assertFalse(disabled["class_conflict_split"]["enabled"])
        self.assertTrue(tuned["class_conflict_split"]["enabled"])
        self.assertEqual(tuned["class_conflict_split"]["minimum_consecutive_conflicting_observations"], 2)
        self.assertEqual(tuned["class_conflict_split"]["maximum_iou_for_split"], 0.15)
