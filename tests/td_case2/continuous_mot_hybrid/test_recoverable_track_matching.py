from __future__ import annotations

import unittest

from tests.td_case2.continuous_mot_hybrid.recoverable_track_matcher import match_recoverable_tracks
from tests.td_case2.continuous_mot_hybrid.recoverable_track_store import RecoverableTrackSnapshot
from tests.td_case2.continuous_mot_hybrid.recovery_candidate_index import RecoveryCandidateIndex
from tests.td_case2.continuous_mot_hybrid.recovery_scoring import RecoveryScoringConfig, score_recovery_candidate


class RecoverableTrackMatchingTests(unittest.TestCase):
    def _entry(
        self,
        *,
        local_object_id: int = 7,
        tracker_id: str = "vehicle_track_0007",
        last_center: tuple[float, float] = (130.0, 130.0),
    ) -> RecoverableTrackSnapshot:
        return RecoverableTrackSnapshot(
            local_object_id=local_object_id,
            last_tracker_id=tracker_id,
            tracker_id_history=[tracker_id],
            object_family="vehicle",
            stable_class="car",
            class_votes={"car": 4},
            last_detector_supported_bbox=[100.0, 100.0, 160.0, 160.0],
            previous_detector_supported_bbox=[92.0, 100.0, 152.0, 160.0],
            last_center=last_center,
            estimated_velocity=(20.0, 0.0),
            last_timestamp_seconds=1.0,
            last_detector_timestamp_seconds=1.0,
            track_duration_seconds=0.8,
            detector_hit_count=4,
            entry_zone="left",
            likely_exit_zone="interior",
            movement_direction="left_to_right",
            bbox_width=60.0,
            bbox_height=60.0,
            bbox_area=3600.0,
            aspect_ratio=1.0,
            detector_confidence=0.88,
            recovery_expiry_timestamp=2.0,
            recovery_reason="tracker_marked_lost",
            histogram_descriptor=None,
        )

    def test_candidate_index_filters_by_family_and_grid(self) -> None:
        index = RecoveryCandidateIndex(frame_width=400, frame_height=300)
        near = self._entry()
        far = self._entry(local_object_id=8, tracker_id="vehicle_track_0008", last_center=(390.0, 290.0))
        candidates = index.query(
            unmatched_detection={
                "tracker_id": "vehicle_track_0100",
                "family": "vehicle",
                "class_name": "car",
                "bbox_xyxy": [120.0, 110.0, 180.0, 170.0],
                "direction_group": "left_to_right",
            },
            recoverable_entries=[near, far],
            timestamp_seconds=1.4,
        )
        self.assertEqual([item.local_object_id for item in candidates], [7])

    def test_scoring_accepts_strong_candidate(self) -> None:
        attempt = score_recovery_candidate(
            unmatched_detection={
                "tracker_id": "vehicle_track_0100",
                "family": "vehicle",
                "class_name": "car",
                "bbox_xyxy": [118.0, 100.0, 178.0, 160.0],
                "zone": "interior",
                "frame_width": 400,
                "frame_height": 300,
            },
            entry=self._entry(),
            timestamp_seconds=1.4,
            detection_histogram=None,
            config=RecoveryScoringConfig(),
        )
        self.assertEqual(attempt["hard_rejection_reasons"], [])
        self.assertGreaterEqual(attempt["total_score"], 0.78)

    def test_matcher_returns_one_to_one_acceptance(self) -> None:
        detection = {
            "tracker_id": "vehicle_track_0100",
            "family": "vehicle",
            "class_name": "car",
            "bbox_xyxy": [118.0, 100.0, 178.0, 160.0],
            "zone": "interior",
            "frame_width": 400,
            "frame_height": 300,
            "histogram_descriptor": None,
        }
        result = match_recoverable_tracks(
            unmatched_detections=[detection],
            candidate_entries_by_tracker_id={"vehicle_track_0100": [self._entry()]},
            timestamp_seconds=1.4,
            scoring_config=RecoveryScoringConfig(),
        )
        self.assertEqual(len(result.accepted_matches), 1)
        self.assertEqual(result.accepted_matches[0]["new_tracker_id"], "vehicle_track_0100")
        self.assertEqual(result.accepted_matches[0]["proposed_local_object_id"], 7)


if __name__ == "__main__":
    unittest.main()
