from __future__ import annotations

import unittest

from tests.td_case2.continuous_mot_hybrid.track_integrity import sanitize_tracks


class IntegrityTests(unittest.TestCase):
    def test_invalid_geometry_and_boundary_rejection(self) -> None:
        track_rows = [
            {
                "track_id": "vehicle_track_0001",
                "class_name": "car",
                "object_family": "vehicle",
                "confirmed": True,
                "start_timestamp_seconds": 0.0,
                "end_timestamp_seconds": 0.6,
                "trajectory": [
                    {"timestamp_seconds": 0.0, "source_frame_index": 0, "bbox_xyxy": [1, 1, 10, 10], "bbox_source": "yolo", "time_since_update_seconds": 0.0},
                    {"timestamp_seconds": 0.3, "source_frame_index": 3, "bbox_xyxy": [110, 110, 120, 120], "bbox_source": "mot_predicted", "time_since_update_seconds": 0.6},
                ],
            }
        ]
        sanitized, report, events = sanitize_tracks(
            track_rows,
            frame_width=120,
            frame_height=120,
            maximum_active_detector_gap_seconds=0.5,
            maximum_visual_bridge_seconds=0.3,
            frozen_window_seconds=0.5,
        )
        self.assertEqual(report["boundary_stuck_tracks"], 1)
        self.assertTrue(any("impossible_center_jump" in item["flags"] for item in events))


if __name__ == "__main__":
    unittest.main()

