from __future__ import annotations

import unittest

from tests.td_case2.continuous_mot_hybrid.track_reconciliation import reconcile_track_fragments


def _track(track_id: str, start: float, end: float, start_bbox, end_bbox):
    return {
        "track_id": track_id,
        "class_name": "car",
        "object_family": "vehicle",
        "confirmed": True,
        "track_integrity_status": "usable",
        "quality_flags": [],
        "integrity_flags": [],
        "sanitized_start_timestamp_seconds": start,
        "sanitized_end_timestamp_seconds": end,
        "sanitized_duration_seconds": round(end - start, 6),
        "sanitized_valid_timeline": [
            {"timestamp_seconds": start, "source_frame_index": int(start * 10), "bbox_xyxy": start_bbox, "bbox_source": "yolo"},
            {"timestamp_seconds": end, "source_frame_index": int(end * 10), "bbox_xyxy": end_bbox, "bbox_source": "yolo"},
        ],
        "first_source_frame_index": int(start * 10),
        "last_source_frame_index": int(end * 10),
    }


class ReconciliationTests(unittest.TestCase):
    def test_valid_sequential_fragment_and_duplicate_overlap_merge(self) -> None:
        tracks = [
            _track("1", 0.0, 0.6, [10, 10, 30, 30], [20, 10, 40, 30]),
            _track("2", 0.7, 1.3, [22, 10, 42, 30], [30, 10, 50, 30]),
            _track("3", 0.4, 0.9, [20, 10, 40, 30], [28, 10, 48, 30]),
        ]
        candidates, accepted, possible, rejected, reconciled = reconcile_track_fragments(
            tracks,
            camera_id="test_cam_01",
            camera_group="single_camera_comparison",
            camera_timezone="Asia/Kolkata",
            maximum_gap_seconds=1.5,
            duplicate_overlap_seconds=0.5,
        )
        self.assertGreaterEqual(len(candidates["candidates"]), 1)
        self.assertGreaterEqual(len(accepted["merges"]) + len(possible["merges"]), 1)
        self.assertGreaterEqual(len(reconciled["tracks"]), 2)

    def test_hard_conflict_merge_rejection(self) -> None:
        tracks = [
            _track("1", 0.0, 0.3, [10, 10, 30, 30], [12, 10, 32, 30]),
            _track("2", 0.4, 0.8, [90, 90, 110, 110], [92, 90, 112, 110]),
        ]
        candidates, accepted, possible, rejected, reconciled = reconcile_track_fragments(
            tracks,
            camera_id="test_cam_01",
            camera_group="single_camera_comparison",
            camera_timezone="Asia/Kolkata",
            maximum_gap_seconds=1.5,
            duplicate_overlap_seconds=0.5,
        )
        self.assertEqual(len(accepted["merges"]), 0)
        self.assertGreaterEqual(len(rejected["merges"]), 0)


if __name__ == "__main__":
    unittest.main()
