import unittest

from tests.td_case2.streaming_tracking_pipeline.schemas import (
    BoundingBox,
    DetectionPacket,
    DetectionRecord,
    TrackedFramePacket,
    TrackedObject,
)
from tests.td_case2.streaming_tracking_pipeline.tracking_metrics import TrackingMetricsAccumulator


def _detections(index, count=1):
    return DetectionPacket(
        source_id="cam_a",
        frame_index=index,
        timestamp_sec=index / 5.0,
        frame_width=100,
        frame_height=80,
        detections=[DetectionRecord(BoundingBox(1, 1, 20, 20), 0.8, 2, "car") for _ in range(count)],
    )


def _tracked(index, track_ids):
    return TrackedFramePacket(
        source_id="cam_a",
        frame_index=index,
        timestamp_sec=index / 5.0,
        frame_width=100,
        frame_height=80,
        tracks=[
            TrackedObject(
                track_id=track_id,
                source_track_id=f"native_{track_id}",
                bbox=BoundingBox(1, 1, 20, 20),
                confidence=0.8,
                class_id=2,
                class_name="car",
                frame_index=index,
                timestamp_sec=index / 5.0,
            )
            for track_id in track_ids
        ],
    )


class TrackingMetricsAccumulatorTest(unittest.TestCase):
    def test_counts_tracks_durations_and_gaps(self):
        metrics = TrackingMetricsAccumulator(expected_physical_objects=1)
        metrics.update(_detections(0), _tracked(0, [1]))
        metrics.update(_detections(2), _tracked(2, [1, 2]))
        metrics.update(_detections(3, count=0), _tracked(3, []))

        payload = metrics.to_dict()

        self.assertEqual(payload["total_frames_processed"], 3)
        self.assertEqual(payload["unique_track_ids"], 2)
        self.assertEqual(payload["maximum_simultaneous_tracks"], 2)
        self.assertEqual(payload["tracks_under_0_5_sec"], 2)
        self.assertEqual(payload["track_gaps_by_id"]["1"], 0)
        self.assertEqual(payload["unique_tracks_minus_expected_physical_objects"], 1)


if __name__ == "__main__":
    unittest.main()
