from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionRecord, DetectionPacket, FramePacket, TrackedObject


class MockStageTests(unittest.TestCase):
    def frame(self, frame_index: int = 1, source_id: str = "cam") -> FramePacket:
        return FramePacket(source_id, frame_index, frame_index / 10.0, 10.0, 100, 100, frame={"frame": frame_index})

    def test_deterministic_detection_ordering_empty_and_metadata(self) -> None:
        detections = [
            DetectionRecord(BoundingBox(20, 10, 40, 30), 0.9, 2, "car"),
            DetectionRecord(BoundingBox(10, 10, 30, 30), 0.5, 1, "person"),
            DetectionRecord(BoundingBox(10, 10, 30, 30), 0.8, 1, "person"),
        ]
        stage = DeterministicMockDetectionStage(detections_by_frame={1: detections})
        packet = self.frame()
        output = stage.process(packet)
        self.assertEqual([item.confidence for item in output.detections], [0.8, 0.5, 0.9])
        self.assertIs(output.frame, packet.frame)
        self.assertEqual(stage.processed_frame_indices, [1])
        self.assertEqual(DeterministicMockDetectionStage().process(packet).detections, [])

    def test_detection_out_of_bounds_rejected(self) -> None:
        stage = DeterministicMockDetectionStage(
            detections_by_frame={1: [DetectionRecord(BoundingBox(0, 0, 200, 10), 0.9, 1, "person")]}
        )
        with self.assertRaises(ValueError):
            stage.process(self.frame())

    def test_mock_tracker_enforcement_reset_and_flush(self) -> None:
        frame = self.frame()
        detection_packet = DetectionPacket("cam", 1, 0.1, 100, 100, frame=frame.frame)
        track = TrackedObject(1, BoundingBox(1, 1, 10, 10), 0.9, 1, "person", 1, 0.1, source_track_id="person_track_0001")
        stage = DeterministicMockTrackingStage(tracks_by_frame={1: [track]})
        output = stage.process(detection_packet)
        self.assertEqual(output.tracks[0].source_track_id, "person_track_0001")
        self.assertIs(output.frame, detection_packet.frame)
        with self.assertRaises(ValueError):
            stage.process(detection_packet)
        self.assertEqual(stage.flush(), [])
        stage.reset()
        self.assertEqual(stage.reset_count, 1)
        stage.process(detection_packet)

    def test_tracker_frame_regression_and_source_switch_rejected(self) -> None:
        stage = DeterministicMockTrackingStage()
        stage.process(DetectionPacket("cam", 2, 0.2, 100, 100))
        with self.assertRaises(ValueError):
            stage.process(DetectionPacket("cam", 1, 0.1, 100, 100))
        stage.reset()
        stage.process(DetectionPacket("cam", 1, 0.1, 100, 100))
        with self.assertRaises(ValueError):
            stage.process(DetectionPacket("other", 2, 0.2, 100, 100))


if __name__ == "__main__":
    unittest.main()

