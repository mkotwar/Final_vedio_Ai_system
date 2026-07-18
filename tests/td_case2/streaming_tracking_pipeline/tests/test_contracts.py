from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.contracts import (
    validate_detection_packet_matches_frame,
    validate_tracked_packet_matches_detection,
)
from tests.td_case2.streaming_tracking_pipeline.mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from tests.td_case2.streaming_tracking_pipeline.schemas import DetectionPacket, FramePacket, TrackedFramePacket
from tests.td_case2.streaming_tracking_pipeline.sequential_pipeline import SequentialContractPipeline
from tests.td_case2.streaming_tracking_pipeline.sources import SyntheticFrameSource


class BadDetectionStage:
    def process(self, packet: FramePacket) -> DetectionPacket:
        return DetectionPacket(packet.source_id, packet.frame_index + 1, packet.timestamp_sec, packet.frame_width, packet.frame_height, frame=packet.frame)


class ContractTests(unittest.TestCase):
    def test_valid_contract_implementations_can_be_used_by_pipeline_and_flush(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=2, source_fps=2, frame_width=10, frame_height=10)
        tracking = DeterministicMockTrackingStage()
        report = SequentialContractPipeline(
            source=source,
            detection_stage=DeterministicMockDetectionStage(),
            tracking_stage=tracking,
        ).run()
        self.assertEqual(report.selected_frames_processed, 2)
        self.assertTrue(report.tracker_flushed)
        self.assertEqual(tracking.flush_count, 1)

    def test_invalid_stage_outputs_are_detected(self) -> None:
        frame = FramePacket("cam", 1, 0.1, 10.0, 100, 100, frame=object())
        bad_detection = DetectionPacket("cam", 2, 0.1, 100, 100, frame=frame.frame)
        with self.assertRaises(ValueError):
            validate_detection_packet_matches_frame(frame, bad_detection)
        detection = DetectionPacket("cam", 1, 0.1, 100, 100, frame=frame.frame)
        bad_tracked = TrackedFramePacket("cam", 1, 0.2, 100, 100, frame=frame.frame)
        with self.assertRaises(ValueError):
            validate_tracked_packet_matches_detection(detection, bad_tracked)

    def test_pipeline_detects_bad_detection_stage(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=1, frame_width=10, frame_height=10)
        with self.assertRaises(ValueError):
            SequentialContractPipeline(
                source=source,
                detection_stage=BadDetectionStage(),
                tracking_stage=DeterministicMockTrackingStage(),
            ).run()
        self.assertTrue(source.closed)

    def test_tracking_stage_frame_regression_is_rejected(self) -> None:
        tracking = DeterministicMockTrackingStage()
        tracking.process(DetectionPacket("cam", 2, 0.2, 100, 100))
        with self.assertRaises(ValueError):
            tracking.process(DetectionPacket("cam", 1, 0.1, 100, 100))

    def test_metadata_preservation_rules(self) -> None:
        frame = FramePacket("cam", 1, 0.1, 10.0, 100, 100, frame={"runtime": True})
        detection = DeterministicMockDetectionStage().process(frame)
        tracked = DeterministicMockTrackingStage().process(detection)
        self.assertEqual(detection.source_id, frame.source_id)
        self.assertEqual(tracked.frame_index, frame.frame_index)
        self.assertIs(tracked.frame, frame.frame)


if __name__ == "__main__":
    unittest.main()

