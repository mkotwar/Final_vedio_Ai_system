from __future__ import annotations

import inspect
import unittest

from tests.td_case2.streaming_tracking_pipeline.adapters import InMemoryPacketSink
from tests.td_case2.streaming_tracking_pipeline.mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionRecord, FramePacket, TrackedObject
from tests.td_case2.streaming_tracking_pipeline.sequential_pipeline import SequentialContractPipeline
from tests.td_case2.streaming_tracking_pipeline.sources import SyntheticFrameSource


class FailingDetectionStage:
    def process(self, packet: FramePacket):
        raise RuntimeError("detector failed")


class FailingTrackingStage(DeterministicMockTrackingStage):
    def process(self, packet):
        raise RuntimeError("tracker failed")


class SequentialPipelineTests(unittest.TestCase):
    def build_pipeline(self):
        source = SyntheticFrameSource(
            source_id="cam",
            total_frames=6,
            source_fps=6.0,
            frame_width=100,
            frame_height=100,
            target_processing_fps=3.0,
        )
        detections = {
            0: [DetectionRecord(BoundingBox(1, 1, 10, 10), 0.9, 1, "person")],
            2: [DetectionRecord(BoundingBox(2, 2, 11, 11), 0.8, 1, "person")],
        }

        def track_factory(packet):
            if packet.frame_index not in detections:
                return []
            detection = detections[packet.frame_index][0]
            return [
                TrackedObject(
                    1,
                    detection.bbox,
                    detection.confidence,
                    detection.class_id,
                    detection.class_name,
                    packet.frame_index,
                    packet.timestamp_sec,
                    source_track_id="person_track_0001",
                )
            ]

        sink = InMemoryPacketSink()
        pipeline = SequentialContractPipeline(
            source=source,
            detection_stage=DeterministicMockDetectionStage(detections_by_frame=detections),
            tracking_stage=DeterministicMockTrackingStage(track_factory=track_factory),
            sink=sink,
        )
        return pipeline, source, sink

    def test_complete_successful_run_counts_order_timestamps_and_sink(self) -> None:
        pipeline, source, sink = self.build_pipeline()
        report = pipeline.run()
        self.assertEqual([item["frame_index"] for item in sink.frames], [0, 2, 4])
        self.assertEqual([item["timestamp_sec"] for item in sink.frames], [0.0, 2 / 6.0, 4 / 6.0])
        self.assertEqual(report.selected_frames_processed, 3)
        self.assertEqual(report.detection_packets_created, 3)
        self.assertEqual(report.tracked_packets_created, 3)
        self.assertEqual(report.total_detections, 2)
        self.assertEqual(report.total_tracked_objects, 2)
        self.assertTrue(report.end_of_stream_reached)
        self.assertTrue(report.source_closed)
        self.assertTrue(sink.closed)
        self.assertNotIn("frame", sink.frames[0])
        self.assertNotIn("frame", sink.detections[0])
        self.assertNotIn("frame", sink.tracked_frames[0])
        self.assertTrue(report.frame_order_valid)
        self.assertTrue(source.closed)

    def test_source_cleanup_after_detector_failure(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=1, frame_width=10, frame_height=10)
        sink = InMemoryPacketSink()
        with self.assertRaises(RuntimeError):
            SequentialContractPipeline(
                source=source,
                detection_stage=FailingDetectionStage(),
                tracking_stage=DeterministicMockTrackingStage(),
                sink=sink,
            ).run()
        self.assertTrue(source.closed)
        self.assertTrue(sink.closed)

    def test_sink_cleanup_after_tracker_failure(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=1, frame_width=10, frame_height=10)
        sink = InMemoryPacketSink()
        with self.assertRaises(RuntimeError):
            SequentialContractPipeline(
                source=source,
                detection_stage=DeterministicMockDetectionStage(),
                tracking_stage=FailingTrackingStage(),
                sink=sink,
            ).run()
        self.assertTrue(source.closed)
        self.assertTrue(sink.closed)

    def test_no_threading_or_queues_imports(self) -> None:
        import tests.td_case2.streaming_tracking_pipeline.sequential_pipeline as module

        source_text = inspect.getsource(module)
        self.assertNotIn("threading", source_text)
        self.assertNotIn("queue", source_text)

    def test_repeatable_output_after_resets(self) -> None:
        pipeline, source, _sink = self.build_pipeline()
        first = pipeline.run().to_dict()
        source.reset()
        pipeline.tracking_stage.reset()
        second_sink = InMemoryPacketSink()
        pipeline.sink = second_sink
        second = pipeline.run().to_dict()
        for key in ("selected_frames_processed", "first_frame_index", "last_frame_index", "total_detections", "total_tracked_objects"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
