import inspect
import tempfile
import unittest

from tests.td_case2.streaming_tracking_pipeline import lifecycle_pipeline
from tests.td_case2.streaming_tracking_pipeline.config import TrackLifecycleConfig
from tests.td_case2.streaming_tracking_pipeline.lifecycle import TrackLifecycleManager
from tests.td_case2.streaming_tracking_pipeline.lifecycle_pipeline import (
    LifecycleArtifactSink,
    SequentialLifecycleTrackingPipeline,
    finalize_step4_artifacts,
)
from tests.td_case2.streaming_tracking_pipeline.schemas import (
    BoundingBox,
    DetectionPacket,
    DetectionRecord,
    FramePacket,
    TrackedFramePacket,
    TrackedObject,
)
from tests.td_case2.streaming_tracking_pipeline.serialization import read_json, read_jsonl


class FakeSource:
    source_id = "cam_a"
    source_fps = 10.0
    frame_width = 100
    frame_height = 80
    target_processing_fps = 5.0
    total_frames = 3

    def __init__(self, log):
        self.log = log
        self.index = 0
        self.closed = False

    def open(self):
        self.log.append("source.open")

    def read(self):
        if self.index >= 3:
            self.log.append("source.eos")
            return None
        self.log.append(f"source.read.{self.index}")
        frame = object()
        packet = FramePacket("cam_a", self.index, self.index * 0.5, 10.0, 100, 80, frame)
        self.index += 1
        return packet

    def close(self):
        self.log.append("source.close")
        self.closed = True

    def reset(self):
        self.index = 0
        self.closed = False


class FakeDetector:
    def __init__(self, log):
        self.log = log

    def process(self, packet):
        self.log.append(f"detect.{packet.frame_index}")
        return DetectionPacket(
            packet.source_id,
            packet.frame_index,
            packet.timestamp_sec,
            packet.frame_width,
            packet.frame_height,
            [DetectionRecord(BoundingBox(1, 1, 20, 20), 0.8, 2, "car")],
            frame=packet.frame,
        )

    def to_dict(self):
        return {"fake": "detector"}


class FakeTracker:
    def __init__(self, log):
        self.log = log
        self.flushed = False

    def process(self, packet):
        self.log.append(f"track.{packet.frame_index}")
        tracks = []
        if packet.frame_index < 2:
            tracks.append(
                TrackedObject(
                    1,
                    BoundingBox(1, 1, 20, 20),
                    0.8,
                    2,
                    "car",
                    packet.frame_index,
                    packet.timestamp_sec,
                    source_track_id="native_1",
                )
            )
        return TrackedFramePacket(packet.source_id, packet.frame_index, packet.timestamp_sec, packet.frame_width, packet.frame_height, tracks, frame=packet.frame)

    def reset(self):
        pass

    def flush(self):
        self.log.append("tracker.flush")
        self.flushed = True
        return []

    def to_dict(self):
        return {"flushed": self.flushed}


class ExplodingLifecycleManager(TrackLifecycleManager):
    def update(self, packet):
        if packet.frame_index == 1:
            raise RuntimeError("lifecycle boom")
        return super().update(packet)


class LifecyclePipelineTest(unittest.TestCase):
    def build_pipeline(self, tmpdir, log, manager=None):
        return SequentialLifecycleTrackingPipeline(
            run_id="run_1",
            source=FakeSource(log),
            detection_stage=FakeDetector(log),
            tracking_stage=FakeTracker(log),
            lifecycle_manager=manager or TrackLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2, maximum_lost_processed_frames=5)),
            sink=LifecycleArtifactSink(tmpdir),
            source_path="video.mp4",
            detector_model_path="model.pt",
            tracking_backend="ultralytics_bytetrack",
        )

    def test_exact_order_artifacts_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = []
            pipeline = self.build_pipeline(tmpdir, log)
            report = pipeline.run()
            finalize_step4_artifacts(tmpdir, report, {"source_id": "cam_a"})

            self.assertEqual(
                log,
                [
                    "source.open",
                    "source.read.0",
                    "detect.0",
                    "track.0",
                    "source.read.1",
                    "detect.1",
                    "track.1",
                    "source.read.2",
                    "detect.2",
                    "track.2",
                    "source.eos",
                    "tracker.flush",
                    "source.close",
                ],
            )
            self.assertTrue(report.tracker_flushed)
            self.assertTrue(report.lifecycle_flushed)
            self.assertEqual(report.lifecycle_metrics["tracks_completed_video_end"], 1)
            self.assertEqual(len(read_jsonl(f"{tmpdir}/04_lifecycle/lifecycle_events.jsonl")), 7)
            self.assertEqual(len(read_jsonl(f"{tmpdir}/04_lifecycle/completed_tracks.jsonl")), 1)
            self.assertEqual(read_json(f"{tmpdir}/reports/step4_lifecycle_report.json")["run_id"], "run_1")
            self.assertNotIn("frame", read_jsonl(f"{tmpdir}/03_tracks/tracked_frame_packets.jsonl")[0])

    def test_cleanup_after_lifecycle_error_and_partial_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = []
            pipeline = self.build_pipeline(
                tmpdir,
                log,
                manager=ExplodingLifecycleManager(TrackLifecycleConfig(minimum_confirmation_observations=2)),
            )
            with self.assertRaisesRegex(RuntimeError, "lifecycle boom"):
                pipeline.run()
            self.assertIn("source.close", log)
            self.assertTrue(pipeline.sink.closed)
            self.assertIsNotNone(pipeline.last_report)
            self.assertIn("lifecycle boom", pipeline.last_report.errors)

    def test_deterministic_repeated_run_and_no_async_primitives(self):
        summaries = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                log = []
                report = self.build_pipeline(tmpdir, log).run()
                summaries.append(report.lifecycle_summary)
        self.assertEqual(summaries[0], summaries[1])
        source_text = inspect.getsource(lifecycle_pipeline)
        self.assertNotIn("threading", source_text)
        self.assertNotIn("multiprocessing", source_text)
        self.assertNotIn("queue.", source_text)


if __name__ == "__main__":
    unittest.main()
