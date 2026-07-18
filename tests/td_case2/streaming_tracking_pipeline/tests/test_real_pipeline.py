import inspect
import tempfile
import unittest

from tests.td_case2.streaming_tracking_pipeline import real_pipeline
from tests.td_case2.streaming_tracking_pipeline.real_pipeline import (
    RealSequentialTrackingPipeline,
    Step3ArtifactSink,
    finalize_step3_artifacts,
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


class _FakeSource:
    source_id = "cam_a"
    source_fps = 10.0
    frame_width = 100
    frame_height = 80
    target_processing_fps = 5.0
    total_frames = 4

    def __init__(self):
        self.index = 0
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def read(self):
        if self.index >= 2:
            return None
        packet = FramePacket("cam_a", self.index * 2, self.index * 0.2, 10.0, 100, 80, frame=object())
        self.index += 1
        return packet

    def close(self):
        self.closed = True

    def reset(self):
        self.index = 0
        self.opened = False
        self.closed = False


class _FakeDetectionStage:
    def __init__(self):
        self.calls = []

    def process(self, packet):
        self.calls.append(packet.frame_index)
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
        return {"calls": list(self.calls)}


class _FakeTrackingStage:
    def __init__(self):
        self.calls = []
        self.flushes = 0

    def process(self, packet):
        self.calls.append(packet.frame_index)
        return TrackedFramePacket(
            packet.source_id,
            packet.frame_index,
            packet.timestamp_sec,
            packet.frame_width,
            packet.frame_height,
            [
                TrackedObject(
                    1,
                    BoundingBox(1, 1, 20, 20),
                    0.8,
                    2,
                    "car",
                    packet.frame_index,
                    packet.timestamp_sec,
                    source_track_id=10,
                )
            ],
            frame=packet.frame,
        )

    def reset(self):
        self.calls.clear()

    def flush(self):
        self.flushes += 1
        return []

    def to_dict(self):
        return {"calls": list(self.calls), "flushes": self.flushes}


class RealSequentialTrackingPipelineTest(unittest.TestCase):
    def test_pipeline_writes_artifacts_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _FakeSource()
            detector = _FakeDetectionStage()
            tracker = _FakeTrackingStage()
            sink = Step3ArtifactSink(tmpdir)
            pipeline = RealSequentialTrackingPipeline(
                run_id="run_1",
                source=source,
                detection_stage=detector,
                tracking_stage=tracker,
                sink=sink,
                source_path="video.mp4",
                detector_model_path="model.pt",
                tracking_backend="ultralytics_bytetrack",
            )

            report = pipeline.run()
            finalize_step3_artifacts(tmpdir, report, {"source_id": "cam_a"})

            self.assertTrue(source.closed)
            self.assertTrue(report.tracker_flushed)
            self.assertEqual(report.selected_frames_processed, 2)
            self.assertEqual(len(read_jsonl(f"{tmpdir}/02_detections/detection_packets.jsonl")), 2)
            self.assertEqual(len(read_jsonl(f"{tmpdir}/03_tracks/track_observations.jsonl")), 2)
            self.assertEqual(read_json(f"{tmpdir}/reports/step3_real_tracking_report.json")["run_id"], "run_1")

    def test_pipeline_module_does_not_import_async_primitives(self):
        source_text = inspect.getsource(real_pipeline)
        self.assertNotIn("threading", source_text)
        self.assertNotIn("multiprocessing", source_text)
        self.assertNotIn("queue.", source_text)


if __name__ == "__main__":
    unittest.main()
