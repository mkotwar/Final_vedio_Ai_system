from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.adapters import (
    InMemoryPacketSink,
    JsonlPacketSink,
    TrackIdNormalizer,
    td_case2_detection_record_to_schema,
    td_case2_frame_group_to_detection_packet,
    td_case2_track_detection_to_tracked_object,
    tracked_object_to_step05_detection_dict,
)
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, FramePacket, TrackedObject
from tests.td_case2.streaming_tracking_pipeline.serialization import read_jsonl


class AdapterTests(unittest.TestCase):
    def test_tracker_id_normalization_policy(self) -> None:
        normalizer = TrackIdNormalizer()
        self.assertEqual(normalizer.normalize(1), 1)
        string_one = normalizer.normalize("1")
        self.assertNotEqual(string_one, 1)
        self.assertEqual(normalizer.normalize("vehicle_track_0001"), 3)
        self.assertEqual(normalizer.normalize("vehicle_track_0001"), 3)
        self.assertEqual(normalizer.get_source_id(3), "vehicle_track_0001")
        with self.assertRaises(ValueError):
            normalizer.normalize("")
        with self.assertRaises(ValueError):
            normalizer.normalize(["bad"])  # type: ignore[arg-type]
        normalizer.reset()
        self.assertIsNone(normalizer.get_source_id(1))

    def test_collision_prevention(self) -> None:
        normalizer = TrackIdNormalizer()
        self.assertEqual(normalizer.normalize(1), 1)
        self.assertEqual(normalizer.normalize("vehicle_track_0001"), 2)

    def test_step03_artifact_conversion_and_invalid_handling(self) -> None:
        detection = {
            "detection_id": "frame_000001_object_001",
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.91,
            "bbox_xyxy": [1, 2, 30, 40],
            "bbox_area_ratio": 0.1,
            "crop_path": "03_yolo_object_crops/crop.jpg",
        }
        record = td_case2_detection_record_to_schema(detection)
        self.assertEqual(record.class_name, "car")
        frame_group = {"frame_idx": 5, "timestamp_seconds": 1.25, "detections": [detection]}
        packet = td_case2_frame_group_to_detection_packet(frame_group, source_id="cam", frame_width=100, frame_height=80)
        self.assertEqual(packet.frame_index, 5)
        self.assertEqual(packet.detections[0].bbox.to_xyxy(), [1.0, 2.0, 30.0, 40.0])
        with self.assertRaises(ValueError):
            td_case2_detection_record_to_schema({"class_name": "car"})

    def test_step04b_artifact_conversion_source_id_and_reverse_compatibility(self) -> None:
        normalizer = TrackIdNormalizer()
        row = {
            "track_id": "vehicle_track_0001",
            "frame_idx": 5,
            "timestamp_seconds": 1.25,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.91,
            "bbox_xyxy": [1, 2, 30, 40],
        }
        track = td_case2_track_detection_to_tracked_object(row, normalizer=normalizer)
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.source_track_id, "vehicle_track_0001")
        compat = tracked_object_to_step05_detection_dict(track, frame_id="frame_000005")
        self.assertEqual(compat["track_id"], "vehicle_track_0001")
        self.assertEqual(compat["frame_id"], "frame_000005")
        with self.assertRaises(ValueError):
            td_case2_track_detection_to_tracked_object({"track_id": "x"}, normalizer=normalizer)

    def test_sinks_exclude_runtime_frame_and_jsonl_round_trip(self) -> None:
        packet = FramePacket("cam", 1, 0.1, 10.0, 100, 100, frame=object())
        memory = InMemoryPacketSink()
        memory.write_frame(packet)
        memory.close()
        memory.close()
        self.assertNotIn("frame", memory.frames[0])
        with tempfile.TemporaryDirectory() as directory:
            sink = JsonlPacketSink(Path(directory))
            sink.write_frame(packet)
            sink.close()
            records = read_jsonl(Path(directory) / "frame_packets.jsonl")
        self.assertEqual(records[0]["frame_index"], 1)
        self.assertNotIn("frame", records[0])


if __name__ == "__main__":
    unittest.main()
