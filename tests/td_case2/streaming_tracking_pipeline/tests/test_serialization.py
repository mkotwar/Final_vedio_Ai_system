from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionRecord, TrackStatus
from tests.td_case2.streaming_tracking_pipeline.serialization import (
    dataclass_to_dict,
    read_json,
    read_jsonl,
    to_json_safe,
    write_json,
    write_jsonl,
)


class SampleEnum(str, Enum):
    VALUE = "value"


@dataclass
class Packet:
    path: Path
    record: DetectionRecord
    values: tuple[int, int]
    status: TrackStatus
    frame: object | None = None


class SerializationTests(unittest.TestCase):
    def test_nested_dataclass_conversion_and_runtime_omission(self) -> None:
        packet = Packet(
            path=Path("a/b.jpg"),
            record=DetectionRecord(BoundingBox(0, 1, 10, 11), 0.9, 2, "car"),
            values=(1, 2),
            status=TrackStatus.CONFIRMED,
            frame=object(),
        )
        payload = dataclass_to_dict(packet)
        self.assertEqual(payload["path"], "a\\b.jpg" if "\\" in str(Path("a/b.jpg")) else "a/b.jpg")
        self.assertEqual(payload["record"]["bbox"]["x2"], 10)
        self.assertEqual(payload["values"], [1, 2])
        self.assertEqual(payload["status"], "confirmed")
        self.assertNotIn("frame", payload)

    def test_enum_and_path_conversion(self) -> None:
        self.assertEqual(to_json_safe(SampleEnum.VALUE), "value")
        self.assertEqual(to_json_safe(Path("x")), "x")

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "payload.json"
            write_json(path, {"b": 2, "a": 1})
            self.assertEqual(read_json(path), {"a": 1, "b": 2})

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"b": 2}])

    def test_unsupported_runtime_object_failure(self) -> None:
        with self.assertRaises(TypeError):
            to_json_safe(object(), exclude_runtime_fields=False)


if __name__ == "__main__":
    unittest.main()
