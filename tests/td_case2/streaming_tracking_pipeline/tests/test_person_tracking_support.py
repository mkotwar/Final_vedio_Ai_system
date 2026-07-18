from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.config import ObjectTrackingConfig
from tests.td_case2.streaming_tracking_pipeline.multi_model_detection import combine_detection_packets
from tests.td_case2.streaming_tracking_pipeline.person_tracking_support import (
    build_person_tracking_audit,
    object_group_for_detection,
    validate_object_tracking_config,
)
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket
from tests.td_case2.streaming_tracking_pipeline.serialization import write_jsonl


class PersonTrackingSupportTests(unittest.TestCase):
    def test_class_mapping_produces_person_group(self) -> None:
        self.assertEqual("person", object_group_for_detection("person"))
        self.assertEqual("vehicle", object_group_for_detection("car"))

    def test_combined_detector_mode_accepts_person_mapping(self) -> None:
        report = validate_object_tracking_config(
            ObjectTrackingConfig(enable_person_tracking=True, track_object_groups=("vehicle", "person")),
            vehicle_model_names={0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
            require_existing_paths=False,
        )

        self.assertTrue(report["supports_person"])
        self.assertTrue(report["supports_vehicle"])

    def test_dual_detector_mode_requires_person_detector_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "person_tracking_enabled_but_no_person_detector"):
            validate_object_tracking_config(
                ObjectTrackingConfig(enable_person_tracking=True, detection_mode="dual", track_object_groups=("vehicle", "person")),
                vehicle_model_names={2: "car"},
                person_model_names={1: "dog"},
                require_existing_paths=False,
            )

    def test_dual_detector_merge_preserves_person_and_vehicle_detections_for_bytetrack(self) -> None:
        frame = _frame()
        vehicle_packet = DetectionPacket("s", 1, 0.1, 100, 100, [_detection("car", "vehicle_model", 2)])
        person_packet = DetectionPacket("s", 1, 0.1, 100, 100, [_detection("person", "person_model", 0)])

        merged = combine_detection_packets(frame, [vehicle_packet, person_packet])

        self.assertEqual(["vehicle", "person"], [item.object_group for item in merged.detections])
        self.assertEqual(["car", "person"], [item.class_name for item in merged.detections])

    def test_audit_reports_zero_person_root_causes_from_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(run_dir / "04_lifecycle" / "completed_tracks.jsonl", [{"source_id": "s", "track_id": 1, "track_generation": 0, "last_class_name": "car", "status": "completed"}])
            write_jsonl(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", [{"source_id": "s", "track_id": 1, "track_generation": 0, "object_class": "car", "object_group": "vehicle"}])

            audit = build_person_tracking_audit(
                run_dir=run_dir,
                vehicle_model_path=None,
                person_model_path=None,
                output_path=run_dir / "person_tracking_audit.json",
            )

            self.assertEqual(0, audit["person_records_written"])
            self.assertIn("person_detector_not_configured", audit["root_causes"])
            self.assertTrue((run_dir / "person_tracking_audit.json").exists())


def _frame() -> FramePacket:
    return FramePacket("s", 1, 0.1, 10.0, 100, 100, frame=object())


def _detection(class_name: str, source: str, class_id: int) -> DetectionRecord:
    return DetectionRecord(
        bbox=BoundingBox(1, 1, 20, 40),
        confidence=0.8,
        class_id=class_id,
        class_name=class_name,
        raw_class_id=class_id,
        raw_class_name=class_name,
        normalized_class_name=class_name,
        object_group="person" if class_name == "person" else "vehicle",
        detector_source=source,
    )


if __name__ == "__main__":
    unittest.main()
