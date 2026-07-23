from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.supervision_conversion import (
    SupervisionDetectionConversionError,
    build_supervision_debug_snapshot,
    to_supervision_detections,
)


def _packet(detections: list[VehicleDetection]) -> DetectionPacket:
    return DetectionPacket(
        camera_code="CAM_001",
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
        frame_number=1,
        video_time_seconds=0.0,
        camera_timestamp=datetime(2026, 7, 22, 10, 0, 0),
        frame_width=640,
        frame_height=480,
        detections=detections,
        inference_time_ms=1.0,
        detector_model="model.pt",
        detector_device="cpu",
        source_fps=19.951,
    )


class SupervisionConversionTests(unittest.TestCase):
    def test_valid_packet_converts_to_expected_shapes(self) -> None:
        detections = to_supervision_detections(
            _packet([VehicleDetection(class_id=0, class_name="car", confidence=0.8, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        )
        self.assertEqual(detections.xyxy.shape, (1, 4))
        self.assertEqual(detections.confidence.shape, (1,))
        self.assertEqual(detections.class_id.shape, (1,))

    def test_empty_packet_converts_to_empty_arrays(self) -> None:
        detections = to_supervision_detections(_packet([]))
        self.assertEqual(detections.xyxy.shape, (0, 4))
        self.assertEqual(detections.confidence.shape, (0,))
        self.assertEqual(detections.class_id.shape, (0,))

    def test_invalid_box_is_rejected(self) -> None:
        with self.assertRaises(SupervisionDetectionConversionError):
            to_supervision_detections(
                _packet([VehicleDetection(class_id=0, class_name="car", confidence=0.8, bbox_xyxy=(5.0, 2.0, 5.0, 12.0))])
            )

    def test_debug_snapshot_preserves_alignment(self) -> None:
        snapshot = build_supervision_debug_snapshot(
            _packet([VehicleDetection(class_id=3, class_name="motorcycle", confidence=0.55, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))])
        )
        self.assertEqual(snapshot.input_detection_count, 1)
        self.assertEqual(snapshot.input_class_ids, [3])
        self.assertEqual(snapshot.input_confidences, [0.55])


if __name__ == "__main__":
    unittest.main()
