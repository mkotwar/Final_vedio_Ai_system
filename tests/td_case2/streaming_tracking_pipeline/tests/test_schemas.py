from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.schemas import (
    BoundingBox,
    ColourResult,
    ColourStatus,
    CropCandidate,
    CropQualityMetrics,
    DetectionRecord,
    FramePacket,
    ObjectRecord,
    PlateResult,
    PlateStatus,
    TrackCompletionReason,
    TrackRecord,
    TrackStatus,
)


def quality() -> CropQualityMetrics:
    return CropQualityMetrics(
        detection_confidence=0.8,
        bbox_area_ratio=0.2,
        edge_touching=False,
        combined_score=0.7,
    )


def crop(*, primary: bool = False, fallback: bool = False) -> CropCandidate:
    return CropCandidate(
        track_id=1,
        frame_index=10,
        timestamp_sec=1.0,
        bbox=BoundingBox(10, 20, 50, 80),
        full_frame_path="frames/001.jpg",
        vehicle_crop_path="crops/001.jpg",
        quality=quality(),
        is_primary=primary,
        is_fallback=fallback,
    )


class SchemaTests(unittest.TestCase):
    def test_valid_and_invalid_bounding_boxes(self) -> None:
        box = BoundingBox(1, 2, 5, 8)
        self.assertEqual(box.to_xyxy(), [1, 2, 5, 8])
        with self.assertRaises(ValueError):
            BoundingBox(1, 2, 1, 8)
        with self.assertRaises(ValueError):
            BoundingBox(float("nan"), 2, 5, 8)

    def test_bbox_metrics_clipping_and_edge_touching(self) -> None:
        box = BoundingBox(1, 2, 5, 8)
        self.assertEqual(box.width, 4)
        self.assertEqual(box.height, 6)
        self.assertEqual(box.area, 24)
        self.assertEqual(box.center, (3, 5))
        clipped = BoundingBox(-5, -5, 20, 20).clip(10, 12)
        self.assertEqual(clipped.to_xyxy(), [0.0, 0.0, 10.0, 12.0])
        self.assertTrue(BoundingBox(0, 5, 10, 20).touches_frame_edge(100, 100, 0.02))

    def test_confidence_validation(self) -> None:
        DetectionRecord(BoundingBox(0, 0, 1, 1), 1.0, 0, "car")
        with self.assertRaises(ValueError):
            DetectionRecord(BoundingBox(0, 0, 1, 1), 1.2, 0, "car")

    def test_frame_index_validation(self) -> None:
        FramePacket("cam", 0, 0.0, 25.0, 1920, 1080, frame=object())
        with self.assertRaises(ValueError):
            FramePacket("cam", -1, 0.0, 25.0, 1920, 1080)

    def test_track_duration_and_dominant_class(self) -> None:
        record = TrackRecord(
            source_id="cam",
            track_id=1,
            status=TrackStatus.CONFIRMED,
            first_seen_frame=10,
            last_seen_frame=20,
            first_seen_sec=1.0,
            last_seen_sec=3.5,
            observation_count=4,
            missed_frame_count=0,
            class_votes={"truck": 2, "car": 2},
        )
        self.assertEqual(record.duration_sec, 2.5)
        self.assertEqual(record.dominant_class, "car")

    def test_completed_track_with_completion_reason(self) -> None:
        record = TrackRecord(
            source_id="cam",
            track_id=1,
            status="completed",
            first_seen_frame=0,
            last_seen_frame=1,
            first_seen_sec=0.0,
            last_seen_sec=0.1,
            observation_count=2,
            missed_frame_count=0,
            completion_reason="video_ended",
        )
        self.assertEqual(record.status, TrackStatus.COMPLETED)
        self.assertEqual(record.completion_reason, TrackCompletionReason.VIDEO_ENDED)

    def test_crop_primary_fallback_conflict(self) -> None:
        with self.assertRaises(ValueError):
            crop(primary=True, fallback=True)

    def test_object_record_serialization(self) -> None:
        record = ObjectRecord(
            source_id="cam",
            track_id=1,
            object_class="car",
            first_seen_frame=10,
            last_seen_frame=20,
            first_seen_sec=1.0,
            last_seen_sec=2.0,
            observation_count=5,
            track_status=TrackStatus.COMPLETED,
            completion_reason=TrackCompletionReason.VIDEO_ENDED,
            primary_crops=[crop(primary=True)],
            fallback_crop=crop(fallback=True),
            plate=PlateResult(normalized_text="DL12AB1234", confidence=0.9, verified=True, status=PlateStatus.VERIFIED),
            colour=ColourResult(normalized_colour="white", confidence=0.8, status=ColourStatus.VERIFIED),
            full_frame_paths=["frames/001.jpg"],
            vehicle_crop_paths=["crops/001.jpg"],
            metadata={"source": "unit_test"},
        )
        payload = record.to_dict()
        self.assertEqual(payload["track_status"], "completed")
        self.assertEqual(payload["plate"]["normalized_text"], "DL12AB1234")
        self.assertEqual(payload["colour"]["normalized_colour"], "white")


if __name__ == "__main__":
    unittest.main()

