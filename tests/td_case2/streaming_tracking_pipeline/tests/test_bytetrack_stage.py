import unittest

from tests.td_case2.streaming_tracking_pipeline.bytetrack_stage import (
    SupervisionByteTrackStage,
    UltralyticsByteTrackStage,
    create_bytetrack_stage,
)
from tests.td_case2.streaming_tracking_pipeline.config import TrackingConfig
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionPacket, DetectionRecord


def _packet(index=0, source_id="cam_a", detections=None):
    return DetectionPacket(
        source_id=source_id,
        frame_index=index,
        timestamp_sec=index / 10.0,
        frame_width=100,
        frame_height=80,
        detections=detections if detections is not None else [
            DetectionRecord(BoundingBox(1, 2, 20, 30), 0.8, 2, "car")
        ],
        frame=object(),
    )


class _FakeUltralyticsTracker:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def update(self, results, img=None):
        self.calls += 1
        self.last_len = len(results)
        return self.rows


class _FakeSupervisionTracker:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def update(self, detections):
        self.calls += 1
        return self.rows


class ByteTrackStageTest(unittest.TestCase):
    def test_ultralytics_adapter_emits_normalized_track_ids(self):
        tracker = _FakeUltralyticsTracker([[1, 2, 20, 30, 44, 0.7, 2, 0]])
        stage = UltralyticsByteTrackStage(
            config=TrackingConfig(backend="ultralytics_bytetrack"),
            source_fps=10.0,
            tracker=tracker,
        )

        tracked = stage.process(_packet())

        self.assertEqual(tracker.calls, 1)
        self.assertEqual(tracker.last_len, 1)
        self.assertEqual(len(tracked.tracks), 1)
        self.assertEqual(tracked.tracks[0].track_id, 44)
        self.assertEqual(tracked.tracks[0].source_track_id, 44)
        self.assertEqual(tracked.tracks[0].class_name, "car")
        self.assertEqual(stage.to_dict()["unique_source_track_ids"], 1)

    def test_rejects_duplicate_regressing_and_second_source_without_reset(self):
        stage = UltralyticsByteTrackStage(
            config=TrackingConfig(backend="ultralytics_bytetrack"),
            source_fps=10.0,
            tracker=_FakeUltralyticsTracker([]),
        )
        stage.process(_packet(1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stage.process(_packet(1))
        with self.assertRaisesRegex(ValueError, "second source"):
            stage.process(_packet(2, source_id="cam_b"))
        stage.reset()
        stage.process(_packet(0, source_id="cam_b"))
        self.assertEqual(stage.to_dict()["reset_count"], 1)

    def test_supervision_fake_tracker_path_and_factory(self):
        tracker = _FakeSupervisionTracker(
            [{"track_id": "native_7", "bbox": [3, 4, 30, 40], "confidence": 0.6, "class_id": 5, "class_name": "bus"}]
        )
        stage = SupervisionByteTrackStage(
            config=TrackingConfig(backend="supervision_bytetrack"),
            source_fps=12.0,
            tracker=tracker,
        )

        tracked = stage.process(_packet())

        self.assertEqual(tracked.tracks[0].source_track_id, "native_7")
        self.assertEqual(tracked.tracks[0].track_id, 1)
        self.assertEqual(tracked.tracks[0].class_name, "bus")
        created = create_bytetrack_stage(TrackingConfig(backend="ultralytics_bytetrack"), source_fps=12.0, tracker=tracker)
        self.assertEqual(created.backend_name, "ultralytics_bytetrack")

    def test_missing_supervision_backend_fails_clearly(self):
        stage = SupervisionByteTrackStage(
            config=TrackingConfig(backend="supervision_bytetrack"),
            source_fps=10.0,
        )
        with self.assertRaisesRegex(RuntimeError, "supervision is not installed"):
            stage.process(_packet())


if __name__ == "__main__":
    unittest.main()
