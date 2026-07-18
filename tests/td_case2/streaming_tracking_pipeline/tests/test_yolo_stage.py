import unittest

from tests.td_case2.streaming_tracking_pipeline.config import DetectionConfig
from tests.td_case2.streaming_tracking_pipeline.schemas import FramePacket
from tests.td_case2.streaming_tracking_pipeline.yolo_stage import UltralyticsYoloDetectionStage


class _ListValue:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _ListValue([row[0] for row in rows])
        self.conf = _ListValue([row[1] for row in rows])
        self.cls = _ListValue([row[2] for row in rows])


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _FakeModel:
    names = {0: "person", 2: "car", 5: "bus", 99: "unknown"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self.boxes)]


def _frame(index=0):
    return FramePacket(
        source_id="cam_a",
        frame_index=index,
        timestamp_sec=index / 10.0,
        source_fps=10.0,
        frame_width=100,
        frame_height=80,
        frame=object(),
    )


class UltralyticsYoloDetectionStageTest(unittest.TestCase):
    def test_process_filters_clips_and_preserves_frame_metadata(self):
        model = _FakeModel(
            [
                ([-5, 2, 50, 40], 0.9, 2),
                ([1, 1, 20, 20], 0.8, 99),
                ([4, 4, 4, 12], 0.7, 2),
            ]
        )
        stage = UltralyticsYoloDetectionStage(
            DetectionConfig(allowed_class_names=("car",), device="cpu"),
            model=model,
        )

        packet = stage.process(_frame())

        self.assertEqual(packet.source_id, "cam_a")
        self.assertIs(packet.frame, model.calls[0]["source"])
        self.assertEqual(len(packet.detections), 1)
        self.assertEqual(packet.detections[0].bbox.to_xyxy(), [0.0, 2.0, 50.0, 40.0])
        self.assertEqual(packet.detections[0].class_name, "car")
        metrics = stage.to_dict()
        self.assertEqual(metrics["raw_detections"], 3)
        self.assertEqual(metrics["filtered_detections"], 1)
        self.assertEqual(metrics["rejected_invalid_boxes"], 1)

    def test_empty_prediction_is_valid_packet(self):
        stage = UltralyticsYoloDetectionStage(DetectionConfig(device="cpu"), model=_FakeModel([]))

        packet = stage.process(_frame(3))

        self.assertEqual(packet.frame_index, 3)
        self.assertEqual(packet.detections, [])
        self.assertEqual(stage.to_dict()["empty_detection_frames"], 1)

    def test_person_class_is_accepted_and_tagged_with_person_group(self):
        model = _FakeModel([([1, 2, 20, 60], 0.85, 0)])
        stage = UltralyticsYoloDetectionStage(
            DetectionConfig(allowed_class_names=("person",), device="cpu"),
            model=model,
        )

        packet = stage.process(_frame())

        self.assertEqual(len(packet.detections), 1)
        self.assertEqual(packet.detections[0].class_name, "person")
        self.assertEqual(packet.detections[0].object_group, "person")


if __name__ == "__main__":
    unittest.main()
