from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.dynamic_yolo_tracking.run_dynamic_tracking_experiment as runner
from experiments.dynamic_yolo_tracking.dynamic_fps_controller import (
    ControllerObservation,
    DynamicFpsConfig,
    DynamicFpsController,
    STATE_BURST,
    STATE_IDLE,
    STATE_LOW,
    STATE_NORMAL,
)
from experiments.dynamic_yolo_tracking.dynamic_frame_decoder import validate_chronological_frame_records
from experiments.step04b_bytetrack.bytetrack_adapter import run_bytetrack_tracking
from experiments.step04b_bytetrack.fragment_merger import merge_track_fragments
from experiments.step04b_bytetrack.tracking_metrics import build_step05_compatible_tracks, validate_step05_compatibility


def _obs(timestamp_seconds: float, **overrides: float | int | None) -> ControllerObservation:
    defaults = {
        "detection_count": 0,
        "active_track_count": 0,
        "avg_center_displacement": 0.0,
        "avg_bbox_area_change": 0.0,
        "avg_direction_change": 0.0,
        "max_track_speed": 0.0,
        "stationary_track_ratio": 0.0,
        "new_track_count": 0,
        "lost_track_count": 0,
        "vehicle_vehicle_proximity": None,
        "vehicle_person_proximity": None,
        "scene_motion_score": 0.0,
        "consecutive_empty_detections": 0,
        "tracker_confidence_instability": 0.0,
    }
    defaults.update(overrides)
    return ControllerObservation(timestamp_seconds=timestamp_seconds, **defaults)


def _frame(frame_idx: int, timestamp_seconds: float) -> dict[str, object]:
    return {
        "frame_id": f"frame_{frame_idx:06d}",
        "frame_idx": frame_idx,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_text": f"{timestamp_seconds:.3f}s",
        "image_path": f"frames/frame_{frame_idx:06d}.jpg",
    }


def _det(frame_idx: int, timestamp_seconds: float, x1: float, y1: float, x2: float, y2: float, *, detection_id: str, class_name: str = "car", confidence: float = 0.8) -> dict[str, object]:
    return {
        "frame_id": f"frame_{frame_idx:06d}",
        "frame_idx": frame_idx,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_text": f"{timestamp_seconds:.3f}s",
        "image_path": f"frames/frame_{frame_idx:06d}.jpg",
        "detection_id": detection_id,
        "class_id": 2 if class_name == "car" else 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_area_ratio": 0.02,
        "crop_path": "",
        "crop_exists": False,
    }


def _track_synthetic(frames: list[dict[str, object]], detections_by_frame: dict[str, list[dict[str, object]]]):
    return run_bytetrack_tracking(
        frame_items=frames,
        detections_by_frame_id=detections_by_frame,
        image_width=1280,
        image_height=720,
        tracking_fps=5.0,
        track_buffer_seconds=2.0,
        high_confidence=0.25,
        low_confidence=0.10,
        match_threshold=0.80,
    )


def _moving_box(frame_idx: int, start_x: int, step_x: int, y: int = 160) -> list[float]:
    x1 = start_x + (frame_idx * step_x)
    return [float(x1), float(y), float(x1 + 90), float(y + 60)]


def _make_frames(count: int, moving_range: tuple[int, int] | None = None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(count):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if moving_range is not None and moving_range[0] <= index <= moving_range[1]:
            x1 = 20 + ((index - moving_range[0]) * 6)
            frame[50:80, x1:x1 + 24] = 255
        frames.append(frame)
    return frames


def _run_single_pass_with_fakes(
    *,
    frames: list[np.ndarray],
    detections_by_frame_idx: dict[int, list[dict[str, object]]],
    heartbeat_seconds: float = 3.0,
):
    metadata = SimpleNamespace(
        video_path=Path("synthetic.mp4"),
        fps=10.0,
        frame_count=len(frames),
        width=int(frames[0].shape[1]),
        height=int(frames[0].shape[0]),
        duration_seconds=len(frames) / 10.0,
    )
    experiment_dir = Path(".")
    saved_paths: list[str] = []
    processor_calls: list[int] = []

    class FakeCapture:
        def __init__(self, frame_items: list[np.ndarray]):
            self._frames = frame_items
            self._index = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self._index >= len(self._frames):
                return False, None
            frame = self._frames[self._index]
            self._index += 1
            return True, frame.copy()

        def release(self) -> None:
            return None

    class FakeProcessor:
        def __init__(self, **kwargs):
            del kwargs

        def process_frame(self, *, frame_item: dict[str, object], original_image: np.ndarray):
            del original_image
            frame_idx = int(frame_item["frame_idx"])
            processor_calls.append(frame_idx)
            rows = [dict(item, frame_id=frame_item["frame_id"], frame_idx=frame_idx, timestamp_seconds=frame_item["timestamp_seconds"], timestamp_text=frame_item["timestamp_text"], image_path=frame_item["image_path"], tracking_state=frame_item["tracking_state"], target_fps=frame_item["target_fps"]) for item in detections_by_frame_idx.get(frame_idx, [])]
            payload = {
                "frame_id": frame_item["frame_id"],
                "frame_idx": frame_idx,
                "timestamp_seconds": frame_item["timestamp_seconds"],
                "timestamp_text": frame_item["timestamp_text"],
                "image_path": frame_item["image_path"],
                "tracking_state": frame_item["tracking_state"],
                "target_fps": frame_item["target_fps"],
                "selection_reason": list(frame_item.get("selection_reason", [])),
                "detections": rows,
            }
            return payload, rows, 0.001

    original_capture = runner.cv2.VideoCapture
    original_processor = runner.YoloFrameProcessor
    original_save = runner._save_frame
    original_build_yolo_config = runner._build_yolo_config
    previous_heartbeat = os.environ.get(runner.ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS)
    os.environ[runner.ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS] = str(heartbeat_seconds)
    try:
        runner.cv2.VideoCapture = lambda *_args, **_kwargs: FakeCapture(frames)
        runner.YoloFrameProcessor = FakeProcessor
        runner._save_frame = lambda _image, image_path: saved_paths.append(str(image_path).replace("\\", "/"))
        runner._build_yolo_config = lambda: SimpleNamespace(
            model_specs=[{"model_role": "fake", "model_path": "fake.pt"}],
            conf_threshold=0.25,
            iou_threshold=0.45,
            device="cpu",
            save_annotated=False,
            save_crops=False,
            track_buffer_seconds=2.0,
            high_confidence=0.25,
            low_confidence=0.10,
            match_threshold=0.80,
            min_track_length=2,
        )
        controller = DynamicFpsController(DynamicFpsConfig())
        result = runner._run_single_pass_dynamic(metadata=metadata, experiment_dir=experiment_dir, controller=controller)
    finally:
        runner.cv2.VideoCapture = original_capture
        runner.YoloFrameProcessor = original_processor
        runner._save_frame = original_save
        runner._build_yolo_config = original_build_yolo_config
        if previous_heartbeat is None:
            os.environ.pop(runner.ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS, None)
        else:
            os.environ[runner.ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS] = previous_heartbeat
    return result, saved_paths, processor_calls, metadata


def test_no_objects_enters_empty():
    controller = DynamicFpsController(DynamicFpsConfig())
    controller.observe(_obs(0.0))
    controller.observe(_obs(1.0))
    controller.observe(_obs(2.0))
    decision = controller.observe(_obs(3.1))
    assert decision.state == STATE_IDLE


def test_object_appears_returns_to_normal():
    controller = DynamicFpsController(DynamicFpsConfig())
    controller.observe(_obs(0.0))
    controller.observe(_obs(1.0))
    controller.observe(_obs(2.0))
    controller.observe(_obs(3.1))
    decision = controller.observe(_obs(4.0, detection_count=1, active_track_count=1))
    assert decision.state == STATE_NORMAL


def test_slow_stationary_object_enters_low():
    controller = DynamicFpsController(DynamicFpsConfig())
    controller.observe(_obs(0.0, detection_count=1, active_track_count=1))
    controller.observe(_obs(1.0, detection_count=1, active_track_count=1, stationary_track_ratio=1.0, max_track_speed=0.005))
    controller.observe(_obs(2.0, detection_count=1, active_track_count=1, stationary_track_ratio=1.0, max_track_speed=0.005))
    decision = controller.observe(_obs(4.1, detection_count=1, active_track_count=1, stationary_track_ratio=1.0, max_track_speed=0.005))
    assert decision.state == STATE_LOW


def test_rapid_movement_enters_burst():
    controller = DynamicFpsController(DynamicFpsConfig())
    controller.observe(_obs(0.0, detection_count=1, active_track_count=1))
    controller.observe(_obs(1.0, detection_count=1, active_track_count=1))
    decision = controller.observe(_obs(2.0, detection_count=1, active_track_count=1, avg_center_displacement=0.13, max_track_speed=0.15))
    assert decision.state == STATE_BURST


def test_variable_frame_indexes_strictly_increasing():
    frames = _make_frames(30, moving_range=(10, 14))
    detections = {10: [_det(10, 1.0, 20, 50, 44, 80, detection_id="d10")]}
    (frame_records, _transitions, _yolo_payload, _yolo_report, _detection_rows, report), _saved, _calls, _meta = _run_single_pass_with_fakes(
        frames=frames,
        detections_by_frame_idx=detections,
    )
    validation = validate_chronological_frame_records(frame_records)
    assert validation["chronological"] is True
    assert validation["duplicate_frame_indexes"] is False
    assert report["duplicate_decoded_frames"] == 0


def test_no_source_frame_is_decoded_twice():
    frames = _make_frames(25)
    (_result, _saved, _calls, _meta) = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx={})
    report = _result[5]
    assert report["video_frames_decoded"] == len(frames)
    assert report["unique_frames_decoded"] == len(frames)
    assert report["duplicate_decoded_frames"] == 0


def test_no_frame_is_sent_to_yolo_twice():
    frames = _make_frames(35, moving_range=(8, 12))
    detections = {8: [_det(8, 0.8, 24, 50, 48, 80, detection_id="d8")], 10: [_det(10, 1.0, 36, 50, 60, 80, detection_id="d10")]}
    (_result, _saved, calls, _meta) = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections)
    assert len(calls) == len(set(calls))
    assert _result[5]["duplicate_yolo_frames"] == 0


def test_empty_road_does_not_continuously_call_yolo():
    frames = _make_frames(40)
    (_result, _saved, calls, _meta) = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx={}, heartbeat_seconds=3.0)
    assert len(calls) <= 2
    assert _result[5]["heartbeat_yolo_calls"] <= 2


def test_empty_frames_are_not_written_to_disk():
    frames = _make_frames(30)
    (_result, saved_paths, _calls, _meta) = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx={})
    assert len(saved_paths) == _result[5]["frames_sent_to_yolo"]
    assert _result[5]["idle_frames_saved"] == 0


def test_motion_entering_roi_immediately_triggers_yolo():
    frames = _make_frames(20, moving_range=(6, 9))
    detections = {6: [_det(6, 0.6, 20, 50, 44, 80, detection_id="d6")]}
    (_result, _saved, calls, _meta) = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections, heartbeat_seconds=0.0)
    assert calls[0] == 6
    assert _result[5]["motion_triggered_yolo_calls"] >= 1


def test_detected_moving_vehicle_enters_normal_state():
    frames = _make_frames(24, moving_range=(6, 12))
    detections = {
        6: [_det(6, 0.6, 20, 50, 44, 80, detection_id="d6")],
        8: [_det(8, 0.8, 34, 50, 58, 80, detection_id="d8")],
    }
    (frame_records, _transitions, _yolo_payload, _yolo_report, _detection_rows, _report), _saved, _calls, _meta = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections)
    assert any(str(item["tracking_state"]) == STATE_NORMAL for item in frame_records)


def test_fast_movement_enters_burst_state():
    frames = _make_frames(40, moving_range=(6, 20))
    detections = {
        6: [_det(6, 0.6, 20, 50, 44, 80, detection_id="d6")],
        10: [_det(10, 1.0, 24, 50, 48, 80, detection_id="d10")],
        12: [_det(12, 1.2, 96, 50, 120, 80, detection_id="d12")],
    }
    (_frame_records, transitions, _yolo_payload, _yolo_report, _detection_rows, _report), _saved, _calls, _meta = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections)
    assert any(str(item["to_state"]) == STATE_BURST for item in transitions)


def test_stationary_vehicle_enters_low_state():
    frames = _make_frames(60, moving_range=(6, 40))
    detections = {index: [_det(index, index / 10.0, 44, 50, 68, 80, detection_id=f"d{index}")] for index in range(6, 41)}
    (_frame_records, transitions, _yolo_payload, _yolo_report, _detection_rows, _report), _saved, _calls, _meta = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections)
    assert any(str(item["to_state"]) == STATE_LOW for item in transitions)


def test_empty_scene_returns_to_empty_state():
    frames = _make_frames(80, moving_range=(6, 12))
    detections = {
        6: [_det(6, 0.6, 20, 50, 44, 80, detection_id="d6")],
        8: [_det(8, 0.8, 32, 50, 56, 80, detection_id="d8")],
    }
    (_frame_records, transitions, _yolo_payload, _yolo_report, _detection_rows, _report), _saved, _calls, _meta = _run_single_pass_with_fakes(frames=frames, detections_by_frame_idx=detections, heartbeat_seconds=1.0)
    assert any(str(item["to_state"]) == STATE_IDLE for item in transitions)


def test_one_moving_vehicle_with_missed_detections_remains_one_track():
    frames = [_frame(index, index * 0.2) for index in range(8)]
    detections = {
        frames[0]["frame_id"]: [_det(0, 0.0, 100, 100, 180, 160, detection_id="d0")],
        frames[1]["frame_id"]: [_det(1, 0.2, 118, 102, 198, 162, detection_id="d1")],
        frames[3]["frame_id"]: [_det(3, 0.6, 154, 106, 236, 168, detection_id="d3")],
        frames[4]["frame_id"]: [_det(4, 0.8, 172, 108, 255, 171, detection_id="d4")],
        frames[6]["frame_id"]: [_det(6, 1.2, 210, 112, 294, 176, detection_id="d6")],
    }
    tracks, _meta = _track_synthetic(frames, detections)
    vehicle_tracks = [track for track in tracks if track["track_type"] == "vehicle"]
    assert len(vehicle_tracks) == 1


def test_two_passing_vehicles_remain_separate():
    frames = [_frame(index, index * 0.2) for index in range(5)]
    detections = {}
    for index, frame in enumerate(frames):
        detections[frame["frame_id"]] = [
            _det(index, index * 0.2, 100 + (index * 20), 100, 180 + (index * 20), 160, detection_id=f"a{index}"),
            _det(index, index * 0.2, 260 + (index * 20), 110, 340 + (index * 20), 170, detection_id=f"b{index}"),
        ]
    tracks, _meta = _track_synthetic(frames, detections)
    vehicle_tracks = [track for track in tracks if track["track_type"] == "vehicle"]
    assert len(vehicle_tracks) == 2


def test_crossing_vehicles_are_not_merged(tmp_path: Path):
    raw_tracks = [
        {
            "track_id": "vehicle_track_0001",
            "track_type": "vehicle",
            "source": "bytetrack_raw",
            "detections": [_det(0, 0.0, 100, 100, 180, 160, detection_id="a0"), _det(1, 0.2, 160, 110, 240, 170, detection_id="a1")],
        },
        {
            "track_id": "vehicle_track_0002",
            "track_type": "vehicle",
            "source": "bytetrack_raw",
            "detections": [_det(2, 0.4, 240, 110, 320, 170, detection_id="b0"), _det(3, 0.6, 180, 100, 260, 160, detection_id="b1")],
        },
    ]
    merged, _audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6, max_prediction_distance=0.05)
    assert meta["post_merge_track_count"] == 2
    assert len(merged) == 2


def test_output_remains_step05_compatible():
    payload, _report = build_step05_compatible_tracks(
        run_dir=Path("."),
        tracks=[
            {
                "track_id": "vehicle_track_0001",
                "track_type": "vehicle",
                "source": "post_merge",
                "detections": [
                    _det(0, 0.0, 100, 100, 180, 160, detection_id="d0"),
                    _det(1, 0.2, 120, 100, 200, 160, detection_id="d1"),
                    _det(2, 0.4, 140, 100, 220, 160, detection_id="d2"),
                    _det(3, 0.6, 160, 100, 240, 160, detection_id="d3"),
                ],
            }
        ],
        image_width=1280,
        image_height=720,
        min_track_length=2,
    )
    compatibility = validate_step05_compatibility(payload)
    assert compatibility["compatible"] is True
