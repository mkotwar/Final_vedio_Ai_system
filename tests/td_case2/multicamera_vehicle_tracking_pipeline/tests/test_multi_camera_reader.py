from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.multi_camera_reader import MultiCameraReader


def _write_test_video(path: Path, *, frame_count: int, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (48, 32))
    if not writer.isOpened():
        raise RuntimeError("Failed to create temporary test video.")
    for index in range(frame_count):
        frame = np.full((32, 48, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class MultiCameraReaderTests(unittest.TestCase):
    def _configs(self, root: Path) -> list[CameraConfig]:
        start_time = datetime(2026, 7, 22, 10, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        return [
            CameraConfig(camera_code="CAM_001", camera_name="A", source_path=root / "camera_1.avi", enabled=True, start_time=start_time),
            CameraConfig(camera_code="CAM_002", camera_name="B", source_path=root / "camera_2.avi", enabled=True, start_time=start_time),
        ]

    def test_sequential_mode_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=2)
            _write_test_video(root / "camera_2.avi", frame_count=2)
            reader = MultiCameraReader(self._configs(root), mode="sequential")
            packets = list(reader)
            self.assertEqual([packet.camera_code for packet in packets], ["CAM_001", "CAM_001", "CAM_002", "CAM_002"])

    def test_round_robin_mode_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=2)
            _write_test_video(root / "camera_2.avi", frame_count=2)
            reader = MultiCameraReader(self._configs(root), mode="round_robin")
            packets = list(reader)
            self.assertEqual([packet.camera_code for packet in packets], ["CAM_001", "CAM_002", "CAM_001", "CAM_002"])

    def test_one_shorter_video_finishing_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=1)
            _write_test_video(root / "camera_2.avi", frame_count=3)
            reader = MultiCameraReader(self._configs(root), mode="round_robin")
            packets = list(reader)
            self.assertEqual([packet.camera_code for packet in packets], ["CAM_001", "CAM_002", "CAM_002", "CAM_002"])

    def test_camera_metadata_never_gets_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=1)
            _write_test_video(root / "camera_2.avi", frame_count=1)
            reader = MultiCameraReader(self._configs(root), mode="round_robin")
            packets = list(reader)
            self.assertEqual(packets[0].camera_name, "A")
            self.assertEqual(packets[1].camera_name, "B")

    def test_max_frame_limit_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=4)
            _write_test_video(root / "camera_2.avi", frame_count=4)
            reader = MultiCameraReader(self._configs(root), mode="round_robin", max_frames_per_camera=2)
            packets = list(reader)
            self.assertEqual(len(packets), 4)
            self.assertEqual([packet.frame_number for packet in packets if packet.camera_code == "CAM_001"], [0, 1])

    def test_all_resources_close_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_test_video(root / "camera_1.avi", frame_count=1)
            _write_test_video(root / "camera_2.avi", frame_count=1)
            reader = MultiCameraReader(self._configs(root), mode="round_robin")
            list(reader)
            self.assertTrue(all(not source.is_open() for source in reader.sources))
