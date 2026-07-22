from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_source import CameraSource


def _write_test_video(path: Path, *, frame_count: int = 5, fps: float = 5.0, size: tuple[int, int] = (64, 48)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Failed to create temporary test video.")
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class CameraSourceTests(unittest.TestCase):
    def test_video_opens_and_metadata_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "camera.avi"
            _write_test_video(video_path, frame_count=4, fps=4.0)
            source = CameraSource(CameraConfig(camera_code="CAM_001", camera_name="North Gate", source_path=video_path, enabled=True))
            source.open()
            metadata = source.metadata()
            self.assertEqual(metadata.camera_code, "CAM_001")
            self.assertEqual(metadata.source_frame_count, 4)
            self.assertAlmostEqual(metadata.source_fps, 4.0, places=2)
            source.close()

    def test_frame_numbers_are_sequential_and_video_time_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "camera.avi"
            _write_test_video(video_path, frame_count=3, fps=5.0)
            source = CameraSource(CameraConfig(camera_code="CAM_001", camera_name="North Gate", source_path=video_path, enabled=True))
            source.open()
            packets = [source.read_next(), source.read_next(), source.read_next()]
            self.assertEqual([packet.frame_number for packet in packets if packet is not None], [0, 1, 2])
            self.assertAlmostEqual(packets[1].video_time_seconds, 0.2, places=3)  # type: ignore[union-attr]
            source.close()

    def test_camera_timestamp_is_calculated_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "camera.avi"
            _write_test_video(video_path, frame_count=2, fps=2.0)
            start_time = datetime(2026, 7, 22, 10, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
            source = CameraSource(CameraConfig(camera_code="CAM_001", camera_name="North Gate", source_path=video_path, enabled=True, start_time=start_time))
            source.open()
            packet_0 = source.read_next()
            packet_1 = source.read_next()
            self.assertEqual(packet_0.camera_timestamp, start_time)  # type: ignore[union-attr]
            self.assertEqual(packet_1.camera_timestamp, start_time + timedelta(seconds=0.5))  # type: ignore[union-attr]
            source.close()

    def test_resources_close_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "camera.avi"
            _write_test_video(video_path)
            with CameraSource(CameraConfig(camera_code="CAM_001", camera_name="North Gate", source_path=video_path, enabled=True)) as source:
                self.assertTrue(source.is_open())
            self.assertFalse(source.is_open())
