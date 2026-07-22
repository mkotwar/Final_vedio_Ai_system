from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfigError, load_camera_configs


class CameraConfigTests(unittest.TestCase):
    def test_valid_configuration_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            (root / "data" / "camera_1.mp4").write_bytes(b"placeholder")
            config_path = root / "config" / "cameras.yaml"
            config_path.write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: North Gate\n'
                '    source_path: data/camera_1.mp4\n'
                '    enabled: true\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n',
                encoding="utf-8",
            )
            configs = load_camera_configs(config_path)
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].camera_code, "CAM_001")

    def test_duplicate_camera_code_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            (root / "data" / "camera_1.mp4").write_bytes(b"placeholder")
            config_path = root / "config" / "cameras.yaml"
            config_path.write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: A\n'
                '    source_path: data/camera_1.mp4\n'
                '    enabled: true\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: B\n'
                '    source_path: data/camera_1.mp4\n'
                '    enabled: true\n',
                encoding="utf-8",
            )
            with self.assertRaises(CameraConfigError):
                load_camera_configs(config_path)

    def test_disabled_camera_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            (root / "data" / "camera_1.mp4").write_bytes(b"placeholder")
            config_path = root / "config" / "cameras.yaml"
            config_path.write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: A\n'
                '    source_path: data/camera_1.mp4\n'
                '    enabled: false\n',
                encoding="utf-8",
            )
            self.assertEqual(load_camera_configs(config_path), [])

    def test_missing_source_path_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            config_path = root / "config" / "cameras.yaml"
            config_path.write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: A\n'
                '    source_path: data/missing.mp4\n'
                '    enabled: true\n',
                encoding="utf-8",
            )
            with self.assertRaises(CameraConfigError):
                load_camera_configs(config_path)

    def test_invalid_start_time_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            (root / "data" / "camera_1.mp4").write_bytes(b"placeholder")
            config_path = root / "config" / "cameras.yaml"
            config_path.write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: A\n'
                '    source_path: data/camera_1.mp4\n'
                '    enabled: true\n'
                '    start_time: "not-a-date"\n',
                encoding="utf-8",
            )
            with self.assertRaises(CameraConfigError):
                load_camera_configs(config_path)
