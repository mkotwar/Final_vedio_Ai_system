from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.repository import RepositoryConstraintError, SimpleVehicleRepository
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.orchestration.multicamera_tracking_orchestrator import MultiCameraTrackingOrchestrator
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.camera_detection_router import CameraDetectionRouter
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracker_factory import TrackerFactory
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig


def _write_test_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (48, 32))
    if not writer.isOpened():
        raise RuntimeError("Failed to create temporary test video.")
    for index in range(frame_count):
        frame = np.full((32, 48, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _FakeDetector:
    def __init__(self) -> None:
        self.loaded_model_name = "fake_detector.pt"
        self.device = "cpu"

    def detect(self, frame_packet: FramePacket) -> DetectionPacket:
        detections = []
        if frame_packet.frame_number % 2 == 0:
            detections.append(VehicleDetection(class_id=0, class_name="car", confidence=0.9, bbox_xyxy=(1.0, 2.0, 10.0, 12.0)))
        return DetectionPacket(
            camera_code=frame_packet.camera_code,
            camera_name=frame_packet.camera_name,
            source_path=frame_packet.source_path,
            frame_number=frame_packet.frame_number,
            video_time_seconds=frame_packet.video_time_seconds,
            camera_timestamp=frame_packet.camera_timestamp,
            frame_width=48,
            frame_height=32,
            detections=detections,
            inference_time_ms=5.0,
            detector_model=self.loaded_model_name,
            detector_device=self.device,
            frame=frame_packet.frame,
        )


class _SharedIdTracker:
    def update(self, results, img=None):
        if len(results) == 0:
            return []
        return [[1, 2, 10, 12, 1, 0.9, 0, 0]]


class _ExplodingRepo(SimpleVehicleRepository):
    def add_vehicle_observations(self, observations):
        raise RepositoryConstraintError("boom")


class TrackingOrchestratorTests(unittest.TestCase):
    def _build_case(self, root: Path) -> None:
        (root / "config").mkdir()
        (root / "data").mkdir()
        _write_test_video(root / "data" / "camera_1.avi", frame_count=2)
        _write_test_video(root / "data" / "camera_2.avi", frame_count=3)
        (root / "config" / "cameras.yaml").write_text(
            'cameras:\n'
            '  - camera_code: CAM_001\n'
            '    camera_name: North Gate\n'
            '    source_path: data/camera_1.avi\n'
            '    enabled: true\n'
            '    start_time: "2026-07-22T10:00:00+05:30"\n'
            '  - camera_code: CAM_002\n'
            '    camera_name: Parking Entry\n'
            '    source_path: data/camera_2.avi\n'
            '    enabled: true\n'
            '    start_time: "2026-07-22T10:00:00+05:30"\n',
            encoding="utf-8",
        )
        (root / "config" / "detection.yaml").write_text(
            'vehicle_detector:\n'
            '  model_path: yolov8n.pt\n'
            '  fallback_model_path: yolov8n.pt\n'
            '  allow_fallback: true\n'
            '  device: cpu\n'
            '  confidence_threshold: 0.25\n'
            '  iou_threshold: 0.45\n'
            '  image_size: 640\n'
            '  allowed_classes:\n'
            '    - car\n'
            '    - bus\n'
            '    - truck\n'
            '    - motorcycle\n',
            encoding="utf-8",
        )
        (root / "config" / "tracking.yaml").write_text(
            'tracking:\n'
            '  backend: ultralytics_bytetrack\n'
            '  track_high_thresh: 0.30\n'
            '  track_low_thresh: 0.10\n'
            '  new_track_thresh: 0.30\n'
            '  match_thresh: 0.80\n'
            '  track_buffer: 30\n'
            '  min_confirmed_observations: 1\n'
            '  max_lost_frames: 30\n'
            '  preserve_state_per_camera: true\n',
            encoding="utf-8",
        )
        (root / "config" / "persistence.yaml").write_text(
            'persistence:\n'
            '  enabled: true\n'
            '  sync_cameras: true\n'
            '  write_completed_tracks_only: true\n'
            '  include_discarded_tracks: false\n'
            '  observation_mode: all\n'
            '  observation_batch_size: 100\n'
            '  observation_sample_every_n: 5\n'
            '  dry_run: false\n'
            '  fail_on_database_error: true\n',
            encoding="utf-8",
        )

    def test_round_robin_metrics_completed_tracks_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_case(root)
            router = CameraDetectionRouter(
                TrackingConfig(min_confirmed_observations=1),
                tracker_factory=TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker()),
                run_id="RUN_TEST",
            )
            orchestrator = MultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                mode="round_robin",
                detector=_FakeDetector(),
                router=router,
                run_id="RUN_TEST",
            )
            result = orchestrator.run(save_sample_frames=True, sample_frame_limit_per_camera=1, output_report=root / "report.json")
            self.assertEqual(result.report["total_frames_processed"], 5)
            self.assertEqual(result.report["total_detections"], 3)
            self.assertEqual(result.report["total_track_observations"], 3)
            self.assertEqual(result.report["total_completed_tracks"], 2)
            self.assertEqual(result.report["cameras"]["CAM_001"]["frames_processed"], 2)
            self.assertEqual(result.report["cameras"]["CAM_002"]["frames_processed"], 3)
            self.assertEqual(result.report["cameras"]["CAM_001"]["active_tracks_at_flush"], 1)
            self.assertEqual(result.report["cameras"]["CAM_002"]["active_tracks_at_flush"], 1)
            completed = result.report["completed_tracks"]
            self.assertEqual({item["track_uuid"] for item in completed}, {"RUN_TEST:CAM_001:TRACK_1", "RUN_TEST:CAM_002:TRACK_1"})
            self.assertTrue((root / "CAM_001" / "sample_000001.jpg").exists())
            self.assertTrue((root / "CAM_002" / "sample_000001.jpg").exists())
            loaded = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["detector"]["actual_model"], "fake_detector.pt")

    def test_persistence_disabled_causes_no_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_case(root)
            repo = SimpleVehicleRepository()
            router = CameraDetectionRouter(
                TrackingConfig(min_confirmed_observations=1),
                tracker_factory=TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker()),
                run_id="RUN_TEST",
            )
            orchestrator = MultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                root / "config" / "persistence.yaml",
                mode="round_robin",
                detector=_FakeDetector(),
                router=router,
                repository=repo,
                run_id="RUN_TEST",
                persistence_overrides={"enabled": False},
            )
            result = orchestrator.run(output_report=root / "report.json")
            self.assertFalse(result.report["persistence"]["enabled"])
            self.assertEqual(repo.list_cameras(), [])

    def test_completed_tracks_persist_and_cameras_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_case(root)
            repo = SimpleVehicleRepository()
            router = CameraDetectionRouter(
                TrackingConfig(min_confirmed_observations=1),
                tracker_factory=TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker()),
                run_id="RUN_TEST",
            )
            orchestrator = MultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                root / "config" / "persistence.yaml",
                mode="round_robin",
                detector=_FakeDetector(),
                router=router,
                repository=repo,
                run_id="RUN_TEST",
                persistence_overrides={"enabled": True},
            )
            result = orchestrator.run(output_report=root / "report.json")
            self.assertEqual(result.report["persistence"]["tracks_inserted"], 2)
            self.assertEqual(result.report["persistence"]["tracks_skipped_discarded"], 0)
            cam1 = repo.get_camera_by_code("CAM_001")
            cam2 = repo.get_camera_by_code("CAM_002")
            self.assertIsNotNone(cam1)
            self.assertIsNotNone(cam2)
            self.assertNotEqual(cam1.id, cam2.id)
            track1 = repo.get_track_by_uuid("RUN_TEST:CAM_001:TRACK_1")
            track2 = repo.get_track_by_uuid("RUN_TEST:CAM_002:TRACK_1")
            self.assertIsNotNone(track1)
            self.assertIsNotNone(track2)
            self.assertNotEqual(track1.camera_id, track2.camera_id)
            self.assertEqual(len(repo.get_track_observations(track1.id)), 1)
            self.assertEqual(len(repo.get_track_observations(track2.id)), 2)

    def test_discarded_tracks_skipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_case(root)
            repo = SimpleVehicleRepository()
            router = CameraDetectionRouter(
                TrackingConfig(min_confirmed_observations=2),
                tracker_factory=TrackerFactory(TrackingConfig(min_confirmed_observations=2), tracker_creator=lambda config: _SharedIdTracker()),
                run_id="RUN_TEST",
            )
            orchestrator = MultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                root / "config" / "persistence.yaml",
                mode="round_robin",
                detector=_FakeDetector(),
                router=router,
                repository=repo,
                run_id="RUN_TEST",
                persistence_overrides={"enabled": True, "include_discarded_tracks": False},
            )
            result = orchestrator.run(output_report=root / "report.json")
            self.assertEqual(result.report["persistence"]["tracks_skipped_discarded"], 1)
            self.assertIsNone(repo.get_track_by_uuid("RUN_TEST:CAM_001:TRACK_1"))

    def test_database_error_behavior_follows_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_case(root)
            router = CameraDetectionRouter(
                TrackingConfig(min_confirmed_observations=1),
                tracker_factory=TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker()),
                run_id="RUN_TEST",
            )
            orchestrator = MultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                root / "config" / "persistence.yaml",
                mode="round_robin",
                detector=_FakeDetector(),
                router=router,
                repository=_ExplodingRepo(),
                run_id="RUN_TEST",
                persistence_overrides={"enabled": True, "fail_on_database_error": False},
            )
            result = orchestrator.run(output_report=root / "report.json")
            self.assertEqual(result.report["persistence"]["tracks_failed"], 2)
            self.assertEqual(len(result.report["persistence"]["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
