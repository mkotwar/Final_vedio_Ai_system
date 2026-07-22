from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..database.client import create_backend_client
from ..database.config import DatabaseConfig
from ..database.repository import SimpleVehicleRepository, SupabaseVehicleRepository, VehicleRepository
from ..detection.detection_config import detection_overrides_from_env, load_detection_config
from ..detection.vehicle_detector import SharedVehicleDetector
from ..orchestration.multi_camera_orchestrator import MultiCameraOrchestrator
from ..persistence.persistence_config import PersistenceConfig, load_persistence_config, persistence_overrides_from_env
from ..persistence.tracking_persistence_service import TrackingPersistenceService
from ..tracking.tracking_config import load_tracking_config, tracking_overrides_from_env
from ..workers.worker_config import WorkerConfig, load_worker_config
from ..workers.worker_supervisor import WorkerSupervisor


@dataclass(slots=True)
class WorkerTrackingRunResult:
    report: dict[str, object]
    report_path: Path


class WorkerMultiCameraTrackingOrchestrator:
    def __init__(
        self,
        camera_config_path: str | Path,
        detection_config_path: str | Path,
        tracking_config_path: str | Path,
        worker_config_path: str | Path,
        persistence_config_path: str | Path | None = None,
        *,
        max_frames_per_camera: int | None = None,
        detection_overrides: dict[str, object] | None = None,
        tracking_overrides: dict[str, object] | None = None,
        worker_overrides: dict[str, object] | None = None,
        persistence_overrides: dict[str, object] | None = None,
        detector: SharedVehicleDetector | None = None,
        repository: VehicleRepository | None = None,
        run_id: str | None = None,
    ) -> None:
        self.camera_orchestrator = MultiCameraOrchestrator(camera_config_path, mode="round_robin", max_frames_per_camera=max_frames_per_camera)
        merged_detection_overrides = detection_overrides_from_env()
        if detection_overrides:
            merged_detection_overrides.update(detection_overrides)
        merged_tracking_overrides = tracking_overrides_from_env()
        if tracking_overrides:
            merged_tracking_overrides.update(tracking_overrides)
        merged_persistence_overrides = persistence_overrides_from_env()
        if persistence_overrides:
            merged_persistence_overrides.update(persistence_overrides)
        self.detection_config = load_detection_config(detection_config_path, overrides=merged_detection_overrides)
        self.tracking_config = load_tracking_config(tracking_config_path, overrides=merged_tracking_overrides)
        self.worker_config = load_worker_config(worker_config_path, overrides=worker_overrides)
        self.persistence_config = (
            load_persistence_config(persistence_config_path, overrides=merged_persistence_overrides)
            if persistence_config_path is not None
            else PersistenceConfig(**merged_persistence_overrides) if merged_persistence_overrides else PersistenceConfig()
        )
        self.max_frames_per_camera = max_frames_per_camera
        self.detector = detector or SharedVehicleDetector(self.detection_config)
        self.repository = repository
        self.run_id = run_id or f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run(
        self,
        *,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        output_report: str | Path | None = None,
    ) -> WorkerTrackingRunResult:
        camera_configs = self.camera_orchestrator.load_enabled_cameras()
        output_dir = Path(output_report).resolve().parent if output_report else self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        persistence_service = None
        if self.worker_config.persist_completed_tracks and self.persistence_config.enabled:
            repository = self.repository or self._build_repository()
            persistence_service = TrackingPersistenceService(repository, self.persistence_config)
            if self.persistence_config.sync_cameras:
                persistence_service.sync_cameras(camera_configs)
        wall_started = time.perf_counter()
        supervisor = WorkerSupervisor(
            camera_configs=camera_configs,
            detector=self.detector,
            tracking_config=self.tracking_config,
            worker_config=self.worker_config,
            max_frames_per_camera=self.max_frames_per_camera,
            persistence_service=persistence_service,
            run_id=self.run_id,
            save_sample_frames=save_sample_frames,
            sample_frame_limit_per_camera=sample_frame_limit_per_camera,
            sample_output_dir=output_dir if save_sample_frames else None,
        )
        result = supervisor.run()
        wall_runtime = time.perf_counter() - wall_started
        tracking_metrics = result.tracking_worker_metrics
        detection_metrics = result.detection_worker_metrics
        total_frames_read = sum(item["frames_read"] for item in result.camera_reader_metrics.values())
        total_completed_tracks = int(tracking_metrics["completed_tracks"])
        total_discarded_tracks = int(tracking_metrics["discarded_tracks"])
        total_track_observations = int(tracking_metrics["track_observations_created"])
        per_camera: dict[str, dict[str, object]] = {}
        for camera_code in sorted(result.camera_reader_metrics):
            per_camera[camera_code] = {
                "frames_read": result.camera_reader_metrics[camera_code]["frames_read"],
                "detections": tracking_metrics["per_camera_detections"].get(camera_code, 0),
                "track_observations": tracking_metrics["per_camera_track_observations"].get(camera_code, 0),
                "completed_tracks": tracking_metrics["per_camera_completed_tracks"].get(camera_code, 0),
                "discarded_tracks": tracking_metrics["per_camera_discarded_tracks"].get(camera_code, 0),
                "first_frame": tracking_metrics["per_camera_first_frame"].get(camera_code),
                "last_frame": tracking_metrics["per_camera_last_frame"].get(camera_code),
                "errors": [
                    {"worker": error.worker_name, "type": error.error_type, "message": error.error_message}
                    for error in result.errors
                    if error.camera_code == camera_code
                ],
            }
        report = {
            "execution_mode": "workers",
            "camera_count": len(camera_configs),
            "run_id": self.run_id,
            "detector": {
                "requested_model": self.detection_config.model_path,
                "actual_model": self.detector.loaded_model_name or self.detection_config.model_path,
                "device": self.detector.device,
            },
            "worker_config": asdict(self.worker_config),
            "total_frames_read": total_frames_read,
            "total_frames_detected": int(detection_metrics["frames_received"]),
            "total_detections": int(detection_metrics["detections_produced"]),
            "total_track_observations": total_track_observations,
            "total_completed_tracks": total_completed_tracks,
            "total_discarded_tracks": total_discarded_tracks,
            "wall_clock_runtime_seconds": wall_runtime,
            "processing_fps": (total_frames_read / wall_runtime) if wall_runtime > 0 else 0.0,
            "workers": {
                "camera_readers": result.camera_reader_metrics,
                "detection_worker": result.detection_worker_metrics,
                "tracking_worker": result.tracking_worker_metrics,
                "persistence_worker": result.persistence_worker_metrics or {},
            },
            "queues": result.queue_metrics,
            "cameras": per_camera,
            "errors": [
                {
                    "worker_name": error.worker_name,
                    "worker_type": error.worker_type,
                    "camera_code": error.camera_code,
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "fatal": error.fatal,
                }
                for error in result.errors
            ],
            "shutdown_clean": result.shutdown_clean,
            "completed_tracks": [
                {
                    "track_uuid": message.track.track_uuid,
                    "camera_code": message.track.camera_code,
                    "local_track_id": message.track.local_track_id,
                    "class_name": message.track.class_name,
                    "observation_count": message.track.observation_count,
                    "state": message.track.state,
                    "persistence_status": getattr(result.persistence_results_by_track_uuid.get(message.track.track_uuid), "status", None),
                }
                for message in result.finalized_tracks
            ],
            "persistence": (
                persistence_service.get_metrics().to_dict() | {"enabled": True, "dry_run": self.persistence_config.dry_run}
                if persistence_service is not None
                else {"enabled": False, "dry_run": self.persistence_config.dry_run}
            ),
            "generated_at": datetime.now().isoformat(),
        }
        report_path = Path(output_report).resolve() if output_report else output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return WorkerTrackingRunResult(report=report, report_path=report_path)

    def _build_repository(self) -> VehicleRepository:
        if self.persistence_config.dry_run:
            return SimpleVehicleRepository()
        database_config = DatabaseConfig.from_env(require_backend_credentials=True)
        return SupabaseVehicleRepository(create_backend_client(database_config))

    @staticmethod
    def _default_output_dir() -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("debug_runs") / "multicamera_vehicle_tracking_pipeline" / f"worker_tracking_validation_{timestamp}"
