from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..database.repository import VehicleRepository
from ..detection.detection_config import detection_overrides_from_env, load_detection_config
from ..enrichment.anpr_config import AnprConfig, load_anpr_config
from ..enrichment.anpr_enrichment_service import AnprEnrichmentService
from ..enrichment.best_plate_selector import select_best_plate_candidates
from ..enrichment.florence_config import FlorenceConfig, load_florence_config
from ..enrichment.florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from ..enrichment.plate_candidate_collector import PlateCandidateCollector
from ..enrichment.florence_vehicle_colour_extractor import FlorenceVehicleColourExtractor
from ..enrichment.vehicle_colour_config import VehicleColourConfig, load_vehicle_colour_config
from ..enrichment.vehicle_colour_enrichment_service import VehicleColourEnrichmentService
from ..detection.vehicle_detector import SharedVehicleDetector
from ..evidence.evidence_config import EvidenceConfig, load_evidence_config
from ..evidence.track_evidence_collector import TrackEvidenceCollector
from ..ingestion.camera_config import CameraConfig, CameraConfigError, apply_file_source_timestamp_policy, load_camera_configs
from ..models.florence_runtime_factory import FlorenceRuntimeFactory
from ..models.plate_detector_runtime_factory import PlateDetectorRuntimeFactory
from ..orchestration.multi_camera_orchestrator import MultiCameraOrchestrator
from ..persistence.analytics_database_client import AnalyticsDatabaseClient
from ..persistence.analytics_repositories import (
    AnalyticsPlateDetectionRepository,
    AnalyticsPlateReadingRepository,
    AnalyticsPlateSummaryRepository,
)
from ..persistence.persistence_backend_factory import build_persistence_service
from ..persistence.persistence_config import PersistenceConfig, load_persistence_config, persistence_overrides_from_env
from ..persistence.persistence_service_protocol import PersistenceServiceProtocol
from ..persistence.track_media_repository import TrackMediaRepository
from ..persistence.vehicle_colour_repository import VehicleColourRepository
from ..persistence.vehicle_class_mapping import normalize_vehicle_class
from ..tracking.tracking_config import load_tracking_config, tracking_overrides_from_env
from ..workers.worker_config import WorkerConfig, load_worker_config
from ..workers.worker_supervisor import WorkerSupervisor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerTrackingRunResult:
    report: dict[str, object]
    report_path: Path


def select_worker_cameras(
    camera_configs: list[CameraConfig],
    *,
    camera_code: str | None = None,
    camera_codes: list[str] | None = None,
    camera_limit: int | None = None,
) -> list[CameraConfig]:
    enabled_cameras = [camera for camera in camera_configs if camera.enabled]
    if not enabled_cameras:
        raise CameraConfigError("At least one camera must be enabled before starting the worker pipeline.")
    enabled_by_code = {camera.camera_code: camera for camera in enabled_cameras}

    selected = list(enabled_cameras)
    requested_codes: list[str] = []
    if camera_code is not None:
        requested_codes.append(str(camera_code))
    if camera_codes:
        requested_codes.extend(str(value) for value in camera_codes)
    if requested_codes:
        unknown = [code for code in requested_codes if code not in enabled_by_code]
        if unknown:
            raise CameraConfigError(f"Requested camera codes are not enabled in cameras.yaml: {', '.join(unknown)}")
        seen: set[str] = set()
        selected = []
        for code in requested_codes:
            if code in seen:
                continue
            seen.add(code)
            selected.append(enabled_by_code[code])
    if camera_limit is not None:
        if int(camera_limit) <= 0:
            raise CameraConfigError("camera_limit must be positive.")
        selected = selected[: int(camera_limit)]
    if not selected:
        raise CameraConfigError("Camera selection resolved to zero enabled cameras.")
    return selected


def validate_selected_camera_sources(camera_configs: list[CameraConfig]) -> None:
    missing_sources = [camera for camera in camera_configs if not camera.source_path.exists()]
    if missing_sources:
        details = ", ".join(f"{camera.camera_code}={camera.source_path}" for camera in missing_sources)
        raise CameraConfigError(f"Selected camera source_path does not exist: {details}")


class WorkerMultiCameraTrackingOrchestrator:
    def __init__(
        self,
        camera_config_path: str | Path,
        detection_config_path: str | Path,
        tracking_config_path: str | Path,
        worker_config_path: str | Path,
        persistence_config_path: str | Path | None = None,
        evidence_config_path: str | Path | None = None,
        florence_config_path: str | Path | None = None,
        vehicle_colour_config_path: str | Path | None = None,
        anpr_config_path: str | Path | None = None,
        *,
        max_frames_per_camera: int | None = None,
        detection_overrides: dict[str, object] | None = None,
        tracking_overrides: dict[str, object] | None = None,
        worker_overrides: dict[str, object] | None = None,
        persistence_overrides: dict[str, object] | None = None,
        detector: SharedVehicleDetector | None = None,
        repository: VehicleRepository | None = None,
        analytics_client: AnalyticsDatabaseClient | None = None,
        persistence_service: PersistenceServiceProtocol | None = None,
        florence_runtime_factory: FlorenceRuntimeFactory | None = None,
        vehicle_colour_service: VehicleColourEnrichmentService | None = None,
        anpr_service: AnprEnrichmentService | None = None,
        florence_model_path: str | Path | None = None,
        florence_adapter_path: str | Path | None = None,
        florence_processor_path: str | Path | None = None,
        florence_device: str | None = None,
        plate_detector_model_path: str | Path | None = None,
        plate_detector_device: str | None = None,
        run_id: str | None = None,
        camera_code: str | None = None,
        camera_codes: list[str] | None = None,
        camera_limit: int | None = None,
    ) -> None:
        self.camera_orchestrator = MultiCameraOrchestrator(camera_config_path, mode="round_robin", max_frames_per_camera=max_frames_per_camera)
        self.camera_config_path = Path(camera_config_path).expanduser().resolve()
        self.evidence_config_path = Path(evidence_config_path).expanduser().resolve() if evidence_config_path is not None else self.camera_config_path.parent / "evidence.yaml"
        self.florence_config_path = Path(florence_config_path).expanduser().resolve() if florence_config_path is not None else self.camera_config_path.parent / "florence.yaml"
        self.vehicle_colour_config_path = Path(vehicle_colour_config_path).expanduser().resolve() if vehicle_colour_config_path is not None else self.camera_config_path.parent / "vehicle_colour.yaml"
        self.anpr_config_path = Path(anpr_config_path).expanduser().resolve() if anpr_config_path is not None else self.camera_config_path.parent / "anpr.yaml"
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
        self.evidence_config = load_evidence_config(self.evidence_config_path) if self.evidence_config_path.exists() else EvidenceConfig()
        self.florence_config = load_florence_config(self.florence_config_path) if self.florence_config_path.exists() else FlorenceConfig()
        self.vehicle_colour_config = (
            load_vehicle_colour_config(self.vehicle_colour_config_path) if self.vehicle_colour_config_path.exists() else VehicleColourConfig()
        )
        self.anpr_config = load_anpr_config(self.anpr_config_path) if self.anpr_config_path.exists() else AnprConfig()
        self.max_frames_per_camera = max_frames_per_camera
        self.detector = detector or SharedVehicleDetector(self.detection_config)
        self.repository = repository
        self.analytics_client = analytics_client
        self.persistence_service = persistence_service
        project_root = self.camera_config_path.parents[4] if len(self.camera_config_path.parents) > 4 else Path.cwd()
        self.florence_runtime_factory = florence_runtime_factory or FlorenceRuntimeFactory(project_root=project_root)
        self.plate_detector_runtime_factory = PlateDetectorRuntimeFactory(project_root=project_root)
        self.vehicle_colour_service = vehicle_colour_service
        self.anpr_service = anpr_service
        self.florence_model_path = florence_model_path
        self.florence_adapter_path = florence_adapter_path
        self.florence_processor_path = florence_processor_path
        self.florence_device = florence_device
        self.plate_detector_model_path = plate_detector_model_path
        self.plate_detector_device = plate_detector_device
        self.run_id = run_id or f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_started_at: datetime | None = None
        self.camera_code = camera_code
        self.camera_codes = list(camera_codes or [])
        self.camera_limit = camera_limit

    def run(
        self,
        *,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        output_report: str | Path | None = None,
    ) -> WorkerTrackingRunResult:
        LOGGER.info("Preparing worker pipeline run %s", self.run_id)
        all_camera_configs = load_camera_configs(self.camera_config_path, include_disabled=True, validate_paths=False)
        camera_configs = select_worker_cameras(
            all_camera_configs,
            camera_code=self.camera_code,
            camera_codes=self.camera_codes,
            camera_limit=self.camera_limit,
        )
        LOGGER.info("Loaded %s selected cameras", len(camera_configs))
        self._run_started_at = datetime.now(timezone.utc)
        camera_configs = apply_file_source_timestamp_policy(camera_configs, run_started_at=self._run_started_at)
        all_camera_configs = apply_file_source_timestamp_policy(all_camera_configs, run_started_at=self._run_started_at)
        validate_selected_camera_sources(camera_configs)
        configured_camera_count = len(all_camera_configs)
        enabled_camera_count = sum(1 for camera in all_camera_configs if camera.enabled)
        disabled_camera_count = configured_camera_count - enabled_camera_count
        output_dir = Path(output_report).resolve().parent if output_report else self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        persistence_service = self.persistence_service
        evidence_collector = TrackEvidenceCollector(self.evidence_config, run_id=self.run_id) if self.evidence_config.enabled else None
        vehicle_colour_service = self.vehicle_colour_service or self._build_vehicle_colour_service()
        anpr_service = self.anpr_service or self._build_anpr_service()
        if self.worker_config.enable_persistence_worker and self.persistence_config.enabled:
            LOGGER.info("Stage: persistence setup")
            persistence_service = persistence_service or self._build_persistence_service()
            if self.persistence_config.sync_cameras:
                persistence_service.sync_cameras(camera_configs)
                LOGGER.info("Stage complete: camera sync to persistence backend")
        LOGGER.info("Stage: video processing started")
        wall_started = time.perf_counter()
        supervisor = WorkerSupervisor(
            camera_configs=camera_configs,
            detector=self.detector,
            tracking_config=self.tracking_config,
            worker_config=self.worker_config,
            max_frames_per_camera=self.max_frames_per_camera,
            persistence_service=persistence_service,
            vehicle_colour_service=vehicle_colour_service,
            anpr_service=anpr_service,
            evidence_collector=evidence_collector,
            run_id=self.run_id,
            save_sample_frames=save_sample_frames,
            sample_frame_limit_per_camera=sample_frame_limit_per_camera,
            sample_output_dir=output_dir if save_sample_frames else None,
        )
        result = supervisor.run()
        wall_runtime = time.perf_counter() - wall_started
        LOGGER.info("Stage complete: video processing finished in %.2fs", wall_runtime)
        tracking_metrics = result.tracking_worker_metrics
        detection_metrics = result.detection_worker_metrics
        total_frames_read = sum(item["frames_read"] for item in result.camera_reader_metrics.values())
        total_completed_tracks = int(tracking_metrics["completed_tracks"])
        total_discarded_tracks = int(tracking_metrics["discarded_tracks"])
        total_track_observations = int(tracking_metrics["track_observations"])
        per_camera: dict[str, dict[str, object]] = {}
        for camera_config in camera_configs:
            camera_code = camera_config.camera_code
            if camera_code not in result.camera_reader_metrics:
                continue
            reader_thread_name = next(worker.name for worker in supervisor.camera_workers if worker.camera_config.camera_code == camera_code)
            per_camera[camera_code] = {
                "camera_name": camera_config.camera_name,
                "camera_code": camera_config.camera_code,
                "source": str(camera_config.source_path),
                "resolved_source_fps": supervisor.router.diagnostics_by_camera().get(camera_code, {}).get("resolved_frame_rate"),
                "frames_read": result.camera_reader_metrics[camera_code]["frames_read"],
                "frames_processed": tracking_metrics["per_camera_frames"].get(camera_code, 0),
                "first_frame_number": result.camera_reader_metrics[camera_code]["first_frame_number"],
                "last_frame_number": result.camera_reader_metrics[camera_code]["last_frame_number"],
                "detections": tracking_metrics["per_camera_detections"].get(camera_code, 0),
                "track_observations": tracking_metrics["per_camera_track_observations"].get(camera_code, 0),
                "unique_local_track_ids": len(tracking_metrics["unique_track_ids_by_camera"].get(camera_code, [])),
                "unique_track_ids": tracking_metrics["unique_track_ids_by_camera"].get(camera_code, []),
                "completed_tracks": tracking_metrics["per_camera_completed_tracks"].get(camera_code, 0),
                "discarded_tracks": tracking_metrics["per_camera_discarded_tracks"].get(camera_code, 0),
                "first_frame": tracking_metrics["per_camera_first_frame"].get(camera_code),
                "last_frame": tracking_metrics["per_camera_last_frame"].get(camera_code),
                "reader_errors": result.camera_reader_metrics[camera_code]["errors"],
                "tracking_errors": 0,
                "reader_thread_name": reader_thread_name,
                "reader_thread_joined_successfully": result.thread_metrics.get(reader_thread_name, {}).get("joined_successfully", False),
                "errors": [
                    {"worker": error.worker_name, "type": error.error_type, "message": error.error_message}
                    for error in result.errors
                    if error.camera_code == camera_code
                ],
            }
        report = {
            "execution_mode": "workers",
            "camera_count": len(camera_configs),
            "configured_camera_count": configured_camera_count,
            "enabled_camera_count": enabled_camera_count,
            "disabled_camera_count": disabled_camera_count,
            "selected_camera_count": len(camera_configs),
            "camera_codes": [camera.camera_code for camera in camera_configs],
            "run_id": self.run_id,
            "detector": {
                "requested_model": self.detection_config.model_path,
                "actual_model": self.detector.loaded_model_name or self.detection_config.model_path,
                "device": self.detector.device,
            },
            "worker_config": asdict(self.worker_config),
            "worker_thread_count": (
                result.runtime_worker_counts["camera_reader_count"]
                + result.runtime_worker_counts["detection_worker_count"]
                + result.runtime_worker_counts["tracking_worker_count"]
                + result.runtime_worker_counts["persistence_worker_count"]
                + result.runtime_worker_counts["vehicle_colour_worker_count"]
                + result.runtime_worker_counts["anpr_worker_count"]
            ),
            "total_frames_read": total_frames_read,
            "total_frames_processed": int(detection_metrics["frames_processed"]),
            "total_frames_detected": int(detection_metrics["frames_received"]),
            "total_detections": int(detection_metrics["detections_produced"]),
            "total_track_observations": total_track_observations,
            "total_completed_tracks": total_completed_tracks,
            "total_discarded_tracks": total_discarded_tracks,
            "wall_clock_runtime_seconds": wall_runtime,
            "processing_fps": (total_frames_read / wall_runtime) if wall_runtime > 0 else 0.0,
            "workers": {
                "camera_reader_count": result.runtime_worker_counts["camera_reader_count"],
                "detection_worker_count": result.runtime_worker_counts["detection_worker_count"],
                "tracking_worker_count": result.runtime_worker_counts["tracking_worker_count"],
                "persistence_worker_count": result.runtime_worker_counts["persistence_worker_count"],
                "vehicle_colour_worker_count": result.runtime_worker_counts["vehicle_colour_worker_count"],
                "anpr_worker_count": result.runtime_worker_counts["anpr_worker_count"],
                "camera_readers": result.camera_reader_metrics,
                "detection_worker": result.detection_worker_metrics,
                "tracking_worker": result.tracking_worker_metrics,
                "persistence_worker": result.persistence_worker_metrics or {},
                "vehicle_colour_worker": result.vehicle_colour_worker_metrics or {},
                "anpr_worker": result.anpr_worker_metrics or {},
                "thread_shutdown": result.thread_metrics,
                "reader_shutdown": result.runtime_worker_counts,
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
                    "canonical_class_name": normalize_vehicle_class(message.track.class_name).value,
                    "observation_count": message.track.observation_count,
                    "state": message.track.state,
                    "persistence_status": getattr(result.persistence_results_by_track_uuid.get(message.track.track_uuid), "status", None),
                    "media_persistence": getattr(result.persistence_results_by_track_uuid.get(message.track.track_uuid), "media_persistence", None),
                    "vehicle_colour": (
                        result.vehicle_colour_results_by_track_uuid[message.track.track_uuid].to_report_payload()
                        if message.track.track_uuid in result.vehicle_colour_results_by_track_uuid
                        else None
                    ),
                    "anpr": (
                        result.anpr_results_by_track_uuid[message.track.track_uuid].to_report_payload()
                        if message.track.track_uuid in result.anpr_results_by_track_uuid
                        else None
                    ),
                    "evidence": message.track.evidence_package.to_dict() if message.track.evidence_package is not None else None,
                }
                for message in result.finalized_tracks
            ],
            "persistence": (
                persistence_service.get_metrics().to_dict() | {"enabled": True, "dry_run": self.persistence_config.dry_run, "backend": self.persistence_config.backend}
                if persistence_service is not None
                else {"enabled": False, "dry_run": self.persistence_config.dry_run, "backend": self.persistence_config.backend}
            ),
            "vehicle_colour": vehicle_colour_service.get_metrics().to_dict() if vehicle_colour_service is not None else {"enabled": False},
            "anpr": anpr_service.get_metrics().to_dict() if anpr_service is not None else {"enabled": False},
            "generated_at": datetime.now().isoformat(),
            "evidence": {"enabled": self.evidence_config.enabled},
            "debug_artifacts": {
                "sample_frames_enabled": save_sample_frames,
                "sample_frame_limit_per_camera": sample_frame_limit_per_camera if save_sample_frames else 0,
                "sample_output_dir": str(output_dir) if save_sample_frames else None,
            },
        }
        LOGGER.info("Stage: finalizing persistence report")
        self._finalize_persistence(persistence_service, report)
        report_path = Path(output_report).resolve() if output_report else output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Stage complete: report written to %s", report_path)
        return WorkerTrackingRunResult(report=report, report_path=report_path)

    def _build_persistence_service(self) -> PersistenceServiceProtocol:
        service = build_persistence_service(
            config=self.persistence_config,
            run_code=self.run_id,
            detection_config=self.detection_config,
            tracking_config=self.tracking_config,
            execution_mode="THREADED",
            runtime_device=self.detector.device,
            artifact_root=Path(self.evidence_config.output_root),
            run_started_at=self._run_started_at,
            repository=self.repository,
            analytics_client=self.analytics_client,
        )
        if service is None:
            raise RuntimeError("Persistence service requested while backend is disabled.")
        return service

    def _build_vehicle_colour_service(self) -> VehicleColourEnrichmentService | None:
        if not self.vehicle_colour_config.enabled:
            return None
        if self.persistence_config.backend not in {"dry_run", "analytics_supabase"}:
            return None
        runtime = self.florence_runtime_factory.get_runtime(
            config=self.florence_config,
            model_path_cli=self.florence_model_path,
            adapter_path_cli=self.florence_adapter_path,
            processor_path_cli=self.florence_processor_path,
            device_override=self.florence_device,
        )
        if runtime is None:
            return None
        extractor = FlorenceVehicleColourExtractor(
            runtime=runtime,
            prompt=self.florence_config.colour_prompt,
            allowed_colours=self.vehicle_colour_config.allowed_colours,
            minimum_confidence=self.vehicle_colour_config.minimum_confidence,
        )
        analytics_client = None
        track_media_repository = None
        vehicle_colour_repository = None
        if self.persistence_config.backend == "analytics_supabase":
            analytics_client = self.analytics_client or AnalyticsDatabaseClient(schema_name="analytics")
            track_media_repository = TrackMediaRepository(analytics_client)
            vehicle_colour_repository = VehicleColourRepository(analytics_client)
        return VehicleColourEnrichmentService(
            extractor=extractor,
            config=self.vehicle_colour_config,
            persistence_config=self.persistence_config,
            artifact_root=Path(self.evidence_config.output_root),
            track_media_repository=track_media_repository,
            vehicle_colour_repository=vehicle_colour_repository,
        )

    def _build_anpr_service(self) -> AnprEnrichmentService | None:
        if not self.anpr_config.enabled:
            return None
        if self.persistence_config.backend not in {"dry_run", "analytics_supabase"}:
            return None
        plate_runtime = self.plate_detector_runtime_factory.get_runtime(
            config=self.anpr_config,
            model_path_cli=self.plate_detector_model_path,
            device_override=self.plate_detector_device,
        )
        florence_runtime = self.florence_runtime_factory.get_runtime(
            config=self.florence_config,
            model_path_cli=self.florence_model_path,
            adapter_path_cli=self.florence_adapter_path,
            processor_path_cli=self.florence_processor_path,
            device_override=self.florence_device,
        )
        if plate_runtime is None or florence_runtime is None:
            return None
        candidate_collector = PlateCandidateCollector(
            detector_runtime=plate_runtime,
            config=self.anpr_config,
            artifact_root=Path(self.evidence_config.output_root),
        )
        ocr_extractor = FlorencePlateOcrExtractor(
            runtime=florence_runtime,
            ocr_config=self.anpr_config.ocr,
            validation_config=self.anpr_config.validation,
        )
        analytics_client = None
        track_media_repository = None
        plate_detection_repository = None
        plate_reading_repository = None
        plate_summary_repository = None
        if self.persistence_config.backend == "analytics_supabase":
            analytics_client = self.analytics_client or AnalyticsDatabaseClient(schema_name="analytics")
            track_media_repository = TrackMediaRepository(analytics_client)
            plate_detection_repository = AnalyticsPlateDetectionRepository(analytics_client)
            plate_reading_repository = AnalyticsPlateReadingRepository(analytics_client)
            plate_summary_repository = AnalyticsPlateSummaryRepository(analytics_client)
        return AnprEnrichmentService(
            config=self.anpr_config,
            persistence_config=self.persistence_config,
            artifact_root=Path(self.evidence_config.output_root),
            candidate_collector=candidate_collector,
            ocr_extractor=ocr_extractor,
            track_media_repository=track_media_repository,
            plate_detection_repository=plate_detection_repository,
            plate_reading_repository=plate_reading_repository,
            plate_summary_repository=plate_summary_repository,
        )

    @staticmethod
    def _finalize_persistence(persistence_service: PersistenceServiceProtocol | None, report: dict[str, object]) -> None:
        if persistence_service is None:
            return
        finalize = getattr(persistence_service, "finalize_run", None)
        if callable(finalize):
            finalize(report)

    @staticmethod
    def _default_output_dir() -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("debug_runs") / "multicamera_vehicle_tracking_pipeline" / f"worker_tracking_validation_{timestamp}"
