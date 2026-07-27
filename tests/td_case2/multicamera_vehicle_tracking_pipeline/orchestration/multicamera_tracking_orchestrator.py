from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ..database.repository import VehicleRepository
from ..detection.detection_config import detection_overrides_from_env, load_detection_config
from ..detection.vehicle_detector import SharedVehicleDetector
from ..evidence.evidence_config import EvidenceConfig, load_evidence_config
from ..evidence.track_evidence_collector import TrackEvidenceCollector
from ..ingestion.camera_config import apply_file_source_timestamp_policy
from ..ingestion.multi_camera_reader import MultiCameraReader
from ..orchestration.multi_camera_orchestrator import MultiCameraOrchestrator
from ..persistence.analytics_database_client import AnalyticsDatabaseClient
from ..persistence.persistence_backend_factory import build_persistence_service
from ..persistence.persistence_config import PersistenceConfig, load_persistence_config, persistence_overrides_from_env
from ..persistence.persistence_service_protocol import PersistenceServiceProtocol
from ..persistence.persistence_models import TrackPersistenceResult
from ..persistence.vehicle_class_mapping import normalize_vehicle_class
from ..tracking.annotation import annotate_tracking_frame
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.tracking_config import TrackingConfig, load_tracking_config, tracking_overrides_from_env
from ..tracking.tracking_models import LocalVehicleTrack

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackingRunResult:
    report: dict[str, object]
    report_path: Path


class MultiCameraTrackingOrchestrator:
    def __init__(
        self,
        camera_config_path: str | Path,
        detection_config_path: str | Path,
        tracking_config_path: str | Path,
        persistence_config_path: str | Path | None = None,
        evidence_config_path: str | Path | None = None,
        *,
        mode: str = "round_robin",
        max_frames_per_camera: int | None = None,
        detection_overrides: dict[str, object] | None = None,
        tracking_overrides: dict[str, object] | None = None,
        persistence_overrides: dict[str, object] | None = None,
        detector: SharedVehicleDetector | None = None,
        router: CameraDetectionRouter | None = None,
        repository: VehicleRepository | None = None,
        analytics_client: AnalyticsDatabaseClient | None = None,
        persistence_service: PersistenceServiceProtocol | None = None,
        run_id: str | None = None,
    ) -> None:
        self.camera_orchestrator = MultiCameraOrchestrator(camera_config_path, mode=mode, max_frames_per_camera=max_frames_per_camera)
        self.camera_config_path = Path(camera_config_path).expanduser().resolve()
        self.evidence_config_path = Path(evidence_config_path).expanduser().resolve() if evidence_config_path is not None else self.camera_config_path.parent / "evidence.yaml"
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
        self.persistence_config = (
            load_persistence_config(persistence_config_path, overrides=merged_persistence_overrides)
            if persistence_config_path is not None
            else PersistenceConfig(**merged_persistence_overrides) if merged_persistence_overrides else PersistenceConfig()
        )
        self.evidence_config = load_evidence_config(self.evidence_config_path) if self.evidence_config_path.exists() else EvidenceConfig()
        self.detector = detector or SharedVehicleDetector(self.detection_config)
        self.run_id = run_id or f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_started_at: datetime | None = None
        self.router = router or CameraDetectionRouter(self.tracking_config, run_id=self.run_id)
        self.repository = repository
        self.analytics_client = analytics_client
        self.persistence_service: PersistenceServiceProtocol | None = persistence_service

    def run(
        self,
        *,
        preview: bool = False,
        preview_scale: float = 1.0,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        output_report: str | Path | None = None,
    ) -> TrackingRunResult:
        self._run_started_at = datetime.now(timezone.utc)
        camera_configs = apply_file_source_timestamp_policy(
            self.camera_orchestrator.load_enabled_cameras(),
            run_started_at=self._run_started_at,
        )
        persistence_results_by_uuid: dict[str, TrackPersistenceResult] = {}
        finalized_tracks_by_uuid: dict[str, LocalVehicleTrack] = {}
        if self.persistence_config.enabled:
            self.persistence_service = self.persistence_service or self._build_persistence_service()
            if self.persistence_config.sync_cameras:
                self.persistence_service.sync_cameras(camera_configs)
        reader = self.camera_orchestrator._build_reader(camera_configs) if hasattr(self.camera_orchestrator, "_build_reader") else MultiCameraReader(
            camera_configs,
            mode=self.camera_orchestrator.mode,
            max_frames_per_camera=self.camera_orchestrator.max_frames_per_camera,
        )
        output_dir = Path(output_report).resolve().parent if output_report else self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_counts = {config.camera_code: 0 for config in camera_configs}
        camera_stats: dict[str, dict[str, object]] = {
            config.camera_code: {
                "camera_name": config.camera_name,
                "frames_processed": 0,
                "detections": 0,
                "track_observations": 0,
                "unique_local_track_ids": 0,
                "completed_tracks": 0,
                "active_tracks_at_flush": 0,
                "class_counts": {},
                "errors": [],
            }
            for config in camera_configs
        }
        unique_track_ids: dict[str, set[int]] = {config.camera_code: set() for config in camera_configs}
        completed_tracks: list[LocalVehicleTrack] = []
        evidence_collector = TrackEvidenceCollector(self.evidence_config, run_id=self.run_id) if self.evidence_config.enabled else None
        total_frames_processed = 0
        total_detections = 0
        total_track_observations = 0
        inference_times_ms: list[float] = []
        wall_started = time.perf_counter()

        try:
            reader.open()
            for frame_packet in reader:
                detection_packet = self.detector.detect(frame_packet)
                tracking_result = self.router.route(detection_packet)
                if evidence_collector is not None:
                    evidence_collector.update(detection_packet, tracking_result.observations)
                stats = camera_stats[frame_packet.camera_code]
                total_frames_processed += 1
                total_detections += len(detection_packet.detections)
                total_track_observations += len(tracking_result.observations)
                inference_times_ms.append(detection_packet.inference_time_ms)
                stats["frames_processed"] = int(stats["frames_processed"]) + 1
                stats["detections"] = int(stats["detections"]) + len(detection_packet.detections)
                stats["track_observations"] = int(stats["track_observations"]) + len(tracking_result.observations)
                class_counts: Counter[str] = Counter(stats["class_counts"])
                for observation in tracking_result.observations:
                    unique_track_ids[frame_packet.camera_code].add(observation.local_track_id)
                    class_counts[observation.class_name] += 1
                stats["class_counts"] = dict(sorted(class_counts.items()))
                stats["unique_local_track_ids"] = len(unique_track_ids[frame_packet.camera_code])
                stats["completed_tracks"] = int(stats["completed_tracks"]) + len(tracking_result.completed_tracks)
                completed_tracks.extend(tracking_result.completed_tracks)
                for completed_track in tracking_result.completed_tracks:
                    if evidence_collector is not None:
                        completed_track.evidence_package = evidence_collector.finalize_track(completed_track)
                    finalized_tracks_by_uuid.setdefault(completed_track.track_uuid, completed_track)
                if self.persistence_service is not None:
                    for completed_track in tracking_result.completed_tracks:
                        if completed_track.track_uuid in persistence_results_by_uuid:
                            continue
                        persistence_results_by_uuid[completed_track.track_uuid] = self.persistence_service.save_completed_track(completed_track)

                annotated_frame = None
                if preview or save_sample_frames:
                    annotated_frame = annotate_tracking_frame(
                        frame_packet,
                        tracking_result.observations,
                        active_track_count=len(tracking_result.active_tracks),
                    )
                if preview and annotated_frame is not None:
                    shown_frame = annotated_frame
                    if preview_scale != 1.0:
                        shown_frame = cv2.resize(shown_frame, None, fx=preview_scale, fy=preview_scale)
                    cv2.imshow("multicamera_tracking_preview", shown_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                if save_sample_frames and annotated_frame is not None and sample_counts[frame_packet.camera_code] < sample_frame_limit_per_camera:
                    sample_counts[frame_packet.camera_code] += 1
                    camera_dir = output_dir / frame_packet.camera_code
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    sample_path = camera_dir / f"sample_{sample_counts[frame_packet.camera_code]:06d}.jpg"
                    cv2.imwrite(str(sample_path), annotated_frame)
        finally:
            reader.close()
            if preview:
                cv2.destroyAllWindows()

        active_counts_before_flush = {
            camera_code: len(self.router.lifecycle.get_active_tracks(camera_code))
            for camera_code in camera_stats
        }
        flush_result = self.router.flush_all()
        completed_tracks.extend(flush_result.completed_tracks)
        for completed_track in flush_result.completed_tracks:
            if evidence_collector is not None:
                completed_track.evidence_package = evidence_collector.finalize_track(completed_track)
            finalized_tracks_by_uuid.setdefault(completed_track.track_uuid, completed_track)
        if self.persistence_service is not None:
            for completed_track in flush_result.completed_tracks:
                if completed_track.track_uuid in persistence_results_by_uuid:
                    continue
                persistence_results_by_uuid[completed_track.track_uuid] = self.persistence_service.save_completed_track(completed_track)
        for camera_code, stats in camera_stats.items():
            completed_count = sum(1 for track in flush_result.completed_tracks if track.camera_code == camera_code)
            stats["completed_tracks"] = int(stats["completed_tracks"]) + completed_count
            stats["active_tracks_at_flush"] = active_counts_before_flush[camera_code]
        tracker_diagnostics = self.router.diagnostics_by_camera() if hasattr(self.router, "diagnostics_by_camera") else {}
        for camera_code, stats in camera_stats.items():
            stats["tracking_diagnostics"] = tracker_diagnostics.get(camera_code, {})

        wall_runtime = time.perf_counter() - wall_started
        deduped_completed_tracks = list(finalized_tracks_by_uuid.values())
        report = {
            "mode": self.camera_orchestrator.mode,
            "camera_count": len(camera_configs),
            "run_id": self.run_id,
            "detector": {
                "requested_model": self.detection_config.model_path,
                "actual_model": self.detector.loaded_model_name or self.detection_config.model_path,
                "device": self.detector.device,
            },
            "tracking": {
                "backend": self.tracking_config.backend,
                "track_high_thresh": self.tracking_config.track_high_thresh,
                "track_low_thresh": self.tracking_config.track_low_thresh,
                "new_track_thresh": self.tracking_config.new_track_thresh,
                "match_thresh": self.tracking_config.match_thresh,
                "track_buffer": self.tracking_config.track_buffer,
                "track_activation_threshold": self.tracking_config.track_activation_threshold,
                "lost_track_buffer": self.tracking_config.lost_track_buffer,
                "minimum_matching_threshold": self.tracking_config.minimum_matching_threshold,
                "frame_rate": self.tracking_config.frame_rate,
                "minimum_consecutive_frames": self.tracking_config.minimum_consecutive_frames,
                "min_confirmed_observations": self.tracking_config.min_confirmed_observations,
                "max_lost_frames": self.tracking_config.max_lost_frames,
                "preserve_state_per_camera": self.tracking_config.preserve_state_per_camera,
                "compatibility_mapping": {
                    "track_high_thresh": "track_activation_threshold",
                    "track_buffer": "lost_track_buffer",
                    "match_thresh": "minimum_matching_threshold",
                },
            },
            "total_frames_processed": total_frames_processed,
            "total_detections": total_detections,
            "total_track_observations": total_track_observations,
            "total_completed_tracks": len(deduped_completed_tracks),
            "average_inference_time_ms": (sum(inference_times_ms) / len(inference_times_ms)) if inference_times_ms else 0.0,
            "wall_clock_runtime_seconds": wall_runtime,
            "processing_fps": (total_frames_processed / wall_runtime) if wall_runtime > 0 else 0.0,
            "cameras": camera_stats,
            "completed_tracks": [self._serialize_completed_track(track, persistence_results_by_uuid.get(track.track_uuid)) for track in deduped_completed_tracks],
            "persistence": self._build_persistence_report(),
            "evidence": {"enabled": self.evidence_config.enabled},
            "generated_at": datetime.now().isoformat(),
        }
        self._finalize_persistence(report)
        report_path = Path(output_report).resolve() if output_report else output_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info(
            "Tracking validation complete cameras=%s frames=%s detections=%s observations=%s completed=%s model=%s device=%s report=%s",
            len(camera_configs),
            total_frames_processed,
            total_detections,
            total_track_observations,
            len(deduped_completed_tracks),
            self.detector.loaded_model_name or self.detection_config.model_path,
            self.detector.device,
            report_path,
        )
        return TrackingRunResult(report=report, report_path=report_path)

    @staticmethod
    def _serialize_completed_track(track: LocalVehicleTrack, persistence_result: TrackPersistenceResult | None = None) -> dict[str, object]:
        payload = {
            "track_uuid": track.track_uuid,
            "camera_code": track.camera_code,
            "local_track_id": track.local_track_id,
            "class_name": track.class_name,
            "canonical_class_name": normalize_vehicle_class(track.class_name).value,
            "first_frame_number": track.first_frame_number,
            "last_frame_number": track.last_frame_number,
            "first_seen_at": track.first_seen_at.isoformat() if track.first_seen_at is not None else None,
            "last_seen_at": track.last_seen_at.isoformat() if track.last_seen_at is not None else None,
            "first_video_time_seconds": track.first_video_time_seconds,
            "last_video_time_seconds": track.last_video_time_seconds,
            "observation_count": track.observation_count,
            "best_confidence": track.best_confidence,
            "state": track.state,
        }
        if persistence_result is not None:
            payload["persistence_status"] = persistence_result.status
            payload["database_track_id"] = persistence_result.database_track_id
            payload["observations_written"] = persistence_result.observations_written
            payload["persistence_error"] = persistence_result.error
            payload["media_persistence"] = persistence_result.media_persistence
        if track.evidence_package is not None:
            payload["evidence"] = track.evidence_package.to_dict()
        return payload

    @staticmethod
    def _default_output_dir() -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("debug_runs") / "multicamera_vehicle_tracking_pipeline" / f"tracking_validation_{timestamp}"

    def _build_persistence_service(self) -> PersistenceServiceProtocol:
        service = build_persistence_service(
            config=self.persistence_config,
            run_code=self.run_id,
            detection_config=self.detection_config,
            tracking_config=self.tracking_config,
            execution_mode="SEQUENTIAL",
            runtime_device=self.detector.device,
            artifact_root=Path(self.evidence_config.output_root),
            run_started_at=self._run_started_at,
            repository=self.repository,
            analytics_client=self.analytics_client,
        )
        if service is None:
            raise RuntimeError("Persistence service requested while backend is disabled.")
        return service

    def _finalize_persistence(self, report: dict[str, object]) -> None:
        if self.persistence_service is None:
            return
        finalize = getattr(self.persistence_service, "finalize_run", None)
        if callable(finalize):
            finalize(report)

    def _build_persistence_report(self) -> dict[str, object]:
        if self.persistence_service is None:
            return {
                "enabled": False,
                "dry_run": bool(self.persistence_config.dry_run),
                "backend": self.persistence_config.backend,
                "cameras_synced": 0,
                "tracks_considered": 0,
                "tracks_inserted": 0,
                "tracks_already_existing": 0,
                "tracks_skipped_discarded": 0,
                "tracks_skipped_invalid_state": 0,
                "tracks_failed": 0,
                "observations_written": 0,
                "errors": [],
                "persistence_status": "disabled",
            }
        metrics = self.persistence_service.get_metrics().to_dict()
        metrics["enabled"] = True
        metrics["dry_run"] = bool(self.persistence_config.dry_run)
        metrics["backend"] = self.persistence_config.backend
        metrics["persistence_status"] = "dry_run" if self.persistence_config.dry_run else "enabled"
        return metrics
