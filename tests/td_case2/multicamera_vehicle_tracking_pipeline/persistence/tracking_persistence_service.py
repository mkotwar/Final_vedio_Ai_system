from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..database.models import CameraRecord, VehicleObservationRecord, VehicleTrackRecord
from ..database.repository import RepositoryConstraintError, VehicleRepository
from ..ingestion.camera_config import CameraConfig
from ..persistence.vehicle_class_mapping import RUNTIME_VEHICLE_CLASSES, is_supported_vehicle_class, normalize_runtime_vehicle_class
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation
from .persistence_config import PersistenceConfig
from .persistence_models import PersistenceRunMetrics, TrackPersistenceResult


SUPPORTED_VEHICLE_CLASSES = set(RUNTIME_VEHICLE_CLASSES)


class PersistenceValidationError(ValueError):
    """Raised when a track or observation is invalid for persistence."""


@dataclass(slots=True)
class PreparedTrackWrite:
    track_record: VehicleTrackRecord
    observation_rows: list[VehicleObservationRecord]


class TrackingPersistenceService:
    def __init__(self, repository: VehicleRepository, config: PersistenceConfig) -> None:
        self.repository = repository
        self.config = config
        self.metrics = PersistenceRunMetrics()
        self.camera_id_by_code: dict[str, object] = {}

    def sync_cameras(self, camera_configs: Iterable[CameraConfig]) -> dict[str, object]:
        synced = 0
        for camera_config in camera_configs:
            existing = self.repository.get_camera_by_code(camera_config.camera_code)
            if existing is None:
                if self.config.dry_run:
                    self.camera_id_by_code[camera_config.camera_code] = f"DRY_RUN:{camera_config.camera_code}"
                else:
                    created = self.repository.create_camera(
                        CameraRecord(
                            camera_code=camera_config.camera_code,
                            camera_name=camera_config.camera_name,
                            source_path=str(camera_config.source_path),
                            enabled=bool(camera_config.enabled),
                        )
                    )
                    self.camera_id_by_code[camera_config.camera_code] = created.id
            else:
                self.camera_id_by_code[camera_config.camera_code] = existing.id
            synced += 1
        self.metrics.cameras_synced = synced
        return dict(self.camera_id_by_code)

    def save_completed_track(self, track: LocalVehicleTrack) -> TrackPersistenceResult:
        self.metrics.tracks_considered += 1
        try:
            result = self._save_track_internal(track)
        except Exception as exc:
            self.metrics.tracks_failed += 1
            message = f"{track.track_uuid}: {exc}"
            self.metrics.errors.append(message)
            if self.config.fail_on_database_error:
                raise
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="failed", database_track_id=None, observations_written=0, error=str(exc))
        self._apply_metrics(result)
        return result

    def save_completed_tracks(self, tracks: Iterable[LocalVehicleTrack]) -> list[TrackPersistenceResult]:
        return [self.save_completed_track(track) for track in tracks]

    def get_metrics(self) -> PersistenceRunMetrics:
        return self.metrics

    def prepare_track_write(self, track: LocalVehicleTrack) -> PreparedTrackWrite:
        self._validate_track(track)
        if track.camera_code not in self.camera_id_by_code:
            raise PersistenceValidationError(f"Unknown camera_code for persistence: {track.camera_code}")
        camera_id = self.camera_id_by_code[track.camera_code]
        runtime_class_name = normalize_runtime_vehicle_class(track.class_name)
        if runtime_class_name is None:
            raise PersistenceValidationError(f"Unsupported vehicle class: {track.class_name}")
        track_record = VehicleTrackRecord(
            camera_id=camera_id,
            local_track_id=track.local_track_id,
            vehicle_class=runtime_class_name,
            track_uuid=track.track_uuid,
            first_seen_at=track.first_seen_at,
            last_seen_at=track.last_seen_at,
            first_frame_number=track.first_frame_number,
            last_frame_number=track.last_frame_number,
            observation_count=track.observation_count,
            best_confidence=track.best_confidence,
            best_frame_path=None,
            best_crop_path=None,
        )
        observation_rows = self._select_observation_rows(track, database_track_id=track_record.id)
        return PreparedTrackWrite(track_record=track_record, observation_rows=observation_rows)

    def _save_track_internal(self, track: LocalVehicleTrack) -> TrackPersistenceResult:
        if track.state == "discarded" and not self.config.include_discarded_tracks:
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="skipped_discarded", database_track_id=None, observations_written=0)
        if self.config.write_completed_tracks_only and track.state != "completed":
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="skipped_invalid_state", database_track_id=None, observations_written=0)
        prepared = self.prepare_track_write(track)
        existing = self.repository.get_track_by_uuid(track.track_uuid)
        if existing is not None:
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="already_exists", database_track_id=str(existing.id), observations_written=0)
        if self.config.dry_run:
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="dry_run", database_track_id=None, observations_written=len(prepared.observation_rows))
        try:
            created_track = self.repository.create_vehicle_track(prepared.track_record)
        except RepositoryConstraintError as exc:
            if "Duplicate track_uuid" in str(exc):
                existing = self.repository.get_track_by_uuid(track.track_uuid)
                return TrackPersistenceResult(track_uuid=track.track_uuid, status="already_exists", database_track_id=str(existing.id) if existing is not None else None, observations_written=0)
            raise
        observation_rows = [
            VehicleObservationRecord(
                vehicle_track_id=created_track.id,
                frame_number=item.frame_number,
                observed_at=item.observed_at,
                bbox_x1=item.bbox_x1,
                bbox_y1=item.bbox_y1,
                bbox_x2=item.bbox_x2,
                bbox_y2=item.bbox_y2,
                confidence=item.confidence,
            )
            for item in prepared.observation_rows
        ]
        written = 0
        for start in range(0, len(observation_rows), self.config.observation_batch_size):
            batch = observation_rows[start : start + self.config.observation_batch_size]
            if not batch:
                continue
            inserted = self.repository.add_vehicle_observations(batch)
            written += len(inserted)
        return TrackPersistenceResult(track_uuid=track.track_uuid, status="inserted", database_track_id=str(created_track.id), observations_written=written)

    def _apply_metrics(self, result: TrackPersistenceResult) -> None:
        if result.status == "inserted":
            self.metrics.tracks_inserted += 1
            self.metrics.observations_written += result.observations_written
        elif result.status == "already_exists":
            self.metrics.tracks_already_existing += 1
        elif result.status == "skipped_discarded":
            self.metrics.tracks_skipped_discarded += 1
        elif result.status == "skipped_invalid_state":
            self.metrics.tracks_skipped_invalid_state += 1
        elif result.status == "dry_run":
            self.metrics.observations_written += result.observations_written
        elif result.status == "failed":
            self.metrics.tracks_failed += 1
            if result.error:
                self.metrics.errors.append(result.error)

    def _validate_track(self, track: LocalVehicleTrack) -> None:
        if not str(track.track_uuid).strip():
            raise PersistenceValidationError("track_uuid must be present.")
        if track.camera_code not in self.camera_id_by_code:
            raise PersistenceValidationError(f"camera_code is not known: {track.camera_code}")
        if int(track.local_track_id) < 0:
            raise PersistenceValidationError("local_track_id must be non-negative.")
        if not is_supported_vehicle_class(track.class_name):
            raise PersistenceValidationError(f"Unsupported vehicle class: {track.class_name}")
        if track.first_seen_at is None or track.last_seen_at is None:
            raise PersistenceValidationError("Track timestamps must be present.")
        if track.last_seen_at < track.first_seen_at:
            raise PersistenceValidationError("Track last_seen_at must not be before first_seen_at.")
        if int(track.first_frame_number) > int(track.last_frame_number):
            raise PersistenceValidationError("first_frame_number must not be after last_frame_number.")
        if int(track.observation_count) < 0:
            raise PersistenceValidationError("observation_count must be non-negative.")
        if not math.isfinite(float(track.best_confidence)):
            raise PersistenceValidationError("best_confidence must be finite.")

    def _select_observation_rows(self, track: LocalVehicleTrack, *, database_track_id: object) -> list[VehicleObservationRecord]:
        selected = self._select_observations(track.observations)
        rows: list[VehicleObservationRecord] = []
        for observation in selected:
            self._validate_observation(track, observation)
            x1, y1, x2, y2 = observation.bbox_xyxy
            rows.append(
                VehicleObservationRecord(
                    vehicle_track_id=database_track_id,
                    frame_number=observation.frame_number,
                    observed_at=observation.camera_timestamp,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    confidence=observation.confidence,
                )
            )
        return rows

    def _select_observations(self, observations: list[TrackObservation]) -> list[TrackObservation]:
        if self.config.observation_mode == "none" or not observations:
            return []
        if self.config.observation_mode == "all":
            return list(observations)
        sample_every = self.config.observation_sample_every_n
        selected: list[TrackObservation] = []
        for index, observation in enumerate(observations):
            is_first = index == 0
            is_last = index == len(observations) - 1
            if is_first or is_last or (index % sample_every == 0):
                selected.append(observation)
        deduped: list[TrackObservation] = []
        seen_keys: set[tuple[int, float]] = set()
        for observation in selected:
            key = (observation.frame_number, observation.video_time_seconds)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(observation)
        return deduped

    def _validate_observation(self, track: LocalVehicleTrack, observation: TrackObservation) -> None:
        if int(observation.frame_number) < 0:
            raise PersistenceValidationError("Observation frame_number must be non-negative.")
        if observation.camera_timestamp is None:
            raise PersistenceValidationError("Observation camera_timestamp must be present.")
        if observation.camera_code != track.camera_code:
            raise PersistenceValidationError("Observation camera_code does not match parent track.")
        if int(observation.local_track_id) != int(track.local_track_id):
            raise PersistenceValidationError("Observation local_track_id does not match parent track.")
        if not math.isfinite(float(observation.confidence)):
            raise PersistenceValidationError("Observation confidence must be finite.")
        x1, y1, x2, y2 = observation.bbox_xyxy
        values = (float(x1), float(y1), float(x2), float(y2))
        if any(not math.isfinite(value) for value in values):
            raise PersistenceValidationError("Observation bbox values must be finite.")
        if x2 <= x1 or y2 <= y1:
            raise PersistenceValidationError("Observation bbox must satisfy x2 > x1 and y2 > y1.")
