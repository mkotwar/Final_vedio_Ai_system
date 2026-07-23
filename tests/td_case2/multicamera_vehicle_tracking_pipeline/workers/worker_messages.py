from __future__ import annotations

from dataclasses import dataclass

from ..persistence.persistence_models import TrackPersistenceResult
from ..tracking.tracking_models import LocalVehicleTrack


@dataclass(frozen=True, slots=True)
class EndOfCameraMessage:
    camera_code: str


@dataclass(frozen=True, slots=True)
class EndOfInputMessage:
    reason: str = "all_cameras_finished"


@dataclass(frozen=True, slots=True)
class WorkerErrorMessage:
    worker_name: str
    worker_type: str
    camera_code: str | None
    error_type: str
    error_message: str
    traceback_text: str | None
    fatal: bool


@dataclass(frozen=True, slots=True)
class CompletedTrackMessage:
    camera_code: str
    track: LocalVehicleTrack


@dataclass(frozen=True, slots=True)
class VehicleColourJobMessage:
    camera_code: str
    track: LocalVehicleTrack
    persistence_result: TrackPersistenceResult


@dataclass(frozen=True, slots=True)
class AnprJobMessage:
    camera_code: str
    track: LocalVehicleTrack
    persistence_result: TrackPersistenceResult
