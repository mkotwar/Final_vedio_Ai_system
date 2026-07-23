from __future__ import annotations

from typing import Protocol

from ..ingestion.camera_config import CameraConfig
from ..tracking.tracking_models import LocalVehicleTrack
from .persistence_models import PersistenceRunMetrics, TrackPersistenceResult


class PersistenceServiceProtocol(Protocol):
    camera_id_by_code: dict[str, object]

    def sync_cameras(self, camera_configs: list[CameraConfig]) -> dict[str, object]: ...

    def save_completed_track(self, track: LocalVehicleTrack) -> TrackPersistenceResult: ...

    def get_metrics(self) -> PersistenceRunMetrics: ...
