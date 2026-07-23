from __future__ import annotations

from pathlib import Path

from ..database.client import create_backend_client
from ..database.config import DatabaseConfig
from ..database.repository import SupabaseVehicleRepository, VehicleRepository
from ..detection.detection_config import DetectionConfig
from ..tracking.tracking_config import TrackingConfig
from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_persistence_service import AnalyticsPersistenceService
from .persistence_config import PersistenceConfig
from .persistence_service_protocol import PersistenceServiceProtocol
from .tracking_persistence_service import TrackingPersistenceService


def build_persistence_service(
    *,
    config: PersistenceConfig,
    run_code: str,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    execution_mode: str,
    runtime_device: str | None,
    artifact_root: Path,
    repository: VehicleRepository | None = None,
    analytics_client: AnalyticsDatabaseClient | None = None,
) -> PersistenceServiceProtocol | None:
    if config.backend == "disabled":
        return None
    if config.backend in {"dry_run", "analytics_supabase"}:
        client = analytics_client
        if config.backend == "analytics_supabase":
            client = client or AnalyticsDatabaseClient(schema_name="analytics")
        return AnalyticsPersistenceService(
            client,
            config,
            run_code=run_code,
            detection_config=detection_config,
            tracking_config=tracking_config,
            execution_mode=execution_mode,
            runtime_device=runtime_device,
            artifact_root=artifact_root,
            enable_database_writes=(config.backend == "analytics_supabase"),
        )
    return TrackingPersistenceService(
        repository or _build_old_public_repository(),
        config,
    )


def _build_old_public_repository() -> VehicleRepository:
    database_config = DatabaseConfig.from_env(require_backend_credentials=True)
    return SupabaseVehicleRepository(create_backend_client(database_config))
