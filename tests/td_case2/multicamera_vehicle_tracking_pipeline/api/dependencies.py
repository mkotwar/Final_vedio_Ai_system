from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request

from ..persistence.analytics_database_client import AnalyticsDatabaseClient, AnalyticsDatabaseClientConfig
from ..persistence.api_read_repository import AnalyticsReadRepository
from .services.camera_service import CameraService
from .services.global_vehicle_service import GlobalVehicleService
from .services.match_service import MatchService
from .services.media_service import MediaService
from .services.run_service import RunService
from .services.track_service import TrackService
from .settings import ApiSettings


SENSITIVE_KEYS = {
    "service_role_key",
    "api_key",
    "token",
    "authorization",
    "model_path",
    "processor_path",
    "adapter_path",
    "local_absolute_path",
}


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in SENSITIVE_KEYS:
                continue
            sanitized[key] = sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    return value


def build_repository_from_settings(settings: ApiSettings) -> AnalyticsReadRepository:
    settings.validate_runtime()
    config = AnalyticsDatabaseClientConfig(
        supabase_url=settings.supabase_url,
        supabase_service_role_key=settings.supabase_service_role_key,
        schema_name="analytics",
    )
    client = AnalyticsDatabaseClient(config=config, schema_name="analytics")
    return AnalyticsReadRepository(client)


def get_repository(request: Request) -> AnalyticsReadRepository:
    return request.app.state.repository


def get_run_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> RunService:
    return RunService(repository)


def get_camera_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> CameraService:
    return CameraService(repository)


def get_track_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> TrackService:
    return TrackService(repository)


def get_global_vehicle_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> GlobalVehicleService:
    return GlobalVehicleService(repository)


def get_match_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> MatchService:
    return MatchService(repository)


def get_media_service(repository: AnalyticsReadRepository = Depends(get_repository)) -> MediaService:
    return MediaService(repository)
