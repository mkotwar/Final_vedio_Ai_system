from __future__ import annotations

from typing import Any
from uuid import UUID

from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase
from .persistence_models import (
    AiModelRecord,
    CameraRecord,
    CameraRunRecord,
    PlateDetectionRecord,
    PlateReadingRecord,
    PlateSummaryRecord,
    ProcessingErrorRecord,
    ProcessingJobRecord,
    ProcessingRunRecord,
    RunModelRecord,
    TrackObservationRecord,
    VehicleTrackRecord,
    VideoSourceRecord,
)


class AnalyticsCameraRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="camera")

    def upsert_camera(self, record: CameraRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="camera_code").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_camera", message=f"Failed to upsert analytics.camera: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_camera")

    def get_camera_by_code(self, camera_code: str) -> dict[str, Any] | None:
        try:
            response = self._table().select("*").eq("camera_code", camera_code).limit(1).execute()
        except Exception as exc:
            raise self._wrap_error(operation="get_camera_by_code", message=f"Failed to query analytics.camera by camera_code: {exc}", cause=exc) from exc
        rows = self._extract_rows(response)
        return rows[0] if rows else None


class AnalyticsVideoSourceRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="video_source")

    def create_video_source(self, record: VideoSourceRecord) -> dict[str, Any]:
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="create_video_source", message=f"Failed to insert analytics.video_source: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="create_video_source")


class AnalyticsProcessingRunRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="processing_run")

    def upsert_processing_run(self, record: ProcessingRunRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="run_code").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_processing_run", message=f"Failed to upsert analytics.processing_run: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_processing_run")


class AnalyticsCameraRunRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="camera_run")

    def upsert_camera_run(self, record: CameraRunRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="processing_run_id,camera_id").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_camera_run", message=f"Failed to upsert analytics.camera_run: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_camera_run")

    def update_camera_run_by_id(self, camera_run_id: UUID, record: CameraRunRecord) -> dict[str, Any]:
        try:
            response = self._table().update(record.to_payload()).eq("id", str(camera_run_id)).execute()
        except Exception as exc:
            raise self._wrap_error(operation="update_camera_run_by_id", message=f"Failed to update analytics.camera_run: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="update_camera_run_by_id")


class AnalyticsProcessingJobRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="processing_job")

    def create_processing_job(self, record: ProcessingJobRecord) -> dict[str, Any]:
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="create_processing_job", message=f"Failed to insert analytics.processing_job: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="create_processing_job")

    def update_processing_job_by_id(self, job_id: UUID, record: ProcessingJobRecord) -> dict[str, Any]:
        try:
            response = self._table().update(record.to_payload()).eq("id", str(job_id)).execute()
        except Exception as exc:
            raise self._wrap_error(operation="update_processing_job_by_id", message=f"Failed to update analytics.processing_job: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="update_processing_job_by_id")


class AnalyticsVehicleTrackRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="vehicle_track")

    def upsert_vehicle_track(self, record: VehicleTrackRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="track_uuid").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_vehicle_track", message=f"Failed to upsert analytics.vehicle_track: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_vehicle_track")

    def get_vehicle_track_by_uuid(self, track_uuid: str) -> dict[str, Any] | None:
        try:
            response = self._table().select("*").eq("track_uuid", track_uuid).limit(1).execute()
        except Exception as exc:
            raise self._wrap_error(operation="get_vehicle_track_by_uuid", message=f"Failed to query analytics.vehicle_track by track_uuid: {exc}", cause=exc) from exc
        rows = self._extract_rows(response)
        return rows[0] if rows else None


class AnalyticsTrackObservationRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="track_observation")

    def upsert_observations_batch(self, records: list[TrackObservationRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        payload = [record.to_payload() for record in records]
        try:
            response = self._table().upsert(payload, on_conflict="vehicle_track_id,frame_number").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_observations_batch", message=f"Failed to upsert analytics.track_observation batch: {exc}", cause=exc) from exc
        return self._extract_rows(response)


class AnalyticsAiModelRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="ai_model")

    def upsert_ai_model(self, record: AiModelRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="model_code").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_ai_model", message=f"Failed to upsert analytics.ai_model: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_ai_model")


class AnalyticsRunModelRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="run_model")

    def upsert_run_model(self, record: RunModelRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="processing_run_id,ai_model_id,stage_name").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_run_model", message=f"Failed to upsert analytics.run_model: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_run_model")


class AnalyticsProcessingErrorRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="processing_error")

    def create_processing_error(self, record: ProcessingErrorRecord) -> dict[str, Any]:
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="create_processing_error", message=f"Failed to insert analytics.processing_error: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="create_processing_error")


class AnalyticsPlateDetectionRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="plate_detection")

    def create_plate_detection(self, record: PlateDetectionRecord) -> dict[str, Any]:
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="create_plate_detection", message=f"Failed to insert analytics.plate_detection: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="create_plate_detection")


class AnalyticsPlateReadingRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="plate_reading")

    def create_plate_reading(self, record: PlateReadingRecord) -> dict[str, Any]:
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="create_plate_reading", message=f"Failed to insert analytics.plate_reading: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="create_plate_reading")


class AnalyticsPlateSummaryRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="plate_summary")

    def upsert_plate_summary(self, record: PlateSummaryRecord) -> dict[str, Any]:
        try:
            response = self._table().upsert(record.to_payload(), on_conflict="vehicle_track_id").execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert_plate_summary", message=f"Failed to upsert analytics.plate_summary: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert_plate_summary")
