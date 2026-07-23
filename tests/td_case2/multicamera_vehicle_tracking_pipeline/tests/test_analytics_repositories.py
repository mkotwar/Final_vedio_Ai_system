from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_repository_base import AnalyticsRepositoryError
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_repositories import (
    AnalyticsAiModelRepository,
    AnalyticsCameraRepository,
    AnalyticsCameraRunRepository,
    AnalyticsProcessingErrorRepository,
    AnalyticsProcessingJobRepository,
    AnalyticsProcessingRunRepository,
    AnalyticsRunModelRepository,
    AnalyticsTrackObservationRepository,
    AnalyticsVehicleTrackRepository,
    AnalyticsVideoSourceRepository,
)
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_models import (
    AiModelRecord,
    CameraRecord,
    CameraRunRecord,
    ProcessingErrorRecord,
    ProcessingJobRecord,
    ProcessingRunRecord,
    RunModelRecord,
    TrackObservationRecord,
    VehicleTrackRecord,
    VideoSourceRecord,
)


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, response_data: list[dict[str, object]] | None = None, *, should_fail: bool = False):
        self.table_name = table_name
        self.response_data = response_data if response_data is not None else []
        self.should_fail = should_fail
        self.calls: list[tuple[str, object]] = []

    def insert(self, payload):
        self.calls.append(("insert", payload))
        return self

    def upsert(self, payload, on_conflict=None):
        self.calls.append(("upsert", {"payload": payload, "on_conflict": on_conflict}))
        return self

    def update(self, payload):
        self.calls.append(("update", payload))
        return self

    def select(self, payload):
        self.calls.append(("select", payload))
        return self

    def eq(self, field, value):
        self.calls.append(("eq", (field, value)))
        return self

    def limit(self, value):
        self.calls.append(("limit", value))
        return self

    def execute(self):
        self.calls.append(("execute", None))
        if self.should_fail:
            raise RuntimeError(f"boom:{self.table_name}")
        return _FakeResponse(self.response_data)


class _FakeAnalyticsClient:
    def __init__(self, table_map: dict[str, _FakeQuery]):
        self.table_map = table_map
        self.tables_requested: list[str] = []

    def table(self, table_name: str):
        self.tables_requested.append(table_name)
        return self.table_map[table_name]


def _utc() -> datetime:
    return datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


class AnalyticsRepositoryTests(unittest.TestCase):
    def test_camera_repository_uses_camera_code_upsert(self) -> None:
        query = _FakeQuery("camera", [{"id": "cam-id", "camera_code": "CAM_001"}])
        repo = AnalyticsCameraRepository(_FakeAnalyticsClient({"camera": query}))
        row = repo.upsert_camera(CameraRecord(camera_code="CAM_001"))
        self.assertEqual(row["camera_code"], "CAM_001")
        self.assertEqual(query.calls[0][0], "upsert")
        self.assertEqual(query.calls[0][1]["on_conflict"], "camera_code")

    def test_video_source_repository_inserts_row(self) -> None:
        query = _FakeQuery("video_source", [{"id": "vs-id"}])
        repo = AnalyticsVideoSourceRepository(_FakeAnalyticsClient({"video_source": query}))
        row = repo.create_video_source(
            VideoSourceRecord(
                camera_id=uuid4(),
                source_type="LOCAL_FILE",
                source_reference="source/videos/CAM_001/test.mp4",
            )
        )
        self.assertEqual(row["id"], "vs-id")
        self.assertEqual(query.calls[0][0], "insert")

    def test_processing_run_repository_upserts_on_run_code(self) -> None:
        query = _FakeQuery("processing_run", [{"id": "run-id", "run_code": "RUN_1"}])
        repo = AnalyticsProcessingRunRepository(_FakeAnalyticsClient({"processing_run": query}))
        repo.upsert_processing_run(ProcessingRunRecord(run_code="RUN_1", execution_mode="THREADED", status="RUNNING"))
        self.assertEqual(query.calls[0][1]["on_conflict"], "run_code")

    def test_camera_run_repository_upserts_on_processing_run_and_camera(self) -> None:
        query = _FakeQuery("camera_run", [{"id": "camera-run-id"}])
        repo = AnalyticsCameraRunRepository(_FakeAnalyticsClient({"camera_run": query}))
        repo.upsert_camera_run(CameraRunRecord(processing_run_id=uuid4(), camera_id=uuid4(), status="RUNNING"))
        self.assertEqual(query.calls[0][1]["on_conflict"], "processing_run_id,camera_id")

    def test_processing_job_repository_inserts_and_updates(self) -> None:
        insert_query = _FakeQuery("processing_job", [{"id": "job-id"}])
        repo = AnalyticsProcessingJobRepository(_FakeAnalyticsClient({"processing_job": insert_query}))
        row = repo.create_processing_job(
            ProcessingJobRecord(
                processing_run_id=uuid4(),
                job_type="PERSIST",
                status="RUNNING",
                started_at=_utc(),
            )
        )
        self.assertEqual(row["id"], "job-id")
        update_query = _FakeQuery("processing_job", [{"id": "job-id", "status": "COMPLETED"}])
        repo = AnalyticsProcessingJobRepository(_FakeAnalyticsClient({"processing_job": update_query}))
        row = repo.update_processing_job_by_id(
            uuid4(),
            ProcessingJobRecord(
                processing_run_id=uuid4(),
                job_type="PERSIST",
                status="COMPLETED",
                started_at=_utc(),
                completed_at=_utc(),
            ),
        )
        self.assertEqual(row["status"], "COMPLETED")
        self.assertEqual(update_query.calls[0][0], "update")

    def test_vehicle_track_repository_uses_track_uuid_idempotency(self) -> None:
        query = _FakeQuery("vehicle_track", [{"id": "track-id", "track_uuid": "RUN:CAM:TRACK_1"}])
        repo = AnalyticsVehicleTrackRepository(_FakeAnalyticsClient({"vehicle_track": query}))
        repo.upsert_vehicle_track(
            VehicleTrackRecord(
                processing_run_id=uuid4(),
                camera_run_id=uuid4(),
                camera_id=uuid4(),
                track_uuid="RUN:CAM:TRACK_1",
                local_track_id=1,
                vehicle_class="3Wheeler",
                first_seen_at=_utc(),
                last_seen_at=_utc(),
                first_frame_number=1,
                last_frame_number=1,
                tracker_backend="supervision_bytetrack",
            )
        )
        self.assertEqual(query.calls[0][1]["on_conflict"], "track_uuid")

    def test_track_observation_repository_batches_upsert(self) -> None:
        query = _FakeQuery("track_observation", [{"id": 1}, {"id": 2}])
        repo = AnalyticsTrackObservationRepository(_FakeAnalyticsClient({"track_observation": query}))
        rows = repo.upsert_observations_batch(
            [
                TrackObservationRecord(vehicle_track_id=uuid4(), camera_id=uuid4(), frame_number=1, observed_at=_utc(), bbox_x1=1.0, bbox_y1=2.0, bbox_x2=3.0, bbox_y2=4.0),
                TrackObservationRecord(vehicle_track_id=uuid4(), camera_id=uuid4(), frame_number=2, observed_at=_utc(), bbox_x1=2.0, bbox_y1=3.0, bbox_x2=4.0, bbox_y2=5.0),
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(query.calls[0][0], "upsert")
        self.assertIsInstance(query.calls[0][1]["payload"], list)
        self.assertEqual(query.calls[0][1]["on_conflict"], "vehicle_track_id,frame_number")

    def test_ai_model_and_run_model_use_business_keys(self) -> None:
        ai_query = _FakeQuery("ai_model", [{"id": "model-id"}])
        ai_repo = AnalyticsAiModelRepository(_FakeAnalyticsClient({"ai_model": ai_query}))
        ai_repo.upsert_ai_model(AiModelRecord(model_code="detector", model_name="Detector"))
        self.assertEqual(ai_query.calls[0][1]["on_conflict"], "model_code")

        run_model_query = _FakeQuery("run_model", [{"id": "run-model-id"}])
        run_model_repo = AnalyticsRunModelRepository(_FakeAnalyticsClient({"run_model": run_model_query}))
        run_model_repo.upsert_run_model(
            RunModelRecord(processing_run_id=uuid4(), ai_model_id=uuid4(), stage_name="DETECT")
        )
        self.assertEqual(run_model_query.calls[0][1]["on_conflict"], "processing_run_id,ai_model_id,stage_name")

    def test_processing_error_repository_inserts_row(self) -> None:
        query = _FakeQuery("processing_error", [{"id": "err-id"}])
        repo = AnalyticsProcessingErrorRepository(_FakeAnalyticsClient({"processing_error": query}))
        row = repo.create_processing_error(ProcessingErrorRecord(severity="ERROR", message="boom"))
        self.assertEqual(row["id"], "err-id")
        self.assertEqual(query.calls[0][0], "insert")

    def test_repository_wraps_low_level_failures(self) -> None:
        query = _FakeQuery("camera", should_fail=True)
        repo = AnalyticsCameraRepository(_FakeAnalyticsClient({"camera": query}))
        with self.assertRaises(AnalyticsRepositoryError) as ctx:
            repo.upsert_camera(CameraRecord(camera_code="CAM_001"))
        self.assertEqual(ctx.exception.table_name, "camera")
        self.assertEqual(ctx.exception.operation, "upsert_camera")


if __name__ == "__main__":
    unittest.main()
