from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_database_client import AnalyticsDatabaseClientError
from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import verify_enrichment_run


class _FakeResponse:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _FakeQuery:
    def __init__(self, table_name: str, dataset: dict[str, list[dict]], errors: dict[str, Exception], tracker: dict[str, object]):
        self._table_name = table_name
        self._dataset = dataset
        self._errors = errors
        self._tracker = tracker
        self._eq_filters: list[tuple[str, object]] = []
        self._in_filters: list[tuple[str, list[object]]] = []
        self._limit: int | None = None
        self._count_requested = False

    def select(self, _columns, count=None):
        self._count_requested = count == "exact"
        return self

    def eq(self, key, value):
        self._eq_filters.append((key, value))
        return self

    def in_(self, key, values):
        self._in_filters.append((key, list(values)))
        return self

    def limit(self, value):
        self._limit = value
        return self

    def execute(self):
        self._tracker["executed_tables"].append(self._table_name)
        error = self._errors.get(self._table_name)
        if error is not None:
            raise error
        rows = [dict(row) for row in self._dataset.get(self._table_name, [])]
        for key, value in self._eq_filters:
            rows = [row for row in rows if row.get(key) == value]
        for key, values in self._in_filters:
            rows = [row for row in rows if row.get(key) in values]
        if self._limit is not None:
            rows = rows[: self._limit]
        count = len(rows) if self._count_requested else None
        return _FakeResponse(rows, count=count)

    def insert(self, *_args, **_kwargs):
        raise AssertionError("Read-only verifier must not insert.")

    def update(self, *_args, **_kwargs):
        raise AssertionError("Read-only verifier must not update.")

    def upsert(self, *_args, **_kwargs):
        raise AssertionError("Read-only verifier must not upsert.")

    def delete(self, *_args, **_kwargs):
        raise AssertionError("Read-only verifier must not delete.")


class _FakeAnalyticsClient:
    def __init__(self, dataset=None, errors=None, schema_name="analytics"):
        self.dataset = dataset or {}
        self.errors = errors or {}
        self.schema_name = schema_name
        self.tracker = {"executed_tables": []}

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.dataset, self.errors, self.tracker)


def _base_dataset():
    return {
        "processing_run": [
            {
                "id": "run-1",
                "run_code": "RUN_20260724_151402",
                "status": "COMPLETED",
                "started_at": "2026-07-24T15:14:02Z",
                "completed_at": "2026-07-24T15:15:30Z",
                "created_at": "2026-07-24T15:14:01Z",
            }
        ],
        "camera_run": [
            {"id": "cam-run-1", "processing_run_id": "run-1", "camera_id": "cam-1", "video_source_id": "vs-1", "status": "COMPLETED", "frames_read": 200, "frames_processed": 143, "completed_tracks_count": 2, "discarded_tracks_count": 0},
            {"id": "cam-run-2", "processing_run_id": "run-1", "camera_id": "cam-2", "video_source_id": "vs-2", "status": "COMPLETED", "frames_read": 200, "frames_processed": 143, "completed_tracks_count": 1, "discarded_tracks_count": 0},
        ],
        "processing_job": [
            {"id": "job-1", "processing_run_id": "run-1"},
            {"id": "job-2", "processing_run_id": "run-1"},
        ],
        "camera": [
            {"id": "cam-1", "camera_code": "CAM_001", "camera_name": "Gate 1"},
            {"id": "cam-2", "camera_code": "CAM_002", "camera_name": "Gate 2"},
        ],
        "video_source": [
            {"id": "vs-1", "camera_id": "cam-1", "source_reference": "data/1test.mp4"},
            {"id": "vs-2", "camera_id": "cam-2", "source_reference": "data/2test.mp4"},
        ],
        "vehicle_track": [
            {"id": "track-1", "processing_run_id": "run-1", "camera_run_id": "cam-run-1", "camera_id": "cam-1", "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4", "local_track_id": 4, "vehicle_class": "CAR", "first_seen_at": "2026-07-24T15:14:10Z", "last_seen_at": "2026-07-24T15:14:20Z", "first_frame_number": 10, "last_frame_number": 30, "first_video_time_seconds": 1.0, "last_video_time_seconds": 3.0, "observation_count": 3, "lifecycle_state": "COMPLETED"},
            {"id": "track-2", "processing_run_id": "run-1", "camera_run_id": "cam-run-2", "camera_id": "cam-2", "track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4", "local_track_id": 4, "vehicle_class": "CAR", "first_seen_at": "2026-07-24T15:14:11Z", "last_seen_at": "2026-07-24T15:14:21Z", "first_frame_number": 11, "last_frame_number": 31, "first_video_time_seconds": 1.1, "last_video_time_seconds": 3.1, "observation_count": 2, "lifecycle_state": "COMPLETED"},
            {"id": "track-3", "processing_run_id": "run-1", "camera_run_id": "cam-run-1", "camera_id": "cam-1", "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_1", "local_track_id": 1, "vehicle_class": "BUS", "first_seen_at": "2026-07-24T15:14:05Z", "last_seen_at": "2026-07-24T15:14:09Z", "first_frame_number": 1, "last_frame_number": 9, "first_video_time_seconds": 0.1, "last_video_time_seconds": 0.9, "observation_count": 2, "lifecycle_state": "COMPLETED"},
        ],
        "track_observation": [
            {"id": "obs-1", "vehicle_track_id": "track-1", "camera_id": "cam-1", "frame_number": 10, "observed_at": "t1", "video_time_seconds": 1.0, "detection_confidence": 0.6, "tracker_confidence": 0.7, "is_key_observation": False},
            {"id": "obs-2", "vehicle_track_id": "track-1", "camera_id": "cam-1", "frame_number": 20, "observed_at": "t2", "video_time_seconds": 2.0, "detection_confidence": 0.9, "tracker_confidence": 0.8, "is_key_observation": True},
            {"id": "obs-3", "vehicle_track_id": "track-1", "camera_id": "cam-1", "frame_number": 30, "observed_at": "t3", "video_time_seconds": 3.0, "detection_confidence": 0.8, "tracker_confidence": 0.9, "is_key_observation": False},
            {"id": "obs-4", "vehicle_track_id": "track-2", "camera_id": "cam-2", "frame_number": 11, "observed_at": "t4", "video_time_seconds": 1.1, "detection_confidence": 0.8, "tracker_confidence": 0.8, "is_key_observation": True},
            {"id": "obs-5", "vehicle_track_id": "track-2", "camera_id": "cam-2", "frame_number": 31, "observed_at": "t5", "video_time_seconds": 3.1, "detection_confidence": 0.85, "tracker_confidence": 0.8, "is_key_observation": False},
            {"id": "obs-6", "vehicle_track_id": "track-3", "camera_id": "cam-1", "frame_number": 1, "observed_at": "t6", "video_time_seconds": 0.1, "detection_confidence": 0.5, "tracker_confidence": 0.7, "is_key_observation": False},
            {"id": "obs-7", "vehicle_track_id": "track-3", "camera_id": "cam-1", "frame_number": 9, "observed_at": "t7", "video_time_seconds": 0.9, "detection_confidence": 0.55, "tracker_confidence": 0.7, "is_key_observation": False},
        ],
        "track_media": [
            {"id": "media-1", "vehicle_track_id": "track-1", "media_type": "BEST_VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "RUN_1/CAM_001/track_4/best_vehicle.jpg", "frame_number": 20, "video_time_seconds": 2.0, "quality_score": 0.95, "sharpness_score": 0.96, "visibility_score": 0.91, "selection_rank": 1, "is_primary": True, "width": 120, "height": 80},
            {"id": "media-2", "vehicle_track_id": "track-1", "media_type": "PLATE_CROP", "storage_provider": "LOCAL", "storage_uri": "RUN_1/CAM_001/track_4/plate.jpg", "frame_number": 20, "video_time_seconds": 2.0, "quality_score": 0.8, "sharpness_score": 0.85, "visibility_score": 0.8, "selection_rank": 1, "is_primary": True, "width": 60, "height": 20},
            {"id": "media-3", "vehicle_track_id": "track-2", "media_type": "BEST_VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "RUN_1/CAM_002/track_4/best_vehicle.jpg", "frame_number": 11, "video_time_seconds": 1.1, "quality_score": 0.9, "sharpness_score": 0.92, "visibility_score": 0.9, "selection_rank": 1, "is_primary": True, "width": 120, "height": 80},
            {"id": "media-4", "vehicle_track_id": "track-3", "media_type": "VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "RUN_1/CAM_001/track_1/vehicle.jpg", "frame_number": 1, "video_time_seconds": 0.1, "quality_score": 0.6, "sharpness_score": 0.6, "visibility_score": 0.6, "selection_rank": 3, "is_primary": False, "width": 100, "height": 70},
        ],
        "vehicle_attribute": [
            {"id": "attr-1", "vehicle_track_id": "track-1", "primary_color": "GREY", "secondary_color": "BLACK", "color_confidence": 0.91, "attribute_source": "florence", "attribute_status": "CURRENT", "metadata": {"backend": "florence", "source_storage_uri": "RUN_1/CAM_001/track_4/best_vehicle.jpg", "raw_model_output": "grey car", "model_path": "C:/secret/model"}},
            {"id": "attr-2", "vehicle_track_id": "track-2", "primary_color": "GREY", "secondary_color": None, "color_confidence": 0.9, "attribute_source": "florence", "attribute_status": "CURRENT", "metadata": {"backend": "florence", "source_storage_uri": "RUN_1/CAM_002/track_4/best_vehicle.jpg"}},
            {"id": "attr-3", "vehicle_track_id": "track-3", "primary_color": "UNKNOWN", "secondary_color": None, "color_confidence": 0.4, "attribute_source": "florence", "attribute_status": "CURRENT", "metadata": {"backend": "florence"}},
        ],
        "plate_detection": [
            {"id": "pd-1", "vehicle_track_id": "track-1", "track_media_id": "media-2", "confidence": 0.88, "bbox_x1": 1, "bbox_y1": 2, "bbox_x2": 3, "bbox_y2": 4},
            {"id": "pd-2", "vehicle_track_id": "track-2", "track_media_id": "media-3", "confidence": 0.89, "bbox_x1": 1, "bbox_y1": 2, "bbox_x2": 3, "bbox_y2": 4},
        ],
        "plate_reading": [
            {"id": "pr-1", "plate_detection_id": "pd-1", "raw_text": "DL8CBF6268", "normalized_text": "DL8CBF6268", "plate_pattern": "STANDARD", "confidence": 0.95, "status": "VERIFIED", "metadata": {"matched_pattern": "STANDARD"}},
            {"id": "pr-2", "plate_detection_id": "pd-2", "raw_text": "DL8CBF6268", "normalized_text": "DL8CBF6268", "plate_pattern": "STANDARD", "confidence": 0.94, "status": "VERIFIED", "metadata": {"matched_pattern": "STANDARD"}},
        ],
        "plate_summary": [
            {"id": "ps-1", "vehicle_track_id": "track-1", "status": "VERIFIED"},
            {"id": "ps-2", "vehicle_track_id": "track-2", "status": "VERIFIED"},
        ],
        "processing_error": [],
    }


class VerifyEnrichmentRunTests(unittest.TestCase):
    def _run_cli(self, argv, dataset=None, errors=None):
        fake_client = _FakeAnalyticsClient(dataset=dataset or _base_dataset(), errors=errors or {})
        output = io.StringIO()
        with patch.object(verify_enrichment_run, "AnalyticsDatabaseClient", return_value=fake_client), contextlib.redirect_stdout(output):
            exit_code = verify_enrichment_run.run(argv)
        return exit_code, output.getvalue(), fake_client

    def test_missing_supabase_url_returns_exit_code_2(self):
        with patch.object(verify_enrichment_run, "AnalyticsDatabaseClient", side_effect=AnalyticsDatabaseClientError("Missing required environment variables: SUPABASE_URL", code="missing")), patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": "secret"}, clear=False):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = verify_enrichment_run.run(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_CONFIGURATION_MISSING)
        self.assertIn("SUPABASE_URL=NOT SET", output.getvalue())

    def test_missing_service_role_key_returns_exit_code_2(self):
        with patch.object(verify_enrichment_run, "AnalyticsDatabaseClient", side_effect=AnalyticsDatabaseClientError("Missing required environment variables: SUPABASE_SERVICE_ROLE_KEY", code="missing")), patch.dict("os.environ", {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = verify_enrichment_run.run(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_CONFIGURATION_MISSING)
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY=NOT SET", output.getvalue())

    def test_run_not_found_returns_exit_code_1(self):
        dataset = _base_dataset()
        dataset["processing_run"] = []
        exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402"], dataset=dataset)
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_QUERY_FAILED)
        self.assertIn("Run not found", output)

    def test_one_valid_run_returns_success(self):
        exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertIn("Scope: Run", output)

    def test_count_queries_succeed(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5)
        self.assertEqual(report["counts"]["vehicle_track"]["count"], 3)
        self.assertEqual(report["counts"]["processing_error"]["count"], 0)

    def test_one_table_query_fails(self):
        exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402"], errors={"track_media": RuntimeError("track_media unavailable")})
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_QUERY_FAILED)
        self.assertIn("track_media unavailable", output)

    def test_no_tracks(self):
        dataset = _base_dataset()
        for table in ("vehicle_track", "track_observation", "track_media", "vehicle_attribute", "plate_detection", "plate_reading", "plate_summary"):
            dataset[table] = []
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5)
        self.assertEqual(report["counts"]["vehicle_track"]["count"], 0)
        self.assertEqual(report["coverage"]["media_coverage"], 0.0)

    def test_zero_safe_coverage_calculations(self):
        self.assertEqual(verify_enrichment_run._percentage(0, 0), 0.0)

    def test_colour_coverage_calculation(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5)
        self.assertEqual(report["coverage"]["colour_coverage"], 100.0)

    def test_plate_coverage_calculation(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5)
        self.assertEqual(report["coverage"]["plate_detection_coverage"], 66.67)

    def test_verified_plate_coverage_calculation(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5)
        self.assertEqual(report["coverage"]["verified_plate_coverage"], 66.67)

    def test_media_coverage_calculation(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5)
        self.assertEqual(report["coverage"]["media_coverage"], 100.0)

    def test_orphan_plate_reading_detected(self):
        dataset = _base_dataset()
        dataset["plate_reading"].append({"id": "pr-x", "plate_detection_id": "missing", "status": "UNKNOWN"})
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5)
        self.assertEqual(report["consistency"]["plate_readings_without_known_plate_detection"], 1)

    def test_duplicate_track_uuid_detected(self):
        dataset = _base_dataset()
        dataset["vehicle_track"].append({**dataset["vehicle_track"][0], "id": "track-x"})
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5)
        self.assertEqual(report["consistency"]["duplicate_track_uuids"], 1)

    def test_strict_mode_returns_exit_code_3(self):
        dataset = _base_dataset()
        dataset["track_media"] = []
        exit_code, _output, _client = self._run_cli(["--run-code", "RUN_20260724_151402", "--strict"], dataset=dataset)
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_STRICT_FAILED)

    def test_non_strict_mode_reports_inconsistency_but_returns_0(self):
        dataset = _base_dataset()
        dataset["track_media"] = []
        exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402"], dataset=dataset)
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertIn("Verification Status: FAIL", output)

    def test_json_output_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "audit.json"
            exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402", "--json-output", str(output_path)])
            self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run"]["run_code"], "RUN_20260724_151402")

    def test_credentials_never_appear_in_output(self):
        with patch.dict("os.environ", {"SUPABASE_URL": "https://secret-project.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "super-secret-key"}, clear=False):
            exit_code, output, _client = self._run_cli(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertNotIn("https://secret-project.supabase.co", output)
        self.assertNotIn("super-secret-key", output)

    def test_no_write_methods_are_called(self):
        exit_code, _output, client = self._run_cli(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertIn("vehicle_track", client.tracker["executed_tables"])

    def test_analytics_schema_is_selected(self):
        captured = {}

        def _factory(*args, **kwargs):
            captured["schema_name"] = kwargs.get("schema_name")
            return _FakeAnalyticsClient(dataset=_base_dataset(), schema_name=kwargs.get("schema_name", "analytics"))

        output = io.StringIO()
        with patch.object(verify_enrichment_run, "AnalyticsDatabaseClient", side_effect=_factory), contextlib.redirect_stdout(output):
            exit_code = verify_enrichment_run.run(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertEqual(captured["schema_name"], "analytics")

    def test_valid_camera_inside_run(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_001")
        self.assertEqual(report["scope"], "camera")
        self.assertEqual(report["camera"]["camera_code"], "CAM_001")

    def test_camera_not_found(self):
        with self.assertRaises(verify_enrichment_run.VerificationQueryError):
            verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_X")

    def test_camera_belongs_to_different_run(self):
        dataset = _base_dataset()
        dataset["camera_run"] = [dataset["camera_run"][1]]
        with self.assertRaises(verify_enrichment_run.VerificationQueryError):
            verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, camera_code="CAM_001")

    def test_camera_scoped_counts(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_001")
        self.assertEqual(report["counts"]["vehicle_track"]["count"], 2)
        self.assertEqual(report["counts"]["plate_detection"]["count"], 1)

    def test_camera_scoped_coverage(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_001")
        self.assertEqual(report["coverage"]["verified_plate_coverage"], 50.0)

    def test_camera_samples_contain_only_selected_camera_tracks(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_001")
        self.assertTrue(all(row["camera_code"] == "CAM_001" for row in report["samples"]["tracks"]))

    def test_valid_track(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["scope"], "track")

    def test_track_not_found(self):
        with self.assertRaises(verify_enrichment_run.VerificationQueryError):
            verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="missing")

    def test_track_belongs_to_different_run(self):
        dataset = _base_dataset()
        dataset["vehicle_track"][0]["processing_run_id"] = "run-other"
        with self.assertRaises(verify_enrichment_run.VerificationQueryError):
            verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")

    def test_track_camera_mismatch(self):
        with self.assertRaises(verify_enrichment_run.VerificationQueryError):
            verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, camera_code="CAM_002", track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")

    def test_observations_summarized_correctly(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4", include_observations=True)
        self.assertEqual(report["track"]["observations"]["total_observation_count"], 3)
        self.assertEqual(report["track"]["observations"]["frame_number_range"], [10, 30])
        self.assertEqual(len(report["track"]["observations"]["rows"]), 3)

    def test_media_records_returned(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(len(report["track"]["media"]), 2)

    def test_vehicle_colour_returned(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["track"]["vehicle_colour"][0]["canonical_colour"], "GREY")

    def test_verified_anpr_returned(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["track"]["anpr"]["results"][0]["normalized_ocr"], "DL8CBF6268")
        self.assertEqual(report["diagnostics"]["missing_enrichment"]["ANPR"], "VERIFIED")

    def test_no_plate_track_handled(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_1")
        self.assertEqual(report["diagnostics"]["missing_enrichment"]["ANPR"], "NO_PLATE")

    def test_plate_detected_but_ocr_missing(self):
        dataset = _base_dataset()
        dataset["plate_reading"] = []
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["diagnostics"]["missing_enrichment"]["ANPR"], "INCOMPLETE")

    def test_local_storage_uri_safely_checked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "debug_runs" / "multicamera_vehicle_tracking_pipeline" / "RUN_1" / "CAM_001" / "track_4" / "best_vehicle.jpg"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("x", encoding="utf-8")
            report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4", artifact_root=Path(temp_dir))
        self.assertEqual(report["track"]["media"][0]["local_reference"]["status"], "LOCAL_FILE_PRESENT")

    def test_absolute_uri_not_treated_as_safe_local_path(self):
        dataset = _base_dataset()
        dataset["track_media"][0]["storage_uri"] = "F:/unsafe/path.jpg"
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["track"]["media"][0]["local_reference"]["status"], "NOT_APPLICABLE")

    def test_same_verified_plate_on_another_camera_is_listed(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["cross_camera_candidates"][0]["track_uuid"], "RUN_20260724_151402:CAM_002:TRACK_4")

    def test_unverified_plate_is_not_used(self):
        dataset = _base_dataset()
        dataset["plate_reading"][0]["status"] = "PROBABLE"
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["cross_camera_candidates"], [])

    def test_same_track_excluded_from_candidates(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertTrue(all(row["track_uuid"] != "RUN_20260724_151402:CAM_001:TRACK_4" for row in report["cross_camera_candidates"]))

    def test_no_candidate_returns_empty_list(self):
        dataset = _base_dataset()
        dataset["plate_reading"][1]["normalized_text"] = "OTHER1234"
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=dataset), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["cross_camera_candidates"], [])

    def test_result_clearly_labelled_as_candidate(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertEqual(report["cross_camera_candidates"][0]["candidate_type"], "PLATE_BASED_CANDIDATE")

    def test_raw_model_paths_excluded(self):
        report = verify_enrichment_run.generate_report(_FakeAnalyticsClient(dataset=_base_dataset()), "RUN_20260724_151402", 5, track_uuid="RUN_20260724_151402:CAM_001:TRACK_4")
        self.assertNotIn("C:/secret/model", json.dumps(report))

    def test_image_bytes_never_requested(self):
        exit_code, _output, client = self._run_cli(["--run-code", "RUN_20260724_151402", "--track-uuid", "RUN_20260724_151402:CAM_001:TRACK_4"])
        self.assertEqual(exit_code, verify_enrichment_run.EXIT_SUCCESS)
        self.assertNotIn("image_bytes", client.tracker["executed_tables"])


if __name__ == "__main__":
    unittest.main()
