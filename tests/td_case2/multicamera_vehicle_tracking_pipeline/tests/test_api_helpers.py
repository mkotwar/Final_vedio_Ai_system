from __future__ import annotations

from copy import deepcopy

from fastapi.testclient import TestClient

from ..api.main import create_app
from ..api.settings import ApiSettings
from ..persistence.analytics_repository_base import AnalyticsRepositoryError
from ..persistence.api_read_repository import Page


class FakeApiRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.raise_health_error = False
        self.raise_run_list_error = False
        self.run_detail = {
            "id": "run-1",
            "run_code": "RUN_20260724_151402",
            "status": "COMPLETED",
            "started_at": "2026-07-24T15:14:02+05:30",
            "completed_at": "2026-07-24T15:19:02+05:30",
            "created_at": "2026-07-24T15:14:01+05:30",
            "pipeline_name": "multicamera_vehicle_tracking",
            "camera_summary": {"configured_camera_count": 2, "active_camera_count": 2, "camera_run_count": 2, "completed_camera_runs": 2},
            "track_summary": {"track_count": 8, "total_track_observations": 200},
            "enrichment_summary": {"tracks_with_colour": 8, "tracks_with_plate_summary": 8, "tracks_with_media": 8},
            "global_object_summary": {"global_vehicle_count": 7},
            "processing_error_summary": {"processing_error_count": 0},
            "metadata": {"service_role_key": "secret", "model_path": "C:/hidden/model"},
        }
        self.track_detail = {
            "track": {
                "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4",
                "camera_code": "CAM_001",
                "vehicle_class": "CAR",
                "primary_colour": "GREY",
                "canonical_plate": "DL8CBF6268",
                "plate_status": "VERIFIED",
                "primary_media": {
                    "media_id": "media-1",
                    "media_type": "BEST_VEHICLE_CROP",
                    "storage_provider": "LOCAL",
                    "storage_uri": "safe/ref.jpg",
                },
            },
            "camera": {"camera_code": "CAM_001", "camera_name": "Entry Gate", "location": "North"},
            "colour": {"primary_colour": "GREY", "colour_confidence": 0.95, "model_path": "unsafe"},
            "plate": {"canonical_plate": "DL8CBF6268", "plate_status": "VERIFIED", "plate_confidence": 0.98},
            "media": [{"media_id": "media-1", "media_type": "BEST_VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "safe/ref.jpg"}],
            "observation_summary": {"count": 12, "first_frame": 10, "last_frame": 40, "key_observation_count": 2},
            "global_membership": {"global_vehicle_code": "GVO:RUN_20260724_151402:943BD1FE7C62"},
            "cross_camera_matches": [{"id": "match-1", "decision": "CONFIRMED"}],
            "errors": [],
        }

    def health(self):
        self.calls.append(("health", {}))
        if self.raise_health_error:
            raise AnalyticsRepositoryError(operation="health", table_name="processing_run", message="boom")
        return {"status": "ok", "service": "multicamera-vehicle-api", "database": "reachable", "schema": "analytics"}

    def list_runs(self, **kwargs):
        self.calls.append(("list_runs", kwargs))
        if self.raise_run_list_error:
            raise AnalyticsRepositoryError(operation="list_runs", table_name="processing_run", message="db blew up with token=abc")
        return Page(
            items=[
                {
                    "id": "run-1",
                    "run_code": "RUN_20260724_151402",
                    "status": kwargs.get("status") or "COMPLETED",
                    "started_at": "2026-07-24T15:14:02+05:30",
                    "completed_at": "2026-07-24T15:19:02+05:30",
                    "created_at": "2026-07-24T15:14:01+05:30",
                    "camera_count": 2,
                    "track_count": 8,
                    "global_vehicle_count": 7,
                    "processing_error_count": 0,
                    "metadata": {"api_key": "hide-me"},
                }
            ],
            page=kwargs["page"],
            page_size=kwargs["page_size"],
            total=1,
        )

    def get_run_detail(self, run_code: str):
        self.calls.append(("get_run_detail", {"run_code": run_code}))
        if run_code != "RUN_20260724_151402":
            return None
        return deepcopy(self.run_detail)

    def list_run_cameras(self, run_code: str, **kwargs):
        self.calls.append(("list_run_cameras", {"run_code": run_code, **kwargs}))
        if run_code != "RUN_20260724_151402":
            return None, Page(items=[], page=kwargs["page"], page_size=kwargs["page_size"], total=0)
        items = [
            {
                "id": "cam-1",
                "camera_code": "CAM_001",
                "camera_name": "Entry Gate",
                "location": "North",
                "camera_run_status": kwargs.get("status") or "COMPLETED",
                "frames_read": 100,
                "frames_processed": 100,
                "detection_count": 12,
                "completed_track_count": 4,
                "discarded_track_count": 1,
            }
        ]
        if kwargs.get("camera_code") and kwargs["camera_code"] != "CAM_001":
            items = []
        return {"id": "run-1"}, Page(items=items, page=kwargs["page"], page_size=kwargs["page_size"], total=len(items))

    def get_camera_in_run(self, run_code: str, camera_code: str):
        self.calls.append(("get_camera_in_run", {"run_code": run_code, "camera_code": camera_code}))
        if run_code != "RUN_20260724_151402":
            return None, None
        if camera_code != "CAM_001":
            return {"id": "run-1"}, None
        return {"id": "run-1"}, {
            "id": "cam-1",
            "camera_code": "CAM_001",
            "camera_name": "Entry Gate",
            "location": "North",
            "camera_run_status": "COMPLETED",
            "frames_read": 100,
            "frames_processed": 100,
            "detection_count": 12,
            "completed_track_count": 4,
            "discarded_track_count": 1,
            "track_count": 4,
            "colour_coverage": 4,
            "plate_coverage": 2,
            "media_coverage": 4,
            "processing_errors": [],
        }

    def list_tracks(self, run_code: str, **kwargs):
        self.calls.append(("list_tracks", {"run_code": run_code, **kwargs}))
        if run_code != "RUN_20260724_151402":
            return None, Page(items=[], page=kwargs["page"], page_size=kwargs["page_size"], total=0)
        return {"id": "run-1"}, Page(
            items=[
                {
                    "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4",
                    "camera_code": kwargs.get("camera_code") or "CAM_001",
                    "local_track_id": 4,
                    "vehicle_class": kwargs.get("vehicle_class") or "CAR",
                    "lifecycle_state": kwargs.get("lifecycle_state") or "COMPLETED",
                    "first_seen_at": "2026-07-24T15:14:30+05:30",
                    "last_seen_at": "2026-07-24T15:14:45+05:30",
                    "first_video_time_seconds": 12.0,
                    "last_video_time_seconds": 27.0,
                    "observation_count": 10,
                    "best_detection_confidence": kwargs.get("minimum_confidence", 0.91) or 0.91,
                    "average_detection_confidence": 0.88,
                    "primary_colour": kwargs.get("colour") or "GREY",
                    "colour_confidence": 0.95,
                    "canonical_plate": kwargs.get("plate") or "DL8CBF6268",
                    "plate_status": kwargs.get("plate_status") or "VERIFIED",
                    "plate_confidence": 0.98,
                    "primary_media": {"media_id": "media-1", "storage_uri": "safe/ref.jpg"},
                }
            ],
            page=kwargs["page"],
            page_size=kwargs["page_size"],
            total=1,
        )

    def get_track_by_uuid(self, track_uuid: str):
        self.calls.append(("get_track_by_uuid", {"track_uuid": track_uuid}))
        if track_uuid != "RUN_20260724_151402:CAM_001:TRACK_4":
            return None
        return deepcopy(self.track_detail)

    def list_track_observations(self, track_uuid: str, **kwargs):
        self.calls.append(("list_track_observations", {"track_uuid": track_uuid, **kwargs}))
        if track_uuid != "RUN_20260724_151402:CAM_001:TRACK_4":
            return None, Page(items=[], page=kwargs["page"], page_size=kwargs["page_size"], total=0)
        items = [
            {
                "frame_number": 12,
                "timestamp": "2026-07-24T15:14:31+05:30",
                "video_time_seconds": 12.4,
                "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
                "detection_confidence": 0.91,
                "tracker_confidence": 0.89,
                "is_key_observation": kwargs.get("key_only", False),
            }
        ]
        return {"id": "track-1"}, Page(items=items, page=kwargs["page"], page_size=kwargs["page_size"], total=1)

    def get_track_row(self, track_uuid: str):
        if track_uuid == "RUN_20260724_151402:CAM_001:TRACK_4":
            return {"id": "track-1"}
        return None

    def list_track_media(self, track_uuid: str):
        self.calls.append(("list_track_media", {"track_uuid": track_uuid}))
        if track_uuid != "RUN_20260724_151402:CAM_001:TRACK_4":
            return []
        return [
            {
                "media_id": "media-1",
                "media_type": "BEST_VEHICLE_CROP",
                "storage_provider": "LOCAL",
                "storage_uri": "safe/ref.jpg",
                "frame_number": 12,
                "captured_at": "2026-07-24T15:14:31+05:30",
                "video_time_seconds": 12.4,
                "width": 200,
                "height": 100,
                "quality_score": 0.9,
                "sharpness_score": 0.8,
                "visibility_score": 0.85,
                "selection_rank": 1,
                "is_primary": True,
            }
        ]

    def get_media_by_id(self, media_id: str):
        self.calls.append(("get_media_by_id", {"media_id": media_id}))
        if media_id == "media-1":
            return {"id": "media-1", "media_type": "BEST_VEHICLE_CROP", "storage_uri": "safe/ref.jpg"}
        if media_id == "media-unsafe-traversal":
            return {"id": media_id, "media_type": "BEST_VEHICLE_CROP", "storage_uri": "../secret.jpg"}
        if media_id == "media-unsafe-absolute":
            return {"id": media_id, "media_type": "BEST_VEHICLE_CROP", "storage_uri": "C:/secret.jpg"}
        return None

    def list_global_vehicles(self, **kwargs):
        self.calls.append(("list_global_vehicles", kwargs))
        return Page(
            items=[
                {
                    "global_vehicle_code": "GVO:RUN_20260724_151402:943BD1FE7C62",
                    "run_code": kwargs.get("run_code") or "RUN_20260724_151402",
                    "status": kwargs.get("status") or "CONFIRMED",
                    "canonical_plate": kwargs.get("plate") or "DL8CBF6268",
                    "canonical_colour": kwargs.get("colour") or "GREY",
                    "canonical_vehicle_class": kwargs.get("vehicle_class") or "CAR",
                    "confidence": kwargs.get("minimum_confidence", 0.95) or 0.95,
                    "camera_count": kwargs.get("minimum_camera_count", 2) or 2,
                    "track_count": 2,
                    "creation_method": "CROSS_CAMERA_MATCH",
                    "first_seen_at": "2026-07-24T15:14:30+05:30",
                    "last_seen_at": "2026-07-24T15:14:50+05:30",
                    "primary_evidence_reference": {"media_id": "media-1", "storage_uri": "safe/ref.jpg"},
                }
            ],
            page=kwargs["page"],
            page_size=kwargs["page_size"],
            total=1,
        )

    def get_global_vehicle_by_code(self, global_vehicle_code: str):
        self.calls.append(("get_global_vehicle_by_code", {"global_vehicle_code": global_vehicle_code}))
        if global_vehicle_code != "GVO:RUN_20260724_151402:943BD1FE7C62":
            return None
        return {
            "global_vehicle": {
                "global_vehicle_code": global_vehicle_code,
                "run_code": "RUN_20260724_151402",
                "status": "CONFIRMED",
                "canonical_plate": "DL8CBF6268",
                "canonical_colour": "GREY",
                "canonical_vehicle_class": "CAR",
                "confidence": 0.95,
                "camera_count": 2,
                "track_count": 2,
            },
            "members": [
                {"track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4", "camera_code": "CAM_001"},
                {"track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4", "camera_code": "CAM_002"},
            ],
            "camera_sequence": [
                {"camera_code": "CAM_001", "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4"},
                {"camera_code": "CAM_002", "track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4"},
            ],
            "confirmed_matches": [{"id": "match-1", "decision": "CONFIRMED"}],
            "possible_matches": [],
            "evidence": [{"media_id": "media-1", "storage_uri": "safe/ref.jpg"}],
        }

    def list_global_vehicle_members(self, global_vehicle_code: str):
        self.calls.append(("list_global_vehicle_members", {"global_vehicle_code": global_vehicle_code}))
        if global_vehicle_code != "GVO:RUN_20260724_151402:943BD1FE7C62":
            return []
        return [
            {"track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4", "camera_code": "CAM_001"},
            {"track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4", "camera_code": "CAM_002"},
        ]

    def list_cross_camera_matches(self, **kwargs):
        self.calls.append(("list_cross_camera_matches", kwargs))
        return Page(
            items=[
                {
                    "id": "match-1",
                    "source_track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4",
                    "candidate_track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4",
                    "source_camera_code": kwargs.get("camera_code") or "CAM_001",
                    "candidate_camera_code": "CAM_002",
                    "decision": kwargs.get("decision") or "CONFIRMED",
                    "overall_score": kwargs.get("minimum_score", 0.95) or 0.95,
                    "plate_score": 1.0,
                    "route_score": 0.8,
                    "time_score": 0.9,
                    "class_score": 1.0,
                    "colour_score": 0.95,
                    "visual_score": 0.75,
                    "decision_reasons": ["plate match", "time gap ok"],
                    "rule_version": kwargs.get("rule_version") or "v1",
                    "linked_global_vehicle_code": "GVO:RUN_20260724_151402:943BD1FE7C62",
                }
            ],
            page=kwargs["page"],
            page_size=kwargs["page_size"],
            total=1,
        )

    def get_cross_camera_match(self, match_id: str):
        self.calls.append(("get_cross_camera_match", {"match_id": match_id}))
        if match_id != "match-1":
            return None
        return self.list_cross_camera_matches(page=1, page_size=25).items[0]


def build_test_client(repository: FakeApiRepository) -> TestClient:
    settings = ApiSettings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="super-secret",
        API_CORS_ORIGINS="http://localhost:5173",
    )
    app = create_app(settings=settings, repository_factory=lambda _: repository)
    app.state.settings = settings
    app.state.repository = repository
    return TestClient(app, raise_server_exceptions=False)
