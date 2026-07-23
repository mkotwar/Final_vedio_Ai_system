from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.repository import SimpleVehicleRepository
from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_config import DetectionConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_persistence_service import AnalyticsPersistenceService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_backend_factory import build_persistence_service
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.tracking_persistence_service import TrackingPersistenceService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig


class PersistenceBackendFactoryTests(unittest.TestCase):
    def test_dry_run_uses_analytics_service_without_client_initialization(self) -> None:
        with patch(
            "tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_backend_factory.AnalyticsDatabaseClient",
            side_effect=AssertionError("AnalyticsDatabaseClient should not be constructed for dry_run"),
        ):
            service = build_persistence_service(
                config=PersistenceConfig(backend="dry_run", persist_track_media=True, track_media_roles=("BEST_OVERALL",)),
                run_code="RUN_TEST",
                detection_config=DetectionConfig(model_path="yolov8n.pt"),
                tracking_config=TrackingConfig(min_confirmed_observations=1),
                execution_mode="THREADED",
                runtime_device="cpu",
                artifact_root=Path("artifacts"),
            )
        self.assertIsInstance(service, AnalyticsPersistenceService)
        self.assertFalse(service.enable_database_writes)
        self.assertIsNone(service.client)

    def test_old_public_uses_legacy_tracking_service(self) -> None:
        repo = SimpleVehicleRepository()
        service = build_persistence_service(
            config=PersistenceConfig(backend="old_public"),
            run_code="RUN_TEST",
            detection_config=DetectionConfig(model_path="yolov8n.pt"),
            tracking_config=TrackingConfig(min_confirmed_observations=1),
            execution_mode="THREADED",
            runtime_device="cpu",
            artifact_root=Path("artifacts"),
            repository=repo,
        )
        self.assertIsInstance(service, TrackingPersistenceService)
        self.assertIs(service.repository, repo)

    def test_analytics_supabase_uses_analytics_service(self) -> None:
        injected_client = object()
        service = build_persistence_service(
            config=PersistenceConfig(backend="analytics_supabase"),
            run_code="RUN_TEST",
            detection_config=DetectionConfig(model_path="yolov8n.pt"),
            tracking_config=TrackingConfig(min_confirmed_observations=1),
            execution_mode="THREADED",
            runtime_device="cpu",
            artifact_root=Path("artifacts"),
            analytics_client=injected_client,  # type: ignore[arg-type]
        )
        self.assertIsInstance(service, AnalyticsPersistenceService)
        self.assertTrue(service.enable_database_writes)


if __name__ == "__main__":
    unittest.main()
