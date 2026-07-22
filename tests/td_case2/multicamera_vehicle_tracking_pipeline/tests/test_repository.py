from __future__ import annotations

import unittest
from datetime import datetime

from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.models import (
    CameraRecord,
    VehicleAttributeRecord,
    VehicleMatchRecord,
    VehicleObservationRecord,
    VehicleSearchFilters,
    VehicleTrackRecord,
)
from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.repository import RepositoryConstraintError, SimpleVehicleRepository


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = SimpleVehicleRepository()
        self.cam_1 = self.repo.create_camera(CameraRecord(camera_code="CAM-001", camera_name="North", source_path="cam1.mp4"))
        self.cam_2 = self.repo.create_camera(CameraRecord(camera_code="CAM-002", camera_name="South", source_path="cam2.mp4"))

    def _track(self, camera_id, local_track_id: int, at: datetime, *, vehicle_class: str = "car", track_uuid: str | None = None) -> VehicleTrackRecord:
        return VehicleTrackRecord(
            camera_id=camera_id,
            local_track_id=local_track_id,
            vehicle_class=vehicle_class,
            track_uuid=track_uuid or f"{camera_id}-{local_track_id}-{at.isoformat()}",
            first_seen_at=at,
            last_seen_at=at,
        )

    def test_camera_insertion(self) -> None:
        self.assertEqual([camera.camera_code for camera in self.repo.list_cameras()], ["CAM-001", "CAM-002"])

    def test_vehicle_track_insertion(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 1, datetime(2026, 7, 22, 9, 0, 0)))
        self.assertEqual(track.local_track_id, 1)

    def test_same_local_track_id_on_different_cameras(self) -> None:
        self.repo.create_vehicle_track(self._track(self.cam_1.id, 8, datetime(2026, 7, 22, 9, 0, 0)))
        self.repo.create_vehicle_track(self._track(self.cam_2.id, 8, datetime(2026, 7, 22, 9, 0, 0)))

    def test_vehicle_attribute_upsert(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 2, datetime(2026, 7, 22, 9, 1, 0)))
        first = self.repo.upsert_vehicle_attributes(VehicleAttributeRecord(vehicle_track_id=track.id, vehicle_colour="white"))
        second = self.repo.upsert_vehicle_attributes(VehicleAttributeRecord(vehicle_track_id=track.id, vehicle_colour="red"))
        self.assertEqual(first.vehicle_track_id, second.vehicle_track_id)
        self.assertEqual(self.repo.search_vehicles(VehicleSearchFilters(vehicle_colour="red"))[0]["vehicle_colour"], "red")

    def test_multiple_ocr_readings_stored_in_json(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 3, datetime(2026, 7, 22, 9, 2, 0)))
        attributes = self.repo.upsert_vehicle_attributes(
            VehicleAttributeRecord(
                vehicle_track_id=track.id,
                plate_text="DL01AB1234",
                plate_pattern="DL01AB12?4",
                plate_readings=[{"text": "DL01AB12?4", "confidence": 0.72}, {"text": "DL01AB1234", "confidence": 0.88}],
            )
        )
        self.assertEqual(len(attributes.plate_readings), 2)

    def test_observation_bulk_insert(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 4, datetime(2026, 7, 22, 9, 3, 0)))
        rows = self.repo.add_vehicle_observations(
            [
                VehicleObservationRecord(vehicle_track_id=track.id, frame_number=10, observed_at=datetime(2026, 7, 22, 9, 3, 0), bbox_x1=1, bbox_y1=2, bbox_x2=3, bbox_y2=4),
                VehicleObservationRecord(vehicle_track_id=track.id, frame_number=11, observed_at=datetime(2026, 7, 22, 9, 3, 1), bbox_x1=2, bbox_y1=3, bbox_x2=4, bbox_y2=5),
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(self.repo.get_track_observations(track.id)), 2)

    def test_exact_plate_search(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 5, datetime(2026, 7, 22, 9, 4, 0)))
        self.repo.upsert_vehicle_attributes(VehicleAttributeRecord(vehicle_track_id=track.id, plate_text="DL01AB1234"))
        results = self.repo.search_vehicles(VehicleSearchFilters(exact_plate="DL01AB1234"))
        self.assertEqual(len(results), 1)

    def test_partial_plate_search(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 6, datetime(2026, 7, 22, 9, 5, 0)))
        self.repo.upsert_vehicle_attributes(VehicleAttributeRecord(vehicle_track_id=track.id, plate_text="DL01AB1234"))
        results = self.repo.search_vehicles(VehicleSearchFilters(partial_plate="AB123"))
        self.assertEqual(len(results), 1)

    def test_colour_and_class_search(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 7, datetime(2026, 7, 22, 9, 6, 0), vehicle_class="truck"))
        self.repo.upsert_vehicle_attributes(VehicleAttributeRecord(vehicle_track_id=track.id, vehicle_colour="white"))
        results = self.repo.search_vehicles(VehicleSearchFilters(vehicle_class="truck", vehicle_colour="white"))
        self.assertEqual(len(results), 1)

    def test_valid_cross_camera_match(self) -> None:
        source = self.repo.create_vehicle_track(self._track(self.cam_1.id, 8, datetime(2026, 7, 22, 9, 7, 0), track_uuid="cam1-track-8"))
        candidate = self.repo.create_vehicle_track(self._track(self.cam_2.id, 8, datetime(2026, 7, 22, 9, 7, 30), track_uuid="cam2-track-8"))
        match = self.repo.create_vehicle_match(
            VehicleMatchRecord(
                source_track_id=source.id,
                candidate_track_id=candidate.id,
                plate_similarity=0.82,
                colour_match=True,
                class_match=True,
                time_gap_seconds=30.0,
                match_score=0.85,
                match_status="probable",
            )
        )
        self.assertEqual(match.match_status, "probable")

    def test_self_match_rejection(self) -> None:
        track = self.repo.create_vehicle_track(self._track(self.cam_1.id, 9, datetime(2026, 7, 22, 9, 8, 0)))
        with self.assertRaises(RepositoryConstraintError):
            self.repo.create_vehicle_match(VehicleMatchRecord(source_track_id=track.id, candidate_track_id=track.id, match_status="rejected"))

    def test_duplicate_match_rejection(self) -> None:
        source = self.repo.create_vehicle_track(self._track(self.cam_1.id, 10, datetime(2026, 7, 22, 9, 9, 0), track_uuid="cam1-track-10"))
        candidate = self.repo.create_vehicle_track(self._track(self.cam_2.id, 10, datetime(2026, 7, 22, 9, 9, 30), track_uuid="cam2-track-10"))
        self.repo.create_vehicle_match(VehicleMatchRecord(source_track_id=source.id, candidate_track_id=candidate.id, match_status="probable"))
        with self.assertRaises(RepositoryConstraintError):
            self.repo.create_vehicle_match(VehicleMatchRecord(source_track_id=source.id, candidate_track_id=candidate.id, match_status="probable"))


if __name__ == "__main__":
    unittest.main()
