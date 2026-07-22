from __future__ import annotations

from datetime import datetime

from ..database.models import CameraRecord, VehicleAttributeRecord, VehicleMatchRecord, VehicleSearchFilters, VehicleTrackRecord
from ..database.repository import SimpleVehicleRepository


def main() -> None:
    repo = SimpleVehicleRepository()
    cam_1 = repo.create_camera(CameraRecord(camera_code="CAM-001", camera_name="North Gate", source_path="rtsp://demo/cam001"))
    cam_2 = repo.create_camera(CameraRecord(camera_code="CAM-002", camera_name="East Exit", source_path="rtsp://demo/cam002"))

    track_1 = repo.create_vehicle_track(
        VehicleTrackRecord(
            camera_id=cam_1.id,
            local_track_id=101,
            vehicle_class="car",
            track_uuid="CAM-001-101-20260722T090000Z",
            first_seen_at=datetime(2026, 7, 22, 9, 0, 0),
            last_seen_at=datetime(2026, 7, 22, 9, 0, 8),
            best_frame_path="cam001/frame_100.jpg",
            best_crop_path="cam001/crop_101.jpg",
        )
    )
    track_2 = repo.create_vehicle_track(
        VehicleTrackRecord(
            camera_id=cam_2.id,
            local_track_id=44,
            vehicle_class="car",
            track_uuid="CAM-002-044-20260722T090035Z",
            first_seen_at=datetime(2026, 7, 22, 9, 0, 35),
            last_seen_at=datetime(2026, 7, 22, 9, 0, 41),
            best_frame_path="cam002/frame_044.jpg",
            best_crop_path="cam002/crop_044.jpg",
        )
    )

    repo.upsert_vehicle_attributes(
        VehicleAttributeRecord(
            vehicle_track_id=track_1.id,
            vehicle_colour="white",
            colour_confidence=0.84,
            plate_text="DL01AB1234",
            plate_pattern="DL01AB12?4",
            plate_confidence=0.91,
            plate_verified=True,
            plate_readings=[{"text": "DL01AB12?4", "confidence": 0.72}, {"text": "DL01AB1234", "confidence": 0.91}],
        )
    )
    repo.upsert_vehicle_attributes(
        VehicleAttributeRecord(
            vehicle_track_id=track_2.id,
            vehicle_colour="white",
            colour_confidence=0.80,
            plate_text="DL01AB12?4",
            plate_pattern="DL01AB12?4",
            plate_confidence=0.75,
            plate_readings=[{"text": "DL01AB12?4", "confidence": 0.75}],
        )
    )
    repo.create_vehicle_match(
        VehicleMatchRecord(
            source_track_id=track_1.id,
            candidate_track_id=track_2.id,
            plate_similarity=0.90,
            colour_match=True,
            class_match=True,
            time_gap_seconds=27.0,
            match_score=0.88,
            match_status="probable",
        )
    )

    results = repo.search_vehicles(VehicleSearchFilters(vehicle_colour="white", vehicle_class="car"))
    print({"cameras": len(repo.list_cameras()), "tracks": len(results), "matches": len(repo.get_vehicle_matches(track_1.id))})


if __name__ == "__main__":
    main()

