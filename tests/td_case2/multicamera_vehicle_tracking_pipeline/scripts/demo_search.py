from __future__ import annotations

from datetime import datetime

from ..database.models import CameraRecord, VehicleAttributeRecord, VehicleSearchFilters, VehicleTrackRecord
from ..database.repository import SimpleVehicleRepository


def main() -> None:
    repo = SimpleVehicleRepository()
    camera = repo.create_camera(CameraRecord(camera_code="CAM-001", camera_name="North Gate"))
    track = repo.create_vehicle_track(
        VehicleTrackRecord(
            camera_id=camera.id,
            local_track_id=5,
            vehicle_class="car",
            track_uuid="CAM-001-5-20260722T094000Z",
            first_seen_at=datetime(2026, 7, 22, 9, 40, 0),
            last_seen_at=datetime(2026, 7, 22, 9, 40, 7),
            best_frame_path="cam001/frame_500.jpg",
            best_crop_path="cam001/crop_500.jpg",
        )
    )
    repo.upsert_vehicle_attributes(
        VehicleAttributeRecord(
            vehicle_track_id=track.id,
            vehicle_colour="white",
            colour_confidence=0.79,
            plate_text="DL01AB1234",
            plate_pattern="DL01AB12?4",
            plate_confidence=0.89,
            plate_verified=True,
            plate_readings=[{"text": "DL01AB1234", "confidence": 0.89}],
        )
    )

    exact = repo.search_vehicles(VehicleSearchFilters(exact_plate="DL01AB1234"))
    partial = repo.search_vehicles(
        VehicleSearchFilters(
            camera_id=camera.id,
            vehicle_colour="white",
            partial_plate="AB123",
            start_time=datetime(2026, 7, 22, 9, 0, 0),
            end_time=datetime(2026, 7, 22, 10, 0, 0),
        )
    )
    print({"exact_plate_matches": len(exact), "partial_plate_matches": len(partial)})


if __name__ == "__main__":
    main()
