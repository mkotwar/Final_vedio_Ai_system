from __future__ import annotations

from datetime import datetime

from ..database.models import CameraRecord, VehicleAttributeRecord, VehicleObservationRecord, VehicleTrackRecord
from ..database.repository import SimpleVehicleRepository


def main() -> None:
    repo = SimpleVehicleRepository()
    camera = repo.create_camera(CameraRecord(camera_code="CAM-001", camera_name="North Gate", source_path="demo_video_001.mp4"))
    track = repo.create_vehicle_track(
        VehicleTrackRecord(
            camera_id=camera.id,
            local_track_id=7,
            vehicle_class="truck",
            track_uuid="CAM-001-7-20260722T091500Z",
            first_seen_at=datetime(2026, 7, 22, 9, 15, 0),
            last_seen_at=datetime(2026, 7, 22, 9, 15, 6),
            observation_count=2,
            best_confidence=0.87,
            best_frame_path="cam001/frame_220.jpg",
            best_crop_path="cam001/crop_220.jpg",
        )
    )
    repo.add_vehicle_observations(
        [
            VehicleObservationRecord(vehicle_track_id=track.id, frame_number=220, observed_at=datetime(2026, 7, 22, 9, 15, 0), bbox_x1=10, bbox_y1=20, bbox_x2=120, bbox_y2=200, confidence=0.81),
            VehicleObservationRecord(vehicle_track_id=track.id, frame_number=225, observed_at=datetime(2026, 7, 22, 9, 15, 1), bbox_x1=12, bbox_y1=24, bbox_x2=122, bbox_y2=205, confidence=0.87),
        ]
    )
    attributes = repo.upsert_vehicle_attributes(
        VehicleAttributeRecord(
            vehicle_track_id=track.id,
            vehicle_colour="blue",
            colour_confidence=0.69,
            plate_text="MH12TR7788",
            plate_pattern="MH12TR7788",
            plate_confidence=0.85,
            plate_verified=True,
            plate_readings=[{"text": "MH12TR7788", "confidence": 0.85}],
        )
    )
    print({"track_uuid": track.track_uuid, "plate": attributes.plate_text, "observation_count": len(repo.get_track_observations(track.id))})


if __name__ == "__main__":
    main()

