from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_models import PersistenceModelValidationError, TrackMediaRecord


class TrackMediaRecordTests(unittest.TestCase):
    def test_valid_relative_path_accepted_and_backslashes_normalized(self) -> None:
        record = TrackMediaRecord(
            vehicle_track_id="track-id",
            media_type="BEST_VEHICLE_CROP",
            storage_uri=r"RUN_1\CAM_001\track_000001\best_overall.jpg",
        )
        self.assertEqual(record.storage_uri, "RUN_1/CAM_001/track_000001/best_overall.jpg")

    def test_windows_absolute_path_rejected(self) -> None:
        with self.assertRaises(PersistenceModelValidationError):
            TrackMediaRecord(vehicle_track_id="track-id", media_type="BEST_VEHICLE_CROP", storage_uri=r"F:\data\best.jpg")

    def test_unix_absolute_path_rejected(self) -> None:
        with self.assertRaises(PersistenceModelValidationError):
            TrackMediaRecord(vehicle_track_id="track-id", media_type="BEST_VEHICLE_CROP", storage_uri="/tmp/best.jpg")

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(PersistenceModelValidationError):
            TrackMediaRecord(vehicle_track_id="track-id", media_type="BEST_VEHICLE_CROP", storage_uri="../best.jpg")

    def test_invalid_dimensions_rejected(self) -> None:
        with self.assertRaises(PersistenceModelValidationError):
            TrackMediaRecord(vehicle_track_id="track-id", media_type="BEST_VEHICLE_CROP", storage_uri="a/b.jpg", width=0)

    def test_invalid_frame_number_rejected(self) -> None:
        with self.assertRaises(PersistenceModelValidationError):
            TrackMediaRecord(vehicle_track_id="track-id", media_type="BEST_VEHICLE_CROP", storage_uri="a/b.jpg", frame_number=-1)

    def test_metadata_copied_safely_and_payload_matches_schema(self) -> None:
        metadata = {"candidate_type": "best_overall"}
        record = TrackMediaRecord(
            vehicle_track_id="track-id",
            media_type="BEST_VEHICLE_CROP",
            storage_uri="RUN_1/CAM_001/track_000001/best_overall.jpg",
            metadata=metadata,
        )
        metadata["candidate_type"] = "mutated"
        payload = record.to_payload()
        self.assertEqual(payload["vehicle_track_id"], "track-id")
        self.assertEqual(payload["media_type"], "BEST_VEHICLE_CROP")
        self.assertEqual(payload["storage_provider"], "LOCAL")
        self.assertEqual(payload["storage_uri"], "RUN_1/CAM_001/track_000001/best_overall.jpg")
        self.assertEqual(payload["metadata"]["candidate_type"], "best_overall")


if __name__ == "__main__":
    unittest.main()
