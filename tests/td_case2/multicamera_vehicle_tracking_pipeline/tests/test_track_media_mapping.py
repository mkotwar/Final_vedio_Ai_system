from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.track_media_mapping import normalize_track_media_role, normalize_track_media_type


class TrackMediaMappingTests(unittest.TestCase):
    def test_every_supported_role_maps(self) -> None:
        self.assertEqual(normalize_track_media_type("best_overall"), "BEST_VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("first"), "VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("middle"), "VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("last"), "VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("highest_confidence"), "VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("largest"), "VEHICLE_CROP")
        self.assertEqual(normalize_track_media_type("sharpest"), "VEHICLE_CROP")

    def test_case_insensitive_input(self) -> None:
        self.assertEqual(normalize_track_media_role("Best_Overall"), "BEST_OVERALL")

    def test_unsupported_role_fails(self) -> None:
        with self.assertRaises(ValueError):
            normalize_track_media_type("thumbnailish")


if __name__ == "__main__":
    unittest.main()
