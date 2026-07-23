from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_models import VehicleColourResult, VehicleColourValidationError


class VehicleColourModelsTests(unittest.TestCase):
    def test_relative_source_uri_is_normalized(self) -> None:
        result = VehicleColourResult(
            canonical_colour="WHITE",
            raw_output="WHITE",
            confidence=0.8,
            status="SUCCESS",
            source_storage_uri=r"RUN_1\CAM_001\track_000001\best_overall.jpg",
        )
        self.assertEqual(result.source_storage_uri, "RUN_1/CAM_001/track_000001/best_overall.jpg")

    def test_absolute_source_uri_is_rejected(self) -> None:
        with self.assertRaises(VehicleColourValidationError):
            VehicleColourResult(
                canonical_colour="WHITE",
                raw_output="WHITE",
                confidence=0.8,
                status="SUCCESS",
                source_storage_uri=r"F:\bad\path.jpg",
            )


if __name__ == "__main__":
    unittest.main()
