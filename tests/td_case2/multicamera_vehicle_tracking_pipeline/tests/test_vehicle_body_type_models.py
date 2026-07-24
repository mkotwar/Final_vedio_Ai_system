from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_body_type_models import VehicleBodyTypeResult, VehicleBodyTypeValidationError


class VehicleBodyTypeModelsTests(unittest.TestCase):
    def test_relative_source_uri_is_normalized(self) -> None:
        result = VehicleBodyTypeResult(
            canonical_body_type="SUV",
            raw_output="SUV",
            confidence=0.8,
            status="SUCCESS",
            source_storage_uri=r"vehicle_1\image.jpg",
        )
        self.assertEqual(result.source_storage_uri, "vehicle_1/image.jpg")

    def test_absolute_source_uri_is_rejected(self) -> None:
        with self.assertRaises(VehicleBodyTypeValidationError):
            VehicleBodyTypeResult(
                canonical_body_type="SUV",
                raw_output="SUV",
                confidence=0.8,
                status="SUCCESS",
                source_storage_uri=r"F:\bad\path.jpg",
            )


if __name__ == "__main__":
    unittest.main()
