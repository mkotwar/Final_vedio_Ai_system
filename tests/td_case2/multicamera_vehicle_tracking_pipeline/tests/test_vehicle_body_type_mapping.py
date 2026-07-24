from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_body_type_mapping import normalize_vehicle_body_type


class VehicleBodyTypeMappingTests(unittest.TestCase):
    def test_known_aliases_normalize(self) -> None:
        self.assertEqual(normalize_vehicle_body_type("sedan"), "SEDAN")
        self.assertEqual(normalize_vehicle_body_type("saloon"), "SEDAN")
        self.assertEqual(normalize_vehicle_body_type("sport utility vehicle"), "SUV")
        self.assertEqual(normalize_vehicle_body_type("pickup truck"), "PICKUP")
        self.assertEqual(normalize_vehicle_body_type("auto rickshaw"), "THREE_WHEELER")
        self.assertEqual(normalize_vehicle_body_type("motorbike"), "MOTORCYCLE")

    def test_case_and_separator_handling(self) -> None:
        self.assertEqual(normalize_vehicle_body_type("three-wheeler"), "THREE_WHEELER")
        self.assertEqual(normalize_vehicle_body_type("MINI_VAN"), "MINIVAN")

    def test_unknown_falls_back(self) -> None:
        self.assertEqual(normalize_vehicle_body_type("spaceship"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
