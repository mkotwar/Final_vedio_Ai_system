from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.vehicle_class_mapping import (
    VehicleClass,
    normalize_runtime_vehicle_class,
    normalize_vehicle_class,
)


class VehicleClassMappingTests(unittest.TestCase):
    def test_runtime_normalization_supports_all_detector_classes(self) -> None:
        self.assertEqual(normalize_runtime_vehicle_class("3Wheeler"), "3wheeler")
        self.assertEqual(normalize_runtime_vehicle_class("bus"), "bus")
        self.assertEqual(normalize_runtime_vehicle_class("car"), "car")
        self.assertEqual(normalize_runtime_vehicle_class("motorcycle"), "motorcycle")
        self.assertEqual(normalize_runtime_vehicle_class("truck"), "truck")

    def test_canonical_normalization_supports_all_detector_classes(self) -> None:
        self.assertEqual(normalize_vehicle_class("3Wheeler"), VehicleClass.THREE_WHEELER)
        self.assertEqual(normalize_vehicle_class("3wheeler"), VehicleClass.THREE_WHEELER)
        self.assertEqual(normalize_vehicle_class("bus"), VehicleClass.BUS)
        self.assertEqual(normalize_vehicle_class("car"), VehicleClass.CAR)
        self.assertEqual(normalize_vehicle_class("motorcycle"), VehicleClass.MOTORCYCLE)
        self.assertEqual(normalize_vehicle_class("truck"), VehicleClass.TRUCK)

    def test_unknown_input_maps_to_unknown(self) -> None:
        self.assertEqual(normalize_vehicle_class("plane"), VehicleClass.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
