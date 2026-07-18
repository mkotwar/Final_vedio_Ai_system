from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.class_normalization import normalize_class_name


class ClassNormalizationTests(unittest.TestCase):
    def test_two_wheelers_are_not_cars(self) -> None:
        cases = {
            "motorcycle": "motorcycle",
            "motorbike": "motorcycle",
            "scooter": "motorcycle",
            "two_wheeler": "motorcycle",
            "two wheeler": "motorcycle",
            "car": "car",
            "bus": "bus",
            "truck": "truck",
            "bicycle": "bicycle",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_class_name(raw).normalized_class_name, expected)

    def test_unknown_vehicle_does_not_default_to_car(self) -> None:
        self.assertEqual(normalize_class_name("3Wheeler").normalized_class_name, "3wheeler")
        self.assertEqual(normalize_class_name("traffic cone").normalized_class_name, "other_object")


if __name__ == "__main__":
    unittest.main()
