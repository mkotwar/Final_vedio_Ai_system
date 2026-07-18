from __future__ import annotations

import unittest
from enum import Enum

from tests.td_case2.streaming_tracking_pipeline.validation import (
    normalize_optional_path,
    validate_allowed_value,
    validate_enum_value,
    validate_finite_float,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_positive_float,
    validate_positive_int,
    validate_probability,
)


class ExampleEnum(str, Enum):
    YES = "yes"


class ValidationTests(unittest.TestCase):
    def test_non_empty_string(self) -> None:
        self.assertEqual(validate_non_empty_string(" cam ", "field"), "cam")
        with self.assertRaises(ValueError):
            validate_non_empty_string(" ", "field")

    def test_finite_float(self) -> None:
        self.assertEqual(validate_finite_float("1.5", "field"), 1.5)
        with self.assertRaises(ValueError):
            validate_finite_float(float("inf"), "field")

    def test_probability(self) -> None:
        self.assertEqual(validate_probability(1.0, "field"), 1.0)
        with self.assertRaises(ValueError):
            validate_probability(1.1, "field")

    def test_integer_validators(self) -> None:
        self.assertEqual(validate_non_negative_int(0, "field"), 0)
        self.assertEqual(validate_positive_int(1, "field"), 1)
        with self.assertRaises(ValueError):
            validate_non_negative_int(-1, "field")
        with self.assertRaises(ValueError):
            validate_positive_int(0, "field")

    def test_positive_float(self) -> None:
        self.assertEqual(validate_positive_float(0.1, "field"), 0.1)
        with self.assertRaises(ValueError):
            validate_positive_float(0.0, "field")

    def test_optional_path_normalization(self) -> None:
        self.assertIsNone(normalize_optional_path(""))
        self.assertTrue(normalize_optional_path("folder/file.mp4").endswith("folder\\file.mp4") or normalize_optional_path("folder/file.mp4").endswith("folder/file.mp4"))

    def test_allowed_value(self) -> None:
        self.assertEqual(validate_allowed_value("YES", {"yes"}, "field"), "yes")
        with self.assertRaises(ValueError):
            validate_allowed_value("no", {"yes"}, "field")

    def test_enum_value(self) -> None:
        self.assertEqual(validate_enum_value("yes", ExampleEnum, "field"), ExampleEnum.YES)
        with self.assertRaises(ValueError):
            validate_enum_value("no", ExampleEnum, "field")


if __name__ == "__main__":
    unittest.main()

