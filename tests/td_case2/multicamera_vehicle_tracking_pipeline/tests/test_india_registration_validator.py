from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.india_registration_validator import validate_indian_registration
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_models import NormalizedRegistrationText


class IndiaRegistrationValidatorTests(unittest.TestCase):
    def test_verifies_standard_plate(self) -> None:
        normalized = NormalizedRegistrationText(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            candidate_values=("MH12AB1234",),
        )
        result = validate_indian_registration(normalized, ocr_confidence=0.9, minimum_length=6, maximum_length=12)
        self.assertTrue(result.is_verified)
        self.assertEqual(result.normalized_text, "MH12AB1234")
        self.assertEqual(result.status, "VERIFIED")

    def test_marks_low_confidence_as_probable(self) -> None:
        normalized = NormalizedRegistrationText(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            candidate_values=("MH12AB1234",),
        )
        result = validate_indian_registration(normalized, ocr_confidence=0.2, minimum_length=6, maximum_length=12)
        self.assertFalse(result.is_verified)
        self.assertEqual(result.status, "PROBABLE")


if __name__ == "__main__":
    unittest.main()
