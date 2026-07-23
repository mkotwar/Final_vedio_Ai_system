from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_text_normalizer import normalize_registration_text


class PlateTextNormalizerTests(unittest.TestCase):
    def test_normalizes_spaces_and_prefixes(self) -> None:
        result = normalize_registration_text("Plate: mh 12 ab 1234", country_profile="INDIA")
        self.assertEqual(result.cleaned_text, "MH12AB1234")
        self.assertIn("MH12AB1234", result.candidate_values)

    def test_generates_position_aware_ambiguity_candidates(self) -> None:
        result = normalize_registration_text("M012AB1234", country_profile="INDIA")
        self.assertTrue(any(candidate.startswith("MO") for candidate in result.candidate_values))


if __name__ == "__main__":
    unittest.main()
