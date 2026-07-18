from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.plate_validation import (
    best_validated_variant,
    candidates_from_ocr_record,
    generate_controlled_variants,
    validate_plate_format,
)
from tests.td_case2.streaming_tracking_pipeline.plate_validation_schemas import PlateValidationConfig


class PlateValidationTests(unittest.TestCase):
    def test_valid_strict_format(self) -> None:
        result = validate_plate_format("DL10CL3277")
        self.assertEqual(result.format_status, "strict_format_match")
        self.assertEqual(result.format_score, 1.0)

    def test_relaxed_format(self) -> None:
        result = validate_plate_format("UP8ICW415O")
        self.assertEqual(result.format_status, "relaxed_format_match")

    def test_invalid_short_text(self) -> None:
        result = validate_plate_format("T")
        self.assertEqual(result.format_status, "not_plate_like")

    def test_numeric_only_text(self) -> None:
        result = validate_plate_format("9714960")
        self.assertEqual(result.format_status, "partial_plate")

    def test_controlled_i_to_1(self) -> None:
        validation, corrected, substitutions = best_validated_variant("UP8ICW4150", PlateValidationConfig())
        self.assertEqual(validation.format_status, "strict_format_match")
        self.assertEqual(corrected, "UP81CW4150")
        self.assertEqual(substitutions[0]["from"], "I")

    def test_controlled_o_to_0(self) -> None:
        validation, corrected, substitutions = best_validated_variant("DLOOCL3277", PlateValidationConfig(maximum_substitutions_per_candidate=2))
        self.assertEqual(validation.format_status, "strict_format_match")
        self.assertEqual(corrected, "DL00CL3277")
        self.assertTrue(substitutions)

    def test_substitution_limits(self) -> None:
        variants = generate_controlled_variants("OOOO", maximum_substitutions_per_candidate=1, maximum_generated_variants=20)
        self.assertTrue(all(len(subs) <= 1 for _, subs in variants))

    def test_variant_limits(self) -> None:
        variants = generate_controlled_variants("OOOOOOOO", maximum_substitutions_per_candidate=2, maximum_generated_variants=5)
        self.assertEqual(len(variants), 5)

    def test_candidate_preserves_raw_text(self) -> None:
        candidates = candidates_from_ocr_record(
            {
                "source_id": "s",
                "track_id": 1,
                "track_generation": 0,
                "raw_text": "UP16BH0400 Sport",
                "status": "success",
            },
            PlateValidationConfig(),
            {"plate_detection_confidence": 0.8},
        )
        self.assertEqual(candidates[0].raw_ocr_text, "UP16BH0400 Sport")
        self.assertEqual(candidates[0].normalized_text, "UP16BH0400")


if __name__ == "__main__":
    unittest.main()
