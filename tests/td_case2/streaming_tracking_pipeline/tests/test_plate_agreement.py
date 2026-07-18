from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.plate_agreement import build_plate_agreement, one_character_disagreement
from tests.td_case2.streaming_tracking_pipeline.plate_validation_schemas import PlateTextCandidate


def candidate(text: str, generation: int = 0) -> PlateTextCandidate:
    return PlateTextCandidate(
        source_id="s",
        track_id=1,
        track_generation=generation,
        raw_ocr_text=text,
        normalized_text=text,
        extracted_text=text,
        corrected_text=None,
        substitutions=[],
        format_status="strict_format_match",
        format_score=1.0,
        ocr_confidence=None,
        plate_detection_confidence=0.8,
        crop_role="primary",
        crop_rank=1,
        frame_index=1,
        timestamp_sec=0.1,
        plate_crop_path=None,
        source_vehicle_crop_path=None,
    )


class PlateAgreementTests(unittest.TestCase):
    def test_exact_agreement(self) -> None:
        result = build_plate_agreement("s", 1, 0, [candidate("UP81CH4158"), candidate("UP81CH4158")])
        self.assertEqual(result.best_support_count, 2)
        self.assertIn("exact_agreement", result.disagreement_reasons)

    def test_one_character_disagreement(self) -> None:
        self.assertTrue(one_character_disagreement("UP81CH4158", "UP81CW4158"))
        result = build_plate_agreement("s", 1, 0, [candidate("UP81CH4158"), candidate("UP81CW4158")])
        self.assertIn("one_character_disagreement", result.disagreement_reasons)

    def test_generation_separation(self) -> None:
        gen0 = build_plate_agreement("s", 1, 0, [candidate("UP81CH4158", 0)])
        gen1 = build_plate_agreement("s", 1, 1, [candidate("UP14CW4087", 1)])
        self.assertNotEqual(gen0.track_generation, gen1.track_generation)
        self.assertNotEqual(gen0.best_candidate, gen1.best_candidate)


if __name__ == "__main__":
    unittest.main()
