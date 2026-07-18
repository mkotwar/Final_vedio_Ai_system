from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.plate_agreement import build_plate_agreement
from tests.td_case2.streaming_tracking_pipeline.plate_validation_metrics import score_candidate
from tests.td_case2.streaming_tracking_pipeline.plate_validation_schemas import PlateTextCandidate, PlateValidationConfig
from tests.td_case2.streaming_tracking_pipeline.run_step8_plate_validation import _finalize_track


def make_candidate(text: str, *, status: str = "strict_format_match", detector: float = 0.8) -> PlateTextCandidate:
    return PlateTextCandidate(
        source_id="s",
        track_id=1,
        track_generation=0,
        raw_ocr_text=text,
        normalized_text=text,
        extracted_text=text,
        corrected_text=None,
        substitutions=[],
        format_status=status,
        format_score=1.0 if status == "strict_format_match" else 0.45 if status == "partial_plate" else 0.05,
        ocr_confidence=None,
        plate_detection_confidence=detector,
        crop_role="primary",
        crop_rank=1,
        frame_index=1,
        timestamp_sec=0.1,
        plate_crop_path="plate.jpg",
        source_vehicle_crop_path="vehicle.jpg",
    )


class PlateValidationMetricsTests(unittest.TestCase):
    def test_score_components_are_deterministic(self) -> None:
        item = make_candidate("UP81CH4158")
        agreement = build_plate_agreement("s", 1, 0, [item])
        left = score_candidate(item, agreement)
        right = score_candidate(item, agreement)
        self.assertEqual(left, right)
        self.assertIn("final_candidate_score", left)

    def test_verified_decision(self) -> None:
        item = make_candidate("UP81CH4158", detector=0.9)
        agreement = build_plate_agreement("s", 1, 0, [item])
        final = _finalize_track(("s", 1, 0), [item], agreement, {}, {"object_class": "car"}, {}, PlateValidationConfig())
        self.assertEqual(final.plate_status, "verified")

    def test_weak_decision(self) -> None:
        item = make_candidate("9714960", status="partial_plate", detector=0.4)
        agreement = build_plate_agreement("s", 1, 0, [item])
        final = _finalize_track(("s", 1, 0), [item], agreement, {}, {"object_class": "car"}, {}, PlateValidationConfig())
        self.assertEqual(final.plate_status, "weak")

    def test_invalid_decision(self) -> None:
        item = make_candidate("T", status="not_plate_like", detector=0.8)
        agreement = build_plate_agreement("s", 1, 0, [item])
        final = _finalize_track(("s", 1, 0), [item], agreement, {}, {"object_class": "car"}, {}, PlateValidationConfig())
        self.assertEqual(final.plate_status, "invalid")

    def test_no_plate_decision(self) -> None:
        final = _finalize_track(("s", 1, 0), [], None, {}, {"selected_plate_candidate": None}, {}, PlateValidationConfig())
        self.assertEqual(final.plate_status, "no_plate_detected")


if __name__ == "__main__":
    unittest.main()
