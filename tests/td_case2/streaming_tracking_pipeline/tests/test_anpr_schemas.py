from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import (
    FlorenceColourResult,
    normalize_raw_plate_text,
    normalize_vehicle_colour,
)


class AnprSchemaTests(unittest.TestCase):
    def test_raw_plate_normalization_is_not_final_validation(self) -> None:
        self.assertEqual(normalize_raw_plate_text(" <OCR> mh-12 ab 1234 "), "MH12AB1234")
        self.assertEqual(normalize_raw_plate_text("not a plate"), "NOTAPLATE")

    def test_colour_normalization(self) -> None:
        self.assertEqual(normalize_vehicle_colour("dark grey car"), "gray")
        self.assertEqual(normalize_vehicle_colour("primary color is white"), "white")
        self.assertEqual(normalize_vehicle_colour("not visible"), "unknown")

    def test_colour_result_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            FlorenceColourResult(
                source_id="s",
                track_id=1,
                track_generation=0,
                crop_role="primary",
                crop_rank=1,
                frame_index=1,
                vehicle_crop_path="crop.jpg",
                raw_text="white",
                normalized_colour="white",
                status="verified",
                prompt="<VQA>",
            )


if __name__ == "__main__":
    unittest.main()
