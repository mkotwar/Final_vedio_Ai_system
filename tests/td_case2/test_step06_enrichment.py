from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_td_case2_step04a_florence_audit import _read_native_tasks
from step_06_ocr_color_enrichment import (
    _parse_color_from_text,
    _select_best_plate_ocr_variant,
    extract_structured_florence_metadata,
)
from vehicle_color import (
    dominant_vehicle_color,
    extract_florence_vehicle_color,
    normalize_color_phrase,
    resolve_vehicle_color,
)


class Step06EnrichmentTests(unittest.TestCase):
    def test_florence_audit_never_schedules_detailed_caption(self) -> None:
        with patch.dict(
            os.environ,
            {"TD_CASE2_FLORENCE_AUDIT_NATIVE_TASKS": "<OCR>,<DETAILED_CAPTION>,<CAPTION>"},
        ):
            self.assertEqual(_read_native_tasks(), ["<OCR>", "<CAPTION>"])

    def test_structured_metadata_only_uses_explicit_caption_evidence(self) -> None:
        metadata = extract_structured_florence_metadata(
            caption_text=(
                "A white Toyota Innova SUV with a roof rack, shown in rear view "
                "on a highway at night with a yellow license plate."
            ),
            ocr_text="DL12CT8289",
            plate_found=True,
            plate_confidence=0.81,
            plate_text="DL12CT8289",
            plate_valid=True,
        )

        self.assertEqual(metadata["vehicle_attributes"]["color"], "white")
        self.assertEqual(metadata["vehicle_attributes"]["make"], "toyota")
        self.assertEqual(metadata["vehicle_attributes"]["model"], "innova")
        self.assertEqual(metadata["vehicle_attributes"]["body_type"], "suv")
        self.assertTrue(metadata["vehicle_attributes"]["roof_rack"])
        self.assertEqual(metadata["vehicle_attributes"]["view"], "rear view")
        self.assertEqual(metadata["license_plate_attributes"]["country_region_hint"], "India state/territory code DL")
        self.assertEqual(metadata["scene_attributes"]["road_type"], "highway")
        self.assertEqual(metadata["scene_attributes"]["lighting"], "night")

    def test_missing_evidence_remains_null(self) -> None:
        metadata = extract_structured_florence_metadata(
            caption_text="A vehicle is visible.",
            ocr_text="",
            plate_found=False,
            plate_confidence=0.0,
            plate_text="",
            plate_valid=False,
        )
        self.assertIsNone(metadata["vehicle_attributes"]["make"])
        self.assertIsNone(metadata["vehicle_attributes"]["model"])
        self.assertIsNone(metadata["scene_attributes"]["weather"])

    def test_first_caption_color_wins(self) -> None:
        self.assertEqual(_parse_color_from_text("A red car with black wheels."), "red")

    def test_free_form_shades_are_normalized_for_search(self) -> None:
        expected = {
            "pearl white": "white",
            "ivory": "white",
            "navy blue": "blue",
            "metallic blue": "blue",
            "burgundy": "red",
            "maroon": "red",
            "champagne gold": "gold",
            "bronze": "brown",
            "charcoal": "gray",
            "graphite": "gray",
            "silver gray": "gray",
        }
        for shade, canonical in expected.items():
            with self.subTest(shade=shade):
                self.assertEqual(normalize_color_phrase(shade), canonical)

    def test_plate_color_is_not_mistaken_for_vehicle_color(self) -> None:
        raw, canonical = extract_florence_vehicle_color("A car with a yellow license plate.")
        self.assertIsNone(raw)
        self.assertIsNone(canonical)

    def test_image_fallback_returns_canonical_color(self) -> None:
        blue_image = np.full((80, 120, 3), (255, 0, 0), dtype=np.uint8)
        color, confidence = dominant_vehicle_color(blue_image)
        self.assertEqual(color, "blue")
        self.assertGreater(confidence, 0.5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "vehicle.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), blue_image))
            result = resolve_vehicle_color("A vehicle with a yellow license plate.", image_path)
        self.assertEqual(result["color"], "blue")
        self.assertEqual(result["source"], "image_dominant_color")

    def test_original_plate_ocr_wins_when_both_variants_are_valid(self) -> None:
        selected = _select_best_plate_ocr_variant(
            [
                {"variant": "padded_original", "ocr_raw": "DL14CK6033"},
                {"variant": "enhanced", "ocr_raw": "OL14CK6033"},
            ]
        )
        self.assertEqual(selected["variant"], "padded_original")

    def test_enhanced_plate_ocr_can_recover_invalid_original(self) -> None:
        selected = _select_best_plate_ocr_variant(
            [
                {"variant": "padded_original", "ocr_raw": "-"},
                {"variant": "enhanced", "ocr_raw": "DL14CK6033"},
            ]
        )
        self.assertEqual(selected["variant"], "enhanced")


if __name__ == "__main__":
    unittest.main()
