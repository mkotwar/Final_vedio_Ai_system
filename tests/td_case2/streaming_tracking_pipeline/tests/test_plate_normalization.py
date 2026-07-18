from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.plate_normalization import extract_plate_substrings, normalize_plate_ocr_text


class PlateNormalizationTests(unittest.TestCase):
    def test_uppercase_punctuation_and_word_removal(self) -> None:
        result = normalize_plate_ocr_text(" up-16 bh 0400 Sport! ")
        self.assertEqual(result.normalized_text, "UP16BH0400")
        self.assertEqual(result.raw_text, " up-16 bh 0400 Sport! ")
        self.assertEqual(result.removed_words, ["SPORT"])

    def test_plate_substring_extraction(self) -> None:
        values = extract_plate_substrings("XXUP16BH0400RE3986")
        self.assertIn("UP16BH0400", values)
        self.assertLessEqual(len(values), 30)


if __name__ == "__main__":
    unittest.main()
