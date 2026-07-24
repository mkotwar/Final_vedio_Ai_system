from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_text_cleaning import clean_florence_text


class FlorenceTextCleaningTests(unittest.TestCase):
    def test_token_wrapped_labels_are_cleaned(self) -> None:
        self.assertEqual(clean_florence_text("</s><s>grey</s>"), "grey")
        self.assertEqual(clean_florence_text("</s><s>blue</s>"), "blue")
        self.assertEqual(clean_florence_text("<pad> white </pad>"), "white")

    def test_repeated_tokens_and_plain_text(self) -> None:
        self.assertEqual(clean_florence_text("<s><s>blue</s></s>"), "blue")
        self.assertEqual(clean_florence_text("  RED  "), "RED")
        self.assertEqual(clean_florence_text("Vehicle colour: silver"), "Vehicle colour: silver")

    def test_empty_and_unknown_xml_like_text(self) -> None:
        self.assertEqual(clean_florence_text(""), "")
        self.assertEqual(clean_florence_text("<vehicle>blue</vehicle>"), "<vehicle>blue</vehicle>")


if __name__ == "__main__":
    unittest.main()
