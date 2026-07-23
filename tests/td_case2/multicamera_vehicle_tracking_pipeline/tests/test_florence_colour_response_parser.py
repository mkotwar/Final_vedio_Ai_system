from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_colour_response_parser import parse_florence_colour_response
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_mapping import SUPPORTED_VEHICLE_COLOURS


class FlorenceColourResponseParserTests(unittest.TestCase):
    def test_valid_json_is_parsed(self) -> None:
        result = parse_florence_colour_response(
            '{"primary_colour": "WHITE", "secondary_colour": null, "confidence": 0.84}',
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "WHITE")
        self.assertEqual(result.confidence, 0.84)

    def test_markdown_json_and_gray_normalization(self) -> None:
        result = parse_florence_colour_response(
            "```json\n{\"primary_colour\": \"gray\"}\n```",
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "GREY")
        self.assertEqual(result.confidence, 0.5)

    def test_plain_label_is_supported(self) -> None:
        result = parse_florence_colour_response(
            "navy",
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "BLUE")


if __name__ == "__main__":
    unittest.main()
