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

    def test_special_tokens_are_cleaned_before_normalization(self) -> None:
        grey = parse_florence_colour_response("</s><s>grey</s>", allowed_colours=SUPPORTED_VEHICLE_COLOURS, default_confidence=0.5)
        blue = parse_florence_colour_response("</s><s>blue</s>", allowed_colours=SUPPORTED_VEHICLE_COLOURS, default_confidence=0.5)
        self.assertEqual(grey.primary_colour, "GREY")
        self.assertEqual(grey.cleaned_output, "grey")
        self.assertEqual(blue.primary_colour, "BLUE")
        self.assertEqual(blue.cleaned_output, "blue")

    def test_explanatory_text_is_supported(self) -> None:
        result = parse_florence_colour_response(
            "The vehicle is white.",
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "WHITE")

    def test_json_primary_color_key_is_supported(self) -> None:
        result = parse_florence_colour_response(
            '{"primary_color": "silver", "confidence": 0.8}',
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "SILVER")
        self.assertEqual(result.confidence, 0.8)

    def test_multiple_ambiguous_colours_return_unknown(self) -> None:
        result = parse_florence_colour_response(
            "The vehicle is red and blue.",
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "UNKNOWN")

    def test_greenhouse_does_not_match_green(self) -> None:
        result = parse_florence_colour_response(
            "greenhouse",
            allowed_colours=SUPPORTED_VEHICLE_COLOURS,
            default_confidence=0.5,
        )
        self.assertEqual(result.primary_colour, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
