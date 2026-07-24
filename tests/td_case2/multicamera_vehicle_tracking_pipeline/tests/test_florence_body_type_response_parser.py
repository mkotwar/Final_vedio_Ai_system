from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_body_type_response_parser import parse_florence_body_type_response
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_body_type_mapping import SUPPORTED_VEHICLE_BODY_TYPES


class FlorenceBodyTypeResponseParserTests(unittest.TestCase):
    def test_plain_label_supported(self) -> None:
        result = parse_florence_body_type_response("SEDAN", allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(result.canonical_body_type, "SEDAN")
        self.assertEqual(result.status, "SUCCESS")

    def test_lowercase_and_explanation_supported(self) -> None:
        result = parse_florence_body_type_response("Vehicle type: hatchback", allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(result.canonical_body_type, "HATCHBACK")

    def test_valid_json_supported(self) -> None:
        result = parse_florence_body_type_response('{"body_type":"suv","confidence":0.77}', allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(result.canonical_body_type, "SUV")
        self.assertEqual(result.confidence, 0.77)

    def test_fenced_json_supported(self) -> None:
        result = parse_florence_body_type_response("```json\n{\"body_type\":\"pickup truck\"}\n```", allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(result.canonical_body_type, "PICKUP")

    def test_invalid_and_empty_output_become_unknown(self) -> None:
        invalid = parse_florence_body_type_response("helicopter", allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        empty = parse_florence_body_type_response("", allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(invalid.canonical_body_type, "UNKNOWN")
        self.assertEqual(empty.status, "PARSE_ERROR")

    def test_confidence_out_of_range_is_clamped(self) -> None:
        result = parse_florence_body_type_response('{"body_type":"sedan","confidence":3}', allowed_body_types=SUPPORTED_VEHICLE_BODY_TYPES, default_confidence=0.5)
        self.assertEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
