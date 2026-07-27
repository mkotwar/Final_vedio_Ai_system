from __future__ import annotations

import unittest

from pydantic import ValidationError

from ..api.search_models import NaturalLanguageSearchRequest, ParsedVehicleSearchIntent


class NaturalLanguageSearchModelTests(unittest.TestCase):
    def test_blank_query_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            NaturalLanguageSearchRequest(query=" ")

    def test_oversized_query_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            NaturalLanguageSearchRequest(query="a" * 501)

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ParsedVehicleSearchIntent.model_validate({"vehicle_class": "CAR", "sql": "select * from vehicle_track"})


if __name__ == "__main__":
    unittest.main()
