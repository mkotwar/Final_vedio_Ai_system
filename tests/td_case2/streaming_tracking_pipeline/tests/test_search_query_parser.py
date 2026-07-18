from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.search_query_parser import parse_vehicle_search_query


class SearchQueryParserTests(unittest.TestCase):
    def test_parses_colour_and_class(self) -> None:
        query = parse_vehicle_search_query("white car")
        self.assertEqual(query.object_classes, ["car"])
        self.assertEqual(query.colours, ["white"])

    def test_parses_status_phrases(self) -> None:
        self.assertEqual(parse_vehicle_search_query("verified plates").plate_statuses, ["verified"])
        self.assertEqual(parse_vehicle_search_query("weak OCR").plate_statuses, ["weak"])
        self.assertEqual(parse_vehicle_search_query("vehicles without plates").plate_statuses, ["no_plate_detected"])

    def test_parses_time_ranges(self) -> None:
        seconds = parse_vehicle_search_query("vehicles between 60 and 120 seconds")
        minutes = parse_vehicle_search_query("white car between 2 and 3 minutes")
        self.assertEqual((seconds.start_time_sec, seconds.end_time_sec), (60.0, 120.0))
        self.assertEqual((minutes.start_time_sec, minutes.end_time_sec), (120.0, 180.0))

    def test_parses_plate_exact_and_prefix(self) -> None:
        self.assertEqual(parse_vehicle_search_query("UP81CH4158").plate_text, "UP81CH4158")
        self.assertEqual(parse_vehicle_search_query("UP81").plate_prefix, "UP81")

    def test_unknown_words_remain_free_text(self) -> None:
        self.assertEqual(parse_vehicle_search_query("shiny white car").free_text_tokens, ["shiny"])

    def test_parses_person_and_two_wheelers(self) -> None:
        self.assertEqual(parse_vehicle_search_query("person in white").object_classes, ["person"])
        self.assertEqual(parse_vehicle_search_query("bicycle").object_classes, ["bicycle"])
        self.assertEqual(parse_vehicle_search_query("scooter").object_classes, ["motorcycle"])
        self.assertEqual(parse_vehicle_search_query("two wheeler").object_classes, ["motorcycle"])


if __name__ == "__main__":
    unittest.main()
