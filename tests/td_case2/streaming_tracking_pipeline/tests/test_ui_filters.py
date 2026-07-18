from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.ui_filters import (
    UIRecordFilters,
    available_filter_values,
    filter_records,
    paginate_records,
    record_has_plate,
    time_overlaps,
)


class UIFilterTests(unittest.TestCase):
    def test_class_colour_and_person_vehicle_separation(self) -> None:
        records = [
            _record("car1", "car", "white", "verified", plate="UP81CH4158"),
            _record("person1", "person", "white", "no_plate_detected"),
            _record("moto1", "motorcycle", "black", "no_plate_detected"),
        ]

        white_cars = filter_records(records, UIRecordFilters(object_class="car", dominant_colour="white"))
        people = filter_records(records, UIRecordFilters(object_class="person"))

        self.assertEqual(["car1"], [record["record_id"] for record in white_cars])
        self.assertEqual(["person1"], [record["record_id"] for record in people])

    def test_plate_filters_do_not_treat_invalid_ocr_as_valid_plate(self) -> None:
        records = [
            _record("verified", "car", "white", "verified", plate="UP81CH4158"),
            _record("weak", "car", "white", "weak", plate="UP81AB1111"),
            _record("invalid", "car", "white", "invalid", plate="UP81BAD999"),
        ]

        exact = filter_records(records, UIRecordFilters(exact_plate="UP81BAD999"))
        prefix_no_weak = filter_records(records, UIRecordFilters(plate_prefix="UP81", include_weak_plates=False))

        self.assertEqual([], exact)
        self.assertEqual(["verified"], [record["record_id"] for record in prefix_no_weak])

    def test_time_filter_uses_interval_overlap(self) -> None:
        record = _record("r1", "truck", "red", "verified", first=60.0, last=120.0, plate="UP81XX0001")

        self.assertTrue(time_overlaps(record, 119.0, 130.0))
        self.assertTrue(time_overlaps(record, 30.0, 60.0))
        self.assertFalse(time_overlaps(record, 121.0, 130.0))

    def test_text_query_reuses_step10_search_and_ranking(self) -> None:
        records = [
            _record("verified", "car", "white", "verified", plate="UP81CH4158"),
            _record("weak", "car", "white", "weak", plate="UP81CH4159"),
            _record("truck", "truck", "red", "no_plate_detected"),
        ]

        results = filter_records(records, UIRecordFilters(text_query="UP81", include_weak_plates=True))

        self.assertEqual(["verified", "weak"], [record["record_id"] for record in results])
        self.assertLess(results[0]["_ui_rank"], results[1]["_ui_rank"])

    def test_records_with_and_without_plates(self) -> None:
        records = [
            _record("verified", "car", "white", "verified", plate="UP81CH4158"),
            _record("weak", "car", "white", "weak", plate="UP81CH4159"),
            _record("none", "car", "white", "no_plate_detected"),
        ]

        with_plate = filter_records(records, UIRecordFilters(plate_presence="with_plate", include_weak_plates=False))
        without_plate = filter_records(records, UIRecordFilters(plate_presence="without_plate", include_weak_plates=True))

        self.assertTrue(record_has_plate(records[1], include_weak_plates=True))
        self.assertEqual(["verified"], [record["record_id"] for record in with_plate])
        self.assertEqual(["none"], [record["record_id"] for record in without_plate])

    def test_pagination_and_available_values(self) -> None:
        records = [_record(f"r{index}", "car", "white", "verified") for index in range(7)]
        page, meta = paginate_records(records, page=2, page_size=3)
        values = available_filter_values(records)

        self.assertEqual(["r3", "r4", "r5"], [record["record_id"] for record in page])
        self.assertEqual(3, meta["total_pages"])
        self.assertEqual(["car"], values["classes"])
        self.assertEqual(["white"], values["colours"])


def _record(
    record_id: str,
    object_class: str,
    colour: str,
    status: str,
    *,
    plate: str | None = None,
    first: float = 1.0,
    last: float = 2.0,
) -> dict[str, object]:
    tokens = [object_class, colour, status]
    if plate:
        tokens.append(plate.upper())
    return {
        "record_id": record_id,
        "source_id": "s",
        "track_id": 1,
        "track_generation": 0,
        "object_class": object_class,
        "normalized_class_name": object_class,
        "dominant_colour": colour,
        "normalized_colour": colour,
        "plate_status": status,
        "plate_text": plate,
        "plate_confidence": 0.8 if plate else 0.0,
        "first_seen_sec": first,
        "last_seen_sec": last,
        "searchable_tokens": tokens,
        "search_text": " ".join(tokens),
    }


if __name__ == "__main__":
    unittest.main()
