from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.structured_search_index import StructuredVehicleSearchIndex


class StructuredSearchIndexTests(unittest.TestCase):
    def test_filters_colour_class_and_status(self) -> None:
        index = StructuredVehicleSearchIndex([_record("r1", "car", "white", "verified", "UP81CH4158")])
        self.assertEqual(index.search("white car").total_matches, 1)
        self.assertEqual(index.search("truck with verified plate").total_matches, 0)
        self.assertEqual(index.search("verified plates").total_matches, 1)

    def test_time_query_uses_overlap(self) -> None:
        index = StructuredVehicleSearchIndex([_record("r1", "car", "white", "verified", "UP81CH4158", first=100, last=130)])
        self.assertEqual(index.search("vehicles between 120 and 180 seconds").total_matches, 1)

    def test_plate_policy_excludes_invalid_and_can_exclude_weak(self) -> None:
        records = [
            _record("verified", "car", "white", "verified", "UP81CH4158"),
            _record("weak", "car", "white", "weak", "UP81CH4158"),
            _record("invalid", "car", "white", "invalid", None),
        ]
        self.assertEqual(StructuredVehicleSearchIndex(records).search("UP81CH4158").total_matches, 2)
        self.assertEqual(StructuredVehicleSearchIndex(records, include_weak_plates=False).search("UP81CH4158").total_matches, 1)


def _record(
    record_id: str,
    object_class: str,
    colour: str,
    status: str,
    plate: str | None,
    *,
    first: float = 1.0,
    last: float = 2.0,
) -> dict[str, object]:
    tokens = ["vehicle", object_class, colour, status]
    if plate:
        tokens.append(plate.lower())
    return {
        "record_id": record_id,
        "source_id": "s",
        "track_id": 1,
        "track_generation": 0,
        "object_class": object_class,
        "normalized_colour": colour,
        "plate_status": status,
        "plate_text": plate,
        "first_seen_sec": first,
        "last_seen_sec": last,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg" if plate else None,
        "searchable_tokens": tokens,
        "search_text": " ".join(tokens),
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
