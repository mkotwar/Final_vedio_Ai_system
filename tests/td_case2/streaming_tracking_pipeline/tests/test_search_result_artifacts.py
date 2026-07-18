from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.search_metrics import build_search_metrics
from tests.td_case2.streaming_tracking_pipeline.search_query_parser import parse_vehicle_search_query
from tests.td_case2.streaming_tracking_pipeline.search_query_schemas import VehicleSearchResponse
from tests.td_case2.streaming_tracking_pipeline.search_result_artifacts import SearchResultArtifactSink
from tests.td_case2.streaming_tracking_pipeline.search_result_ranker import rank_vehicle_record
from tests.td_case2.streaming_tracking_pipeline.serialization import read_json


class SearchResultArtifactTests(unittest.TestCase):
    def test_writes_search_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            query = parse_vehicle_search_query("white car")
            result = rank_vehicle_record(_record(), query)
            response = VehicleSearchResponse(query=query, total_records_searched=1, total_matches=1, results=[result], runtime_sec=0.01)
            metrics = build_search_metrics([response], records_indexed=1)
            paths = SearchResultArtifactSink(temp_dir).write(
                index_summary={"records_indexed": 1},
                responses=[response],
                summary=metrics,
                report={"summary": metrics},
            )
            self.assertTrue(Path(paths["validation_search_results"]).exists())
            self.assertEqual(read_json(paths["summary"])["records_indexed"], 1)


def _record() -> dict[str, object]:
    return {
        "record_id": "r1",
        "source_id": "s",
        "track_id": 1,
        "track_generation": 0,
        "object_class": "car",
        "normalized_colour": "white",
        "plate_status": "verified",
        "plate_text": "UP81CH4158",
        "first_seen_sec": 1.0,
        "last_seen_sec": 2.0,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg",
        "searchable_tokens": ["vehicle", "car", "white", "up81ch4158"],
        "search_text": "vehicle car white up81ch4158",
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
