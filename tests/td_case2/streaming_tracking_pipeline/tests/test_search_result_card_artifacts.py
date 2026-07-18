from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.search_result_card_artifacts import SearchResultCardArtifactSink
from tests.td_case2.streaming_tracking_pipeline.search_result_card_schemas import VehicleResultCard, VehicleResultCardPackage
from tests.td_case2.streaming_tracking_pipeline.serialization import read_json, read_jsonl


class SearchResultCardArtifactTests(unittest.TestCase):
    def test_writes_card_artifacts_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package = VehicleResultCardPackage(
                raw_query="white car",
                parsed_query={"raw_query": "white car"},
                total_matches=1,
                returned_cards=1,
                cards=[_card()],
                runtime_sec=0.1,
            )
            paths = SearchResultCardArtifactSink(temp_dir).write(
                packages=[package],
                summary={"cards_created": 1},
                report={"summary": {"cards_created": 1}},
                write_html_preview=True,
            )
            self.assertTrue(Path(paths["result_card_packages"]).exists())
            self.assertTrue(Path(paths["html_preview"]).exists())
            self.assertEqual(read_json(paths["summary"])["cards_created"], 1)
            self.assertEqual(read_jsonl(paths["result_card_packages"])[0]["raw_query"], "white car")


def _card() -> VehicleResultCard:
    return VehicleResultCard(
        rank=1,
        record_id="r1",
        source_id="s",
        track_id=1,
        track_generation=0,
        title="White Car - UP81CH4158",
        subtitle="Track 1 · 1.0s-2.0s",
        time_label="1.0s-2.0s",
        plate_label="UP81CH4158",
        colour_label="White",
        status_badge="Verified",
        confidence_label="0.900",
        object_class="car",
        colour="white",
        plate_text="UP81CH4158",
        plate_status="verified",
        plate_confidence=0.9,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        duration_sec=1.0,
        thumbnail_path="vehicle.jpg",
        secondary_image_path="plate.jpg",
        search_score=10.0,
    )


if __name__ == "__main__":
    unittest.main()
