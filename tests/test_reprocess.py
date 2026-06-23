from app.core.config import settings
from app.services.event_aggregation import EventAggregationService
from app.services.pipeline_contract import event_catalog_path, frame_catalog_path, write_json_file


def test_reprocess_video_from_frame_catalog(tmp_path, monkeypatch):
    original_metadata_dir = settings.METADATA_DIR
    original_events_dir = settings.EVENTS_DIR
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.EVENTS_DIR = tmp_path / "events"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    indexed = {}

    def fake_index_events(video_id, events):
        indexed["video_id"] = video_id
        indexed["events"] = events
        return True

    monkeypatch.setattr("app.services.search_service.SearchService.index_events", fake_index_events)

    try:
        from app.services.search_service import SearchService

        video_id = "reprocess-test-video"
        frames = [
            {
                "frame_id": f"{video_id}_f0001",
                "video_id": video_id,
                "timestamp_seconds": 0.0,
                "timestamp_start_seconds": 0.0,
                "timestamp_end_seconds": 1.0,
                "scene_type": "outdoor street",
                "caption": "A blue car driving on the street.",
                "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
                "activities": ["driving"],
            },
            {
                "frame_id": f"{video_id}_f0002",
                "video_id": video_id,
                "timestamp_seconds": 1.0,
                "timestamp_start_seconds": 1.0,
                "timestamp_end_seconds": 2.0,
                "scene_type": "outdoor street",
                "caption": "A blue car driving on the street.",
                "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
                "activities": ["driving"],
            },
        ]
        write_json_file(frame_catalog_path(video_id), frames)

        new_events = EventAggregationService.process_events(video_id, frames)
        assert len(new_events) == 1
        assert event_catalog_path(video_id).exists()

        assert SearchService.index_events(video_id, new_events) is True
        assert indexed["video_id"] == video_id
        assert indexed["events"] == new_events
    finally:
        settings.METADATA_DIR = original_metadata_dir
        settings.EVENTS_DIR = original_events_dir
