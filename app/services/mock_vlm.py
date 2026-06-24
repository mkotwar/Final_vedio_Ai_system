import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from app.schemas.frame import FrameRichMetadata
from app.core.utils import calculate_time_snippet
from app.services.vlm_utils import format_timestamp_human_vlm, generate_search_text

class MockVLMService:
    """Mock service generating synthetic plausible rich frame metadata for development and tests."""

    @classmethod
    def _generate_mock_metadata(
        cls, frame_id: str, video_id: str, timestamp_seconds: float
    ) -> FrameRichMetadata:
        """Generates synthetic plausible rich frame metadata for development and tests."""
        ts_human = format_timestamp_human_vlm(timestamp_seconds)

        parts = frame_id.split("_f")
        frame_idx_str = parts[-1] if len(parts) > 1 else "0001"
        frame_path = f"data/frames/{video_id}/frame_{frame_idx_str}.jpg"

        is_even = int(timestamp_seconds) % 2 == 0

        if is_even:
            mock_data = {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_seconds": timestamp_seconds,
                "timestamp_human": ts_human,
                "frame_path": frame_path,
                "scene_type": "outdoor street",
                "scene_description": "An outdoor city street view under daylight.",
                "objects": [
                    {
                        "id": "car_1",
                        "type": "vehicle",
                        "subtype": "car",
                        "color": "blue",
                        "condition": "moving",
                        "attributes": ["moving", "sedan"],
                    },
                    {
                        "id": "person_1",
                        "type": "pedestrian",
                        "subtype": "person",
                        "color": "black",
                        "condition": "normal",
                        "attributes": ["walking", "carrying bag"],
                    },
                ],
                "events": [],
                "people_count": 1,
                "activities": ["driving", "walking"],
                "keywords": ["street", "traffic", "daylight", "city"],
                "caption": "A blue car driving down a busy city street while a pedestrian walks on the sidewalk.",
            }
        else:
            mock_data = {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_seconds": timestamp_seconds,
                "timestamp_human": ts_human,
                "frame_path": frame_path,
                "scene_type": "indoor office",
                "scene_description": "An indoor office meeting room workspace.",
                "objects": [
                    {
                        "id": "chair_1",
                        "type": "furniture",
                        "subtype": "chair",
                        "color": "grey",
                        "condition": "normal",
                        "attributes": ["office chair", "mesh back"],
                    },
                    {
                        "id": "laptop_1",
                        "type": "electronics",
                        "subtype": "laptop",
                        "color": "silver",
                        "condition": "normal",
                        "attributes": ["open", "on table"],
                    },
                ],
                "events": [],
                "people_count": 0,
                "activities": ["working", "sitting"],
                "keywords": ["office", "workplace", "corporate", "desk"],
                "caption": "A laptop open on a table next to a grey office chair in an empty conference room.",
            }

        time_snippet = calculate_time_snippet(timestamp_seconds, interval_seconds=1.0)
        mock_ocr = {
            "detected_text": ["GATE 1", "MH12AB1234"] if is_even else ["OFFICE ENTRY", "VISITOR"],
            "license_plates": ["MH12AB1234"] if is_even else [],
        }
        mock_data.update(time_snippet)
        mock_data["ocr"] = mock_ocr
        mock_data["detected_objects"] = []
        mock_data["tracked_entities"] = []
        mock_data["track_ids"] = []
        mock_data["candidate_reasons"] = []
        mock_data["object_counts"] = {}
        mock_data["search_text"] = generate_search_text(mock_data)

        return FrameRichMetadata(**mock_data)

    @classmethod
    async def generate_metadata_batch(
        cls, batch_frames: List[Tuple[str, str, float, Path]]
    ) -> List[Tuple[FrameRichMetadata, Dict[str, float]]]:
        """Runs batch image inference on MockVLMService to extract structured metadata.

        Args:
            batch_frames: List of tuples (frame_id, video_id, timestamp_seconds, frame_absolute_path)

        Returns:
            list: List of tuples of validated FrameRichMetadata objects and dummy timings.
        """
        logger.info(f"MockVLMService processing batch of {len(batch_frames)} frames...")
        res = []
        for frame_tuple in batch_frames:
            frame_id, video_id, ts = frame_tuple[:3]
            mock_meta = cls._generate_mock_metadata(frame_id, video_id, ts)
            res.append((mock_meta, {
                "ocr_ms": 0.0,
                "vlm_ms": 0.0,
                "json_repair_ms": 0.0,
                "validation_ms": 0.0,
            }))
        return res
