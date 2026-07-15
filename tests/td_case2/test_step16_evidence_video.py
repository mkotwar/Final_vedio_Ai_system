from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_checks import write_json
from step_16_evidence_video_generation import (  # noqa: E402
    EVIDENCE_INDEX_NAME,
    EVIDENCE_VIDEO_NAME,
    REPORT_NAME,
    EvidenceVideoConfig,
    build_evidence_video,
    select_evidence_events,
)


class Step16EvidenceVideoTests(unittest.TestCase):
    def _make_run_dir(self) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="td_case2_step16_"))
        for index, color in enumerate([(255, 255, 255), (220, 220, 220), (180, 180, 180)], start=1):
            image = np.full((120, 160, 3), color, dtype=np.uint8)
            image_path = temp_dir / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(image_path), image)

        write_json(
            temp_dir / "01_video_info.json",
            {
                "input_video_path": str(temp_dir / "input.mp4"),
                "video_name": "input.mp4",
                "duration_seconds": 12.0,
                "duration_text": "00:12",
                "width": 160,
                "height": 120,
            },
        )
        write_json(
            temp_dir / "02A_adaptive_frames.json",
            {
                "selected_frames": [
                    {"frame_id": "frame_000001", "timestamp_seconds": 1.0, "image_path": "frame_000001.jpg"},
                    {"frame_id": "frame_000002", "timestamp_seconds": 2.0, "image_path": "frame_000002.jpg"},
                    {"frame_id": "frame_000003", "timestamp_seconds": 3.0, "image_path": "frame_000003.jpg"},
                ]
            },
        )
        write_json(
            temp_dir / "04B_tracks.json",
            {
                "tracks": [
                    {
                        "track_id": "vehicle_track_0001",
                        "detections": [
                            {
                                "frame_id": "frame_000001",
                                "timestamp_seconds": 1.0,
                                "class_name": "car",
                                "bbox_xyxy": [10, 10, 90, 80],
                            },
                            {
                                "frame_id": "frame_000002",
                                "timestamp_seconds": 2.0,
                                "class_name": "car",
                                "bbox_xyxy": [14, 12, 94, 82],
                            },
                        ],
                    },
                    {
                        "track_id": "person_track_0002",
                        "detections": [
                            {
                                "frame_id": "frame_000003",
                                "timestamp_seconds": 3.0,
                                "class_name": "person",
                                "bbox_xyxy": [100, 20, 130, 100],
                            }
                        ],
                    },
                ]
            },
        )
        write_json(
            temp_dir / "11_full_scene_event_candidates.json",
            {
                "candidate_events": [
                    {
                        "candidate_event_id": "scene_evt_000001",
                        "event_type": "possible_collision_or_near_miss",
                        "context_start_seconds": 0.5,
                        "context_end_seconds": 2.5,
                        "best_timestamp_seconds": 2.0,
                        "involved_track_ids": ["vehicle_track_0001"],
                        "involved_classes": ["car"],
                        "representative_frame": {"image_path": "frame_000002.jpg"},
                    }
                ]
            },
        )
        write_json(
            temp_dir / "12_selected_top_event_candidates.json",
            {
                "selected_candidates": [
                    {
                        "candidate_event_id": "scene_evt_000001",
                        "event_type": "possible_collision_or_near_miss",
                        "context_start_seconds": 0.5,
                        "context_end_seconds": 2.5,
                        "best_timestamp_seconds": 2.0,
                        "ranking_score": 0.62,
                        "representative_frame_path": "frame_000002.jpg",
                    }
                ]
            },
        )
        write_json(
            temp_dir / "14_vlm_event_reviews.json",
            {
                "reviews": [
                    {
                        "source_candidate_ids": ["scene_evt_000001"],
                        "model_review": {
                            "review_decision": "normal_context",
                            "summary_caption": "Normal traffic only.",
                            "confidence": 0.4,
                        },
                    }
                ]
            },
        )
        write_json(
            temp_dir / "14_final_video_summary.json",
            {"overall_status": "normal_no_clear_event_detected"}
        )
        (temp_dir / "13_vlm_event_inputs").mkdir(parents=True, exist_ok=True)
        strip_image = np.full((90, 240, 3), (120, 160, 220), dtype=np.uint8)
        cv2.imwrite(str(temp_dir / "13_vlm_event_inputs" / "vlm_input_000001_strip.jpg"), strip_image)
        write_json(
            temp_dir / "13_vlm_event_inputs.json",
            {
                "vlm_inputs": [
                    {
                        "vlm_input_id": "vlm_input_000001",
                        "best_timestamp_seconds": 2.0,
                        "best_timestamp_text": "00:02",
                        "source_candidate_ids": ["scene_evt_000001"],
                        "source_event_types": ["possible_collision_or_near_miss"],
                        "media": {
                            "temporal_strip_path": "13_vlm_event_inputs/vlm_input_000001_strip.jpg",
                            "contact_sheet_path": None,
                            "primary_frame_path": "frame_000002.jpg",
                        },
                    }
                ]
            },
        )
        write_json(
            temp_dir / "07B_traffic_object_search_index.json",
            {
                "records": [
                    {
                        "object_record_id": "obj_track_vehicle_track_0001",
                        "track_id": "vehicle_track_0001",
                        "object_type": "vehicle",
                        "class_name": "car",
                        "timestamp_seconds": 1.5,
                        "first_seen_seconds": 1.0,
                        "last_seen_seconds": 2.0,
                        "confidence": 0.93,
                        "quality": "fallback",
                        "verified_vehicle_color": "white",
                        "verified_license_plate": "DL1AB1234",
                        "possible_plate_text": None,
                        "weak_ocr_text": ["DL1AB1234"],
                        "full_frame_path": "frame_000002.jpg",
                        "frame_id": "frame_000002",
                        "bbox_xyxy": [12, 10, 96, 84],
                        "search_text": "white car DL1AB1234",
                    },
                    {
                        "object_record_id": "obj_track_vehicle_track_dup",
                        "track_id": "vehicle_track_0001",
                        "object_type": "vehicle",
                        "class_name": "car",
                        "timestamp_seconds": 2.2,
                        "first_seen_seconds": 2.0,
                        "last_seen_seconds": 2.2,
                        "confidence": 0.88,
                        "quality": "fallback",
                        "verified_vehicle_color": "white",
                        "verified_license_plate": "DL1AB1234",
                        "possible_plate_text": None,
                        "weak_ocr_text": ["DL1AB1234"],
                        "full_frame_path": "frame_000002.jpg",
                        "frame_id": "frame_000002",
                        "bbox_xyxy": [12, 10, 96, 84],
                        "search_text": "duplicate white car DL1AB1234",
                    },
                    {
                        "object_record_id": "obj_track_person_track_0002",
                        "track_id": "person_track_0002",
                        "object_type": "person",
                        "class_name": "person",
                        "timestamp_seconds": 3.0,
                        "first_seen_seconds": 3.0,
                        "last_seen_seconds": 3.0,
                        "confidence": 0.7,
                        "quality": "primary",
                        "verified_vehicle_color": None,
                        "verified_license_plate": None,
                        "possible_plate_text": None,
                        "weak_ocr_text": [],
                        "full_frame_path": "frame_000003.jpg",
                        "frame_id": "frame_000003",
                        "bbox_xyxy": [100, 20, 130, 100],
                        "search_text": "person detected",
                    },
                ]
            },
        )
        return temp_dir

    def test_select_evidence_events_prefers_searchable_object_fallback_and_dedupes_plate(self) -> None:
        run_dir = self._make_run_dir()
        config = EvidenceVideoConfig(
            clip_fps=2,
            header_seconds=1.0,
            summary_seconds=1.0,
            object_context_before_seconds=0.5,
            object_context_after_seconds=0.5,
            max_object_events=10,
            include_person_events=True,
            include_normal_context_scene_events=False,
        )

        events, diagnostics = select_evidence_events(run_dir, config)

        self.assertEqual(diagnostics["scene_events_selected"], 0)
        self.assertEqual(diagnostics["object_events_selected"], 2)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["license_plate"], "DL1AB1234")
        self.assertEqual(events[1]["class_names"], ["person"])

    def test_build_evidence_video_writes_video_index_and_report(self) -> None:
        run_dir = self._make_run_dir()
        config = EvidenceVideoConfig(
            clip_fps=2,
            header_seconds=1.0,
            summary_seconds=1.0,
            object_context_before_seconds=0.5,
            object_context_after_seconds=0.5,
            max_object_events=10,
            include_person_events=True,
            include_normal_context_scene_events=False,
        )

        index_payload, report_payload = build_evidence_video(run_dir, config)

        self.assertEqual(index_payload["status"], "success")
        self.assertEqual(report_payload["status"], "success")
        self.assertEqual(report_payload["event_count"], 5)
        self.assertEqual(report_payload["vehicles_detected"], 1)
        self.assertEqual(report_payload["persons_detected"], 1)
        self.assertEqual(report_payload["license_plates_detected"], 1)
        self.assertEqual(report_payload["object_detection_gallery_frame_count"], 2)
        self.assertEqual(report_payload["vlm_input_gallery_frame_count"], 1)
        self.assertTrue((run_dir / EVIDENCE_VIDEO_NAME).exists())
        self.assertTrue((run_dir / EVIDENCE_INDEX_NAME).exists())
        self.assertTrue((run_dir / REPORT_NAME).exists())

        saved_index = json.loads((run_dir / EVIDENCE_INDEX_NAME).read_text(encoding="utf-8"))
        self.assertEqual(len(saved_index["clips"]), 5)
        self.assertIn("clip_content_start_seconds", saved_index["clips"][0])
        self.assertIn("object_detection_frame_gallery", [item["source_type"] for item in saved_index["clips"]])
        self.assertIn("vlm_input_gallery", [item["source_type"] for item in saved_index["clips"]])


if __name__ == "__main__":
    unittest.main()
