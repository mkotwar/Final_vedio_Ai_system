from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_checks import write_json
from step_11_full_scene_event_candidates import (  # noqa: E402
    _build_search_record_indexes,
    _merge_triggers_into_candidates,
    normalize_search_index_records,
    resolve_step11_search_index,
    run_full_scene_event_candidate_generation,
)


def _step11_event_config() -> dict[str, float | int | bool]:
    return {
        "window_seconds": 2.0,
        "window_stride_seconds": 1.0,
        "merge_gap_seconds": 3.0,
        "max_event_seconds": 12.0,
        "context_before_seconds": 1.0,
        "context_after_seconds": 1.0,
        "min_candidate_score": 0.35,
        "top_k_preview": 5,
        "save_flat": True,
        "include_search_metadata": True,
    }


def _make_minimal_run_dir() -> Path:
    run_dir = Path(tempfile.mkdtemp(prefix="td_case2_step11_"))
    write_json(
        run_dir / "01_video_info.json",
        {
            "fps": 25.0,
            "frame_count": 75,
            "duration_seconds": 3.0,
            "width": 1280,
            "height": 720,
        },
    )
    write_json(
        run_dir / "02A_adaptive_frames.json",
        {
            "selected_frames": [
                {
                    "frame_id": "frame_000001",
                    "frame_idx": 1,
                    "timestamp_seconds": 1.0,
                    "image_path": "frame_000001.jpg",
                    "motion_score": 0.01,
                    "motion_pixels_ratio": 0.01,
                    "histogram_change_score": 0.01,
                    "motion_blob_count": 1,
                }
            ]
        },
    )
    write_json(run_dir / "03_yolo_detections.json", {"detections": []})
    write_json(run_dir / "04B_tracks.json", {"tracks": []})
    return run_dir


def _active_record(
    *,
    object_record_id: str = "object_000012",
    track_id: str | None = "vehicle_track_0004",
    detection_id: str | None = "det_0004",
    verified_vehicle_color: str | None = "white",
    verified_license_plate: str | None = "DL01AB1234",
    possible_plate_text: str | None = "DL01AB1235",
    searchable_tokens: list[str] | None = None,
    crop_path: str | None = "03_yolo_object_crops/crop_0004.jpg",
    full_frame_path: str | None = "02A_frames/frame_0004.jpg",
) -> dict[str, object]:
    return {
        "object_record_id": object_record_id,
        "source_type": "track",
        "track_id": track_id,
        "detection_id": detection_id,
        "object_type": "vehicle",
        "class_name": "car",
        "timestamp_seconds": 4.0,
        "first_seen_seconds": 3.0,
        "last_seen_seconds": 5.0,
        "verified_vehicle_color": verified_vehicle_color,
        "verified_license_plate": verified_license_plate,
        "possible_plate_text": possible_plate_text,
        "searchable_tokens": searchable_tokens or ["white", "car", "DL01AB1234"],
        "crop_path": crop_path,
        "full_frame_path": full_frame_path,
    }


def _legacy_record(track_id: str = "vehicle_track_0004") -> dict[str, object]:
    return {
        "search_record_id": "veh_search_000001",
        "track_id": track_id,
        "vehicle_class": "car",
        "dominant_class_name": "car",
        "vehicle_color": "blue",
        "verified_license_plate": "MH01AB1234",
        "verified_license_plate_valid": True,
        "possible_license_plate_candidates": [{"text": "MH01AB1235"}],
        "weak_ocr_text": ["MH01AB12S4"],
        "search_terms": ["blue", "car", "MH01AB1234"],
        "best_crop_path": "05_selected_track_crops/track_0004.jpg",
        "best_full_frame_path": "05_selected_full_frames/frame_0004.jpg",
        "start_timestamp_seconds": 2.0,
        "end_timestamp_seconds": 6.0,
    }


def _candidate_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    raw_triggers = [
        {
            "trigger_id": "raw_evt_trigger_000001",
            "event_type": "sudden_stop",
            "timestamp_seconds": 4.0,
            "timestamp_text": "00:04",
            "window_id": "scene_win_000001",
            "score": 0.6,
            "trigger_reasons": ["sudden_speed_change"],
            "involved_track_ids": ["vehicle_track_0004"],
            "involved_classes": ["car"],
            "representative_frame_path": "frame_000004.jpg",
            "evidence": {
                "window": {
                    "object_count_max": 1,
                    "vehicle_count_max": 1,
                    "person_count_max": 0,
                    "motion_score_max": 0.8,
                    "motion_pixels_ratio_max": 0.4,
                }
            },
        }
    ]
    selected_frames = [
        {
            "frame_id": "frame_000004",
            "frame_idx": 4,
            "timestamp_seconds": 4.0,
            "image_path": "frame_000004.jpg",
        }
    ]
    track_by_id = {
        "vehicle_track_0004": {
            "track_id": "vehicle_track_0004",
            "track_quality": "good",
            "best_detection_id": "det_0004",
        }
    }
    return raw_triggers, selected_frames, track_by_id


class Step11SearchEnrichmentTests(unittest.TestCase):
    def test_active_07b_is_preferred_over_legacy(self) -> None:
        run_dir = _make_minimal_run_dir()
        write_json(run_dir / "07B_traffic_object_search_index.json", {"records": [_active_record()]})
        write_json(run_dir / "07_vehicle_search_index.json", {"records": [_legacy_record()]})

        resolved = resolve_step11_search_index(run_dir)

        self.assertEqual(resolved["status"], "loaded")
        self.assertEqual(resolved["source_type"], "active_07B")
        self.assertEqual(resolved["source_filename"], "07B_traffic_object_search_index.json")
        self.assertFalse(resolved["legacy_fallback_used"])
        self.assertEqual(resolved["records_loaded"], 1)
        self.assertEqual(resolved["records"][0]["object_record_id"], "object_000012")

    def test_legacy_fallback_is_used_when_active_is_missing(self) -> None:
        run_dir = _make_minimal_run_dir()
        write_json(run_dir / "07_vehicle_search_index.json", {"records": [_legacy_record()]})

        resolved = resolve_step11_search_index(run_dir)

        self.assertEqual(resolved["status"], "loaded")
        self.assertEqual(resolved["source_type"], "legacy_07")
        self.assertTrue(resolved["legacy_fallback_used"])
        self.assertEqual(resolved["records_loaded"], 1)
        self.assertEqual(resolved["records"][0]["verified_vehicle_color"], "blue")

    def test_step11_continues_when_no_search_index_exists(self) -> None:
        run_dir = _make_minimal_run_dir()

        output_payload, flat_payload, report_payload, diagnostics_payload = run_full_scene_event_candidate_generation(
            run_dir=run_dir,
            event_config=_step11_event_config(),
        )

        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(flat_payload, [])
        self.assertEqual(report_payload["search_index"]["status"], "missing")
        self.assertEqual(report_payload["search_index"]["records_loaded"], 0)
        self.assertEqual(diagnostics_payload["search_index"]["status"], "missing")

    def test_active_07b_normalization_preserves_required_fields(self) -> None:
        normalized = normalize_search_index_records(
            {"records": [_active_record(searchable_tokens=["white", "car", "DL01AB1234", "DL01AB1235"])]},
            source_type="active_07B",
        )

        self.assertEqual(len(normalized), 1)
        record = normalized[0]
        self.assertEqual(record["track_id"], "vehicle_track_0004")
        self.assertEqual(record["verified_vehicle_color"], "white")
        self.assertEqual(record["verified_license_plate"], "DL01AB1234")
        self.assertEqual(record["possible_plate_text"], ["DL01AB1235"])
        self.assertEqual(record["searchable_tokens"], ["white", "car", "DL01AB1234", "DL01AB1235"])
        self.assertEqual(record["crop_path"], "03_yolo_object_crops/crop_0004.jpg")
        self.assertEqual(record["full_frame_path"], "02A_frames/frame_0004.jpg")

    def test_candidate_is_enriched_by_track_id(self) -> None:
        raw_triggers, selected_frames, track_by_id = _candidate_inputs()
        records = normalize_search_index_records({"records": [_active_record()]}, source_type="active_07B")
        by_track_id, by_detection_id = _build_search_record_indexes(records)

        candidates = _merge_triggers_into_candidates(
            raw_triggers=raw_triggers,
            selected_frames=selected_frames,
            track_by_id=track_by_id,
            record_by_track_id=by_track_id,
            record_by_detection_id=by_detection_id,
            search_source_type="active_07B",
            context_before_seconds=1.0,
            context_after_seconds=1.0,
            merge_gap_seconds=2.0,
            max_event_seconds=6.0,
            video_duration_seconds=8.0,
            include_search_metadata=True,
        )

        enrichment = candidates[0]["search_enrichment"]
        self.assertTrue(enrichment["matched"])
        self.assertEqual(enrichment["matched_record_count"], 1)
        self.assertEqual(enrichment["track_ids"], ["vehicle_track_0004"])
        self.assertEqual(enrichment["verified_plates"], ["DL01AB1234"])

    def test_candidate_does_not_false_match_similar_record(self) -> None:
        raw_triggers, selected_frames, track_by_id = _candidate_inputs()
        records = normalize_search_index_records(
            {"records": [_active_record(track_id="vehicle_track_0008", detection_id="det_0008")]},
            source_type="active_07B",
        )
        by_track_id, by_detection_id = _build_search_record_indexes(records)

        candidates = _merge_triggers_into_candidates(
            raw_triggers=raw_triggers,
            selected_frames=selected_frames,
            track_by_id=track_by_id,
            record_by_track_id=by_track_id,
            record_by_detection_id=by_detection_id,
            search_source_type="active_07B",
            context_before_seconds=1.0,
            context_after_seconds=1.0,
            merge_gap_seconds=2.0,
            max_event_seconds=6.0,
            video_duration_seconds=8.0,
            include_search_metadata=True,
        )

        enrichment = candidates[0]["search_enrichment"]
        self.assertFalse(enrichment["matched"])
        self.assertEqual(enrichment["matched_record_count"], 0)
        self.assertEqual(enrichment["track_ids"], [])

    def test_multiple_records_for_one_candidate_are_deduplicated_in_order(self) -> None:
        raw_triggers, selected_frames, track_by_id = _candidate_inputs()
        records = normalize_search_index_records(
            {
                "records": [
                    _active_record(
                        object_record_id="object_000012",
                        verified_license_plate="DL01AB1234",
                        possible_plate_text="DL01AB1235",
                        searchable_tokens=["white", "car", "DL01AB1234"],
                    ),
                    _active_record(
                        object_record_id="object_000013",
                        verified_license_plate="DL01AB1234",
                        possible_plate_text="DL01AB1236",
                        searchable_tokens=["white", "car", "DL01AB1236"],
                    ),
                ]
            },
            source_type="active_07B",
        )
        by_track_id, by_detection_id = _build_search_record_indexes(records)

        candidates = _merge_triggers_into_candidates(
            raw_triggers=raw_triggers,
            selected_frames=selected_frames,
            track_by_id=track_by_id,
            record_by_track_id=by_track_id,
            record_by_detection_id=by_detection_id,
            search_source_type="active_07B",
            context_before_seconds=1.0,
            context_after_seconds=1.0,
            merge_gap_seconds=2.0,
            max_event_seconds=6.0,
            video_duration_seconds=8.0,
            include_search_metadata=True,
        )

        enrichment = candidates[0]["search_enrichment"]
        self.assertEqual(enrichment["matched_record_count"], 2)
        self.assertEqual(enrichment["object_record_ids"], ["object_000012", "object_000013"])
        self.assertEqual(enrichment["track_ids"], ["vehicle_track_0004"])
        self.assertEqual(enrichment["colors"], ["white"])
        self.assertEqual(enrichment["verified_plates"], ["DL01AB1234"])
        self.assertEqual(enrichment["possible_plate_texts"], ["DL01AB1235", "DL01AB1236"])

    def test_invalid_active_json_falls_back_to_legacy(self) -> None:
        run_dir = _make_minimal_run_dir()
        (run_dir / "07B_traffic_object_search_index.json").write_text("{invalid", encoding="utf-8")
        write_json(run_dir / "07_vehicle_search_index.json", {"records": [_legacy_record()]})

        resolved = resolve_step11_search_index(run_dir)

        self.assertEqual(resolved["status"], "loaded")
        self.assertEqual(resolved["source_type"], "legacy_07")
        self.assertTrue(resolved["legacy_fallback_used"])
        self.assertTrue(any("Failed to load active Step 07B search index" in item for item in resolved["warnings"]))

    def test_candidate_logic_is_unchanged_by_enrichment(self) -> None:
        raw_triggers, selected_frames, track_by_id = _candidate_inputs()
        empty_by_track_id, empty_by_detection_id = _build_search_record_indexes([])
        enriched_records = normalize_search_index_records({"records": [_active_record()]}, source_type="active_07B")
        enriched_by_track_id, enriched_by_detection_id = _build_search_record_indexes(enriched_records)

        baseline_candidates = _merge_triggers_into_candidates(
            raw_triggers=raw_triggers,
            selected_frames=selected_frames,
            track_by_id=track_by_id,
            record_by_track_id=empty_by_track_id,
            record_by_detection_id=empty_by_detection_id,
            search_source_type="none",
            context_before_seconds=1.0,
            context_after_seconds=1.0,
            merge_gap_seconds=2.0,
            max_event_seconds=6.0,
            video_duration_seconds=8.0,
            include_search_metadata=True,
        )
        enriched_candidates = _merge_triggers_into_candidates(
            raw_triggers=raw_triggers,
            selected_frames=selected_frames,
            track_by_id=track_by_id,
            record_by_track_id=enriched_by_track_id,
            record_by_detection_id=enriched_by_detection_id,
            search_source_type="active_07B",
            context_before_seconds=1.0,
            context_after_seconds=1.0,
            merge_gap_seconds=2.0,
            max_event_seconds=6.0,
            video_duration_seconds=8.0,
            include_search_metadata=True,
        )

        self.assertEqual(
            [
                (
                    item["candidate_event_id"],
                    item["event_type"],
                    item["best_timestamp_seconds"],
                    item["candidate_score"],
                    item["involved_track_ids"],
                )
                for item in baseline_candidates
            ],
            [
                (
                    item["candidate_event_id"],
                    item["event_type"],
                    item["best_timestamp_seconds"],
                    item["candidate_score"],
                    item["involved_track_ids"],
                )
                for item in enriched_candidates
            ],
        )
        self.assertFalse(baseline_candidates[0]["search_enrichment"]["matched"])
        self.assertTrue(enriched_candidates[0]["search_enrichment"]["matched"])


if __name__ == "__main__":
    unittest.main()
