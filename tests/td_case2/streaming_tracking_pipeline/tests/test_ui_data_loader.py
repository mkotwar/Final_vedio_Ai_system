from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.ui_data_loader import (
    append_manual_plate_review,
    build_object_evidence,
    ensure_ui_state_files,
    get_record_detail,
    load_manual_plate_reviews,
    load_run_artifacts,
    resolve_artifact_path,
    summarize_records,
)


class UIDataLoaderTests(unittest.TestCase):
    def test_loads_json_jsonl_and_reports_missing_optional_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "01_source" / "source_metadata.json", {"source_fps": 30.0, "target_processing_fps": 10.0})
            _write_jsonl(
                run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl",
                [
                    _record("r1", "car", "white", "verified", plate="UP81CH4158"),
                    _record("r2", "motorcycle", "black", "no_plate_detected"),
                ],
            )

            artifacts = load_run_artifacts(run_dir, repo_root=run_dir)

            self.assertEqual(2, len(artifacts.searchable_records))
            self.assertEqual(30.0, artifacts.source_metadata["source_fps"])
            self.assertTrue(any("04_lifecycle" in missing for missing in artifacts.missing_artifacts))

    def test_summary_counts_person_vehicle_and_plate_statuses(self) -> None:
        records = [
            _record("r1", "car", "white", "verified"),
            _record("r2", "motorcycle", "black", "weak"),
            _record("r3", "person", "blue", "no_plate_detected"),
            _record("r4", "bicycle", "red", "invalid"),
        ]

        summary = summarize_records(records, {"duration_sec": 12.0})

        self.assertEqual(4, summary["total_tracked_objects"])
        self.assertEqual(1, summary["cars"])
        self.assertEqual(1, summary["motorcycles_two_wheelers"])
        self.assertEqual(1, summary["persons"])
        self.assertEqual(1, summary["bicycles"])
        self.assertEqual(1, summary["verified_plates"])
        self.assertEqual(1, summary["weak_plates"])
        self.assertEqual(1, summary["invalid_ocr"])

    def test_resolves_paths_against_run_dir_and_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            image = run_dir / "images" / "crop.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpg")
            repo_image = root / "repo_crop.jpg"
            repo_image.write_bytes(b"jpg")

            self.assertEqual(image, resolve_artifact_path("images/crop.jpg", run_dir=run_dir, repo_root=root))
            self.assertEqual(repo_image, resolve_artifact_path("repo_crop.jpg", run_dir=run_dir, repo_root=root))
            self.assertIsNone(resolve_artifact_path("missing.jpg", run_dir=run_dir, repo_root=root))

    def test_manual_review_writes_separate_ui_state_without_mutating_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source_artifact = run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl"
            _write_jsonl(source_artifact, [_record("r1", "car", "white", "verified")])
            before = source_artifact.read_text(encoding="utf-8")

            paths = ensure_ui_state_files(run_dir)
            review_path = append_manual_plate_review(run_dir, record_id="r1", decision="needs_review", notes="blur")
            reviews = load_manual_plate_reviews(run_dir)

            self.assertEqual(paths["manual_plate_reviews"], review_path)
            self.assertEqual("needs_review", reviews[0]["decision"])
            self.assertEqual(before, source_artifact.read_text(encoding="utf-8"))

    def test_get_record_detail_joins_by_source_track_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_jsonl(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", [_record("r1", "car", "white", "verified")])
            _write_jsonl(run_dir / "08_plate_validation" / "final_track_anpr_results.jsonl", [{"source_id": "s", "track_id": 1, "track_generation": 0, "final_plate_text": "UP81CH4158"}])
            _write_jsonl(run_dir / "06_selected_crops" / "selected_track_crop_sets.jsonl", [{"source_id": "s", "track_id": 1, "track_generation": 0, "selection_status": "primary"}])
            _write_jsonl(run_dir / "04_lifecycle" / "completed_tracks.jsonl", [{"source_id": "s", "track_id": 1, "track_generation": 0, "observation_count": 3}])

            artifacts = load_run_artifacts(run_dir, repo_root=run_dir)
            detail = get_record_detail(artifacts, "r1")

            self.assertEqual("UP81CH4158", detail["plate_validation"]["final_plate_text"])
            self.assertEqual("primary", detail["selected_crops"]["selection_status"])
            self.assertEqual(3, detail["completed_track"]["observation_count"])
            self.assertIn("object_evidence", detail)

    def test_build_object_evidence_prefers_selected_crop_full_frame_matching_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            full_frame = run_dir / "frames" / "frame_10.jpg"
            crop = run_dir / "crops" / "crop_10.jpg"
            full_frame.parent.mkdir(parents=True)
            crop.parent.mkdir(parents=True)
            full_frame.write_bytes(b"frame")
            crop.write_bytes(b"crop")
            record = _record("r1", "car", "white", "verified")
            record["representative_vehicle_crop_path"] = "crops/crop_10.jpg"
            record["representative_frame_index"] = 10
            record["representative_timestamp_sec"] = 1.0
            _write_jsonl(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", [record])
            _write_jsonl(
                run_dir / "06_selected_crops" / "selected_track_crop_sets.jsonl",
                [
                    {
                        "source_id": "s",
                        "track_id": 1,
                        "track_generation": 0,
                        "primary_crops": [
                            {
                                "vehicle_crop_path": "crops/crop_10.jpg",
                                "full_frame_path": "frames/frame_10.jpg",
                                "frame_index": 10,
                                "timestamp_sec": 1.0,
                            }
                        ],
                    }
                ],
            )

            artifacts = load_run_artifacts(run_dir, repo_root=run_dir)
            evidence = build_object_evidence(artifacts.searchable_records[0], artifacts)

            self.assertEqual("frames/frame_10.jpg", evidence["full_frame_path"])
            self.assertEqual("crops/crop_10.jpg", evidence["object_crop_path"])
            self.assertEqual(10, evidence["frame_index"])
            self.assertNotIn("Full frame unavailable", evidence["warnings"])

    def test_build_object_evidence_falls_back_when_full_frame_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            crop = run_dir / "crops" / "crop_10.jpg"
            crop.parent.mkdir(parents=True)
            crop.write_bytes(b"crop")
            record = _record("person1", "person", "blue", "no_plate_detected")
            record["representative_vehicle_crop_path"] = "crops/crop_10.jpg"
            _write_jsonl(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", [record])

            artifacts = load_run_artifacts(run_dir, repo_root=run_dir)
            evidence = build_object_evidence(artifacts.searchable_records[0], artifacts)

            self.assertIsNone(evidence["full_frame_path"])
            self.assertEqual("crops/crop_10.jpg", evidence["object_crop_path"])
            self.assertEqual("person", evidence["object_class"])
            self.assertIn("Full frame unavailable", evidence["warnings"])

    def test_build_object_evidence_reports_frame_mismatch_when_only_unmatched_full_frame_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            full_frame = run_dir / "frames" / "frame_12.jpg"
            crop = run_dir / "crops" / "crop_10.jpg"
            full_frame.parent.mkdir(parents=True)
            crop.parent.mkdir(parents=True)
            full_frame.write_bytes(b"frame")
            crop.write_bytes(b"crop")
            record = _record("r1", "car", "white", "verified")
            record["representative_vehicle_crop_path"] = "crops/crop_10.jpg"
            record["representative_frame_index"] = 10
            _write_jsonl(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", [record])
            _write_jsonl(
                run_dir / "05_crops" / "completed_track_crop_bundles.jsonl",
                [
                    {
                        "source_id": "s",
                        "track_id": 1,
                        "track_generation": 0,
                        "candidates": [{"vehicle_crop_path": "other.jpg", "full_frame_path": "frames/frame_12.jpg", "frame_index": 12}],
                    }
                ],
            )

            artifacts = load_run_artifacts(run_dir, repo_root=run_dir)
            evidence = build_object_evidence(artifacts.searchable_records[0], artifacts)

            self.assertEqual("frames/frame_12.jpg", evidence["full_frame_path"])
            self.assertIn("Full frame/object crop frame mismatch", evidence["warnings"])


def _record(record_id: str, object_class: str, colour: str, status: str, *, plate: str | None = None) -> dict[str, object]:
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
        "plate_confidence": 0.9 if plate else 0.0,
        "first_seen_sec": 1.0,
        "last_seen_sec": 2.0,
        "searchable_tokens": [object_class, colour, status],
        "search_text": f"{object_class} {colour} {status}",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
