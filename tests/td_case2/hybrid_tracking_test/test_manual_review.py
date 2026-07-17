from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.td_case2.hybrid_tracking_test.manual_review_data import ManualReviewRepository


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    post_dir = run_dir / "hybrid_tracking_test" / "post_tracking_v2"
    _write_json(
        run_dir / "hybrid_tracking_test" / "04c_hybrid_config.json",
        {
            "run_dir": str(run_dir),
            "video_path": str(tmp_path / "missing.mp4"),
        },
    )
    _write_json(
        run_dir / "hybrid_tracking_test" / "04c_hybrid_tracking_report.json",
        {
            "video_metadata": {
                "input_video_path": str(tmp_path / "missing.mp4"),
                "video_name": "missing.mp4",
                "fps": 30.0,
                "frame_count": 300,
                "duration_seconds": 10.0,
                "width": 1280,
                "height": 720,
            }
        },
    )
    packages = [
        {
            "camera_id": "test_cam_01",
            "local_object_id": 1,
            "object_family": "vehicle",
            "final_class": "car",
            "source_raw_track_ids": [101],
            "start_timestamp_seconds": 1.0,
            "end_timestamp_seconds": 3.0,
            "duration_seconds": 2.0,
            "first_source_frame_index": 30,
            "last_source_frame_index": 90,
            "quality_level": "medium",
            "quality_score": 0.9,
            "downstream_status": "manual_review",
            "warnings": ["short_track"],
            "manual_review_reasons": ["short_track"],
            "reconciliation_status": "single_segment",
            "entry_boundary": "left",
            "exit_boundary": "top",
            "motion_direction": "bottom_to_top",
            "track_integrity_status": "usable",
        },
        {
            "camera_id": "test_cam_01",
            "local_object_id": 2,
            "object_family": "vehicle",
            "final_class": "truck",
            "source_raw_track_ids": [102, 103],
            "start_timestamp_seconds": 3.0,
            "end_timestamp_seconds": 6.0,
            "duration_seconds": 3.0,
            "first_source_frame_index": 91,
            "last_source_frame_index": 180,
            "quality_level": "low",
            "quality_score": 0.7,
            "downstream_status": "manual_review",
            "warnings": ["class_instability"],
            "manual_review_reasons": ["class_instability"],
            "reconciliation_status": "merged",
            "entry_boundary": "left",
            "exit_boundary": "top",
            "motion_direction": "bottom_to_top",
            "track_integrity_status": "usable",
        },
    ]
    _write_json(post_dir / "05v2_local_identity_packages.json", {"status": "success", "packages": packages})
    _write_json(post_dir / "05v2_local_identity_packages_flat.json", {"status": "success", "rows": []})
    _write_json(
        post_dir / "05v2_local_identity_package_report.json",
        {
            "status": "success",
            "total_packages": 2,
            "ready_packages": 0,
            "fallback_packages": 0,
            "manual_review_packages": 2,
            "rejected_packages": 0,
            "vehicle_packages": 2,
            "person_packages": 0,
        },
    )
    _write_json(
        post_dir / "05v2_representative_frames.json",
        {
            "status": "success",
            "objects": [
                {
                    "local_object_id": 1,
                    "representative_frames": {
                        "primary": {"crop_path": str(tmp_path / "crop1.jpg"), "full_frame_path": str(tmp_path / "full1.jpg")},
                        "alternatives": [],
                    },
                },
                {
                    "local_object_id": 2,
                    "representative_frames": {
                        "primary": {"crop_path": str(tmp_path / "crop2.jpg"), "full_frame_path": str(tmp_path / "full2.jpg")},
                        "alternatives": [],
                    },
                },
            ],
        },
    )
    _write_json(
        post_dir / "04d2_reconciled_tracks.json",
        {
            "status": "success",
            "tracks": [
                {
                    "local_object_id": 1,
                    "source_raw_track_ids": [101],
                    "merge_evidence": [],
                },
                {
                    "local_object_id": 2,
                    "source_raw_track_ids": [102, 103],
                    "merge_evidence": [{"from_track_id": 102, "to_track_id": 103, "merge_score": 0.8}],
                },
            ],
        },
    )
    _write_json(
        post_dir / "04d2_track_merge_events.json",
        {
            "status": "success",
            "events": [
                {
                    "local_object_id": 2,
                    "source_track_ids": [102, 103],
                    "merge_evidence": [{"from_track_id": 102, "to_track_id": 103, "merge_score": 0.8}],
                }
            ],
        },
    )
    _write_json(
        post_dir / "04d2_reconciliation_candidates.json",
        {
            "status": "success",
            "candidates": [
                {"from_track_id": 101, "to_track_id": 104, "candidate_type": "sequential_fragment", "object_family": "vehicle"}
            ],
        },
    )
    _write_json(
        post_dir / "04d2_rejected_merge_candidates.json",
        {"status": "success", "candidates": []},
    )
    _write_json(
        post_dir / "04d2_track_quality_report.json",
        {
            "status": "success",
            "tracks": [
                {"track_id": 101, "object_family": "vehicle", "final_class": "car", "integrity_status": "usable", "quality_level": "medium", "quality_score": 0.9},
                {"track_id": 102, "object_family": "vehicle", "final_class": "truck", "integrity_status": "usable", "quality_level": "low", "quality_score": 0.7},
                {"track_id": 103, "object_family": "vehicle", "final_class": "truck", "integrity_status": "usable", "quality_level": "low", "quality_score": 0.7},
                {"track_id": 104, "object_family": "vehicle", "final_class": "car", "integrity_status": "usable", "quality_level": "medium", "quality_score": 0.8},
            ],
        },
    )
    return run_dir, post_dir


def test_loading_local_identity_packages(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    assert len(repo.packages) == 2
    assert repo.available_input_files()


def test_saving_and_updating_review(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    repo.save_object_review(
        {
            "local_object_id": 1,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0001",
            "object_review_status": "correct_single_object",
            "crop_review_status": "primary_crop_good",
            "timeline_review_status": "timeline_correct",
            "class_review_status": "class_correct",
            "downstream_decision": "ready",
            "manual_class": "car",
        }
    )
    repo.save_object_review(
        {
            "local_object_id": 1,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0099",
            "object_review_status": "wrong_class",
            "crop_review_status": "alternative_crop_better",
            "timeline_review_status": "start_too_early",
            "class_review_status": "should_be_truck",
            "downstream_decision": "fallback",
            "manual_class": "truck",
        }
    )
    payload = json.loads(repo.paths.object_reviews_json.read_text(encoding="utf-8"))
    assert len(payload["reviews"]) == 1
    assert payload["reviews"][0]["manual_real_object_id"] == "vehicle_0099"


def test_autosave_behavior_and_restart(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    repo.save_object_review(
        {
            "local_object_id": 2,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0002",
            "object_review_status": "fragmented_object",
            "crop_review_status": "crop_uncertain",
            "timeline_review_status": "timeline_uncertain",
            "class_review_status": "class_uncertain",
            "downstream_decision": "manual_review",
            "manual_class": "truck",
            "same_real_object_as_local_object_ids": [1],
        }
    )
    assert repo.paths.progress_json.exists()
    restarted = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    assert restarted.get_object_review(2) is not None
    assert restarted.get_object_review(2).same_real_object_as_local_object_ids == [1]


def test_duplicate_mapping_and_ground_truth_grouping(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    repo.save_object_review(
        {
            "local_object_id": 1,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0001",
            "object_review_status": "correct_single_object",
            "crop_review_status": "primary_crop_good",
            "timeline_review_status": "timeline_correct",
            "class_review_status": "class_correct",
            "downstream_decision": "ready",
            "manual_class": "car",
        }
    )
    repo.save_object_review(
        {
            "local_object_id": 2,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0001",
            "object_review_status": "duplicate_track",
            "crop_review_status": "alternative_crop_better",
            "timeline_review_status": "timeline_correct",
            "class_review_status": "should_be_car",
            "downstream_decision": "fallback",
            "manual_class": "car",
            "same_real_object_as_local_object_ids": [1],
        }
    )
    ground_truth = json.loads(repo.paths.ground_truth_json.read_text(encoding="utf-8"))
    assert len(ground_truth["objects"]) == 1
    assert ground_truth["objects"][0]["matched_local_object_ids"] == [1, 2]


def test_merge_and_possible_merge_reviews(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    repo.save_merge_review(
        {
            "local_object_id": 2,
            "source_track_ids": [102, 103],
            "decision": "merge_incorrect",
            "incorrect_reason": "different appearance",
        }
    )
    repo.save_possible_merge_review(
        {
            "candidate_key": "101->104",
            "from_track_id": 101,
            "to_track_id": 104,
            "decision": "accept_merge",
        }
    )
    merge_payload = json.loads(repo.paths.merge_reviews_json.read_text(encoding="utf-8"))
    possible_payload = json.loads(repo.paths.possible_merge_reviews_json.read_text(encoding="utf-8"))
    assert merge_payload["reviews"][0]["decision"] == "merge_incorrect"
    assert possible_payload["reviews"][0]["decision"] == "accept_merge"


def test_csv_export_summary_and_progress(tmp_path: Path) -> None:
    run_dir, post_dir = _build_fixture(tmp_path)
    repo = ManualReviewRepository(run_dir=run_dir, post_tracking_dir=post_dir)
    repo.save_object_review(
        {
            "local_object_id": 1,
            "camera_id": "test_cam_01",
            "manual_real_object_id": "vehicle_0001",
            "object_review_status": "false_detection",
            "crop_review_status": "all_crops_bad",
            "timeline_review_status": "timeline_uncertain",
            "class_review_status": "class_uncertain",
            "downstream_decision": "reject",
            "manual_class": "car",
            "false_detection_reason": "shadow",
        }
    )
    with repo.paths.object_reviews_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["object_review_status"] == "false_detection"
    summary = json.loads(repo.paths.summary_json.read_text(encoding="utf-8"))
    progress = json.loads(repo.paths.progress_json.read_text(encoding="utf-8"))
    assert summary["false_detection_count"] == 1
    assert progress["reviewed_objects"] == 1
