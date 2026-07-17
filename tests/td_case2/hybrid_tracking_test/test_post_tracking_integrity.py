from __future__ import annotations

import unittest

from tests.td_case2.hybrid_tracking_test.kcf_drift_detector import DriftDetectionConfig, detect_kcf_drift_segments
from tests.td_case2.hybrid_tracking_test.reconciliation_candidate_index import CandidateIndexConfig, generate_reconciliation_candidates
from tests.td_case2.hybrid_tracking_test.representative_frame_validator import (
    RepresentativeFrameValidationConfig,
    effective_detector_support_score,
    validate_representative_observation,
)
from tests.td_case2.hybrid_tracking_test.track_fragment_reconciliation import MergeScoringConfig, reconcile_track_fragments
from tests.td_case2.hybrid_tracking_test.track_quality import build_track_quality_report, evaluate_track_quality
from tests.td_case2.hybrid_tracking_test.track_timeline_rebuilder import rebuild_track_timelines
from tests.td_case2.hybrid_tracking_test.trajectory_sanitizer import SanitizationConfig, sanitize_track_timeline


class PostTrackingIntegrityTests(unittest.TestCase):
    def test_rebuilds_track_timeline_and_corrects_summary_start(self) -> None:
        raw_track = {
            "track_id": 1,
            "class_name": "person",
            "object_family": "person",
            "start_timestamp_seconds": 2.4,
            "end_timestamp_seconds": 38.2,
            "duration_seconds": 35.8,
            "first_source_frame_index": 789,
            "last_source_frame_index": 1146,
            "trajectory": [],
            "class_votes": {"person": 1.0},
            "detection_hits": 1,
            "propagation_hits": 0,
            "kcf_failures": 0,
            "is_confirmed": True,
        }
        frames = []
        for index, source_frame in enumerate(range(789, 825, 3), start=263):
            frames.append(
                {
                    "source_frame_index": source_frame,
                    "processed_frame_index": index,
                    "timestamp_seconds": round(source_frame / 30.0, 1),
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 0,
                            "class_name": "person",
                            "object_family": "person",
                            "bbox_xyxy": [1235.0, 2.0, 1279.0, 212.0],
                            "bbox_source": "kcf",
                            "status": "propagated_unconfirmed",
                            "kcf_success": True,
                            "frames_since_detection": 8,
                            "seconds_since_detection": 0.9,
                            "last_detection_confidence": 0.6,
                            "reactivation_count": 0,
                            "validation": {"valid": True},
                        }
                    ],
                }
            )
        rebuilt_tracks, report, _ = rebuild_track_timelines(
            [raw_track],
            {"status": "success", "frames": frames},
            {"video_metadata": {"fps": 30.0}},
        )
        rebuilt = rebuilt_tracks[0]
        self.assertEqual(rebuilt["actual_start_timestamp_seconds"], 26.3)
        self.assertIn("summary_start_mismatch", rebuilt["integrity_flags"])
        self.assertEqual(report["timeline_corrections"], 1)

    def test_detects_timestamp_frame_mismatch(self) -> None:
        rebuilt_tracks, _, _ = rebuild_track_timelines(
            [
                {
                    "track_id": 7,
                    "class_name": "car",
                    "object_family": "vehicle",
                    "start_timestamp_seconds": 1.0,
                    "end_timestamp_seconds": 1.0,
                    "duration_seconds": 0.0,
                    "first_source_frame_index": 30,
                    "last_source_frame_index": 30,
                    "trajectory": [],
                    "class_votes": {"car": 1.0},
                }
            ],
            {
                "status": "success",
                "frames": [
                    {
                        "source_frame_index": 30,
                        "processed_frame_index": 10,
                        "timestamp_seconds": 4.0,
                        "tracks": [
                            {
                                "track_id": 7,
                                "class_id": 2,
                                "class_name": "car",
                                "object_family": "vehicle",
                                "bbox_xyxy": [100, 100, 200, 200],
                                "bbox_source": "yolo",
                                "status": "confirmed",
                                "kcf_success": True,
                                "frames_since_detection": 0,
                                "seconds_since_detection": 0.0,
                                "last_detection_confidence": 0.9,
                                "reactivation_count": 0,
                                "validation": {"valid": True},
                            }
                        ],
                    }
                ],
            },
            {"video_metadata": {"fps": 30.0}},
        )
        self.assertIn("timestamp_frame_mismatch", rebuilt_tracks[0]["integrity_flags"])

    def test_detects_repeated_identical_boundary_stuck_kcf_without_false_stationary_yolo(self) -> None:
        track = {
            "track_id": 1,
            "rebuilt_timeline": [
                {"source_frame_index": 780, "processed_frame_index": 260, "timestamp_seconds": 26.0, "bbox_xyxy": [1200, 5, 1279, 210], "bbox_source": "yolo", "kcf_success": True},
                {"source_frame_index": 783, "processed_frame_index": 261, "timestamp_seconds": 26.1, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 786, "processed_frame_index": 262, "timestamp_seconds": 26.2, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 789, "processed_frame_index": 263, "timestamp_seconds": 26.3, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 792, "processed_frame_index": 264, "timestamp_seconds": 26.4, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 795, "processed_frame_index": 265, "timestamp_seconds": 26.5, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 798, "processed_frame_index": 266, "timestamp_seconds": 26.6, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 801, "processed_frame_index": 267, "timestamp_seconds": 26.7, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 804, "processed_frame_index": 268, "timestamp_seconds": 26.8, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 807, "processed_frame_index": 269, "timestamp_seconds": 26.9, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
                {"source_frame_index": 810, "processed_frame_index": 270, "timestamp_seconds": 27.0, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True},
            ],
        }
        drift = detect_kcf_drift_segments(track, frame_width=1280, frame_height=720, config=DriftDetectionConfig())
        self.assertIn("frozen_kcf_box", drift["flags"])
        self.assertIn("boundary_stuck_box", drift["flags"])

        stationary_supported = {
            "track_id": 2,
            "rebuilt_timeline": [
                {"source_frame_index": 0, "processed_frame_index": 0, "timestamp_seconds": 0.0, "bbox_xyxy": [100, 100, 200, 200], "bbox_source": "yolo", "kcf_success": True},
                {"source_frame_index": 3, "processed_frame_index": 1, "timestamp_seconds": 0.1, "bbox_xyxy": [100, 100, 200, 200], "bbox_source": "yolo", "kcf_success": True},
                {"source_frame_index": 6, "processed_frame_index": 2, "timestamp_seconds": 0.2, "bbox_xyxy": [100, 100, 200, 200], "bbox_source": "yolo", "kcf_success": True},
            ],
        }
        stationary_drift = detect_kcf_drift_segments(stationary_supported, frame_width=1280, frame_height=720, config=DriftDetectionConfig())
        self.assertNotIn("frozen_kcf_box", stationary_drift["flags"])

    def test_sanitizer_trims_invalid_kcf_tail_and_downgrades_quality(self) -> None:
        rebuilt = {
            "track_id": 1,
            "class_name": "person",
            "object_family": "person",
            "is_confirmed": True,
            "detection_hits": 1,
            "propagation_hits": 8,
            "kcf_failures": 0,
            "class_votes": {"person": 1.0},
            "actual_end_timestamp_seconds": 26.8,
            "summary_mismatch_fields": ["summary_start_mismatch"],
            "rebuilt_timeline": [
                {"source_frame_index": 780, "processed_frame_index": 260, "timestamp_seconds": 26.0, "bbox_xyxy": [1200, 5, 1279, 210], "bbox_source": "yolo", "kcf_success": True, "seconds_since_detection": 0.0, "last_detection_confidence": 0.8, "validation": {"valid": True}, "object_family": "person"},
                {"source_frame_index": 783, "processed_frame_index": 261, "timestamp_seconds": 26.1, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True, "seconds_since_detection": 0.1, "last_detection_confidence": 0.8, "validation": {"valid": True}, "object_family": "person"},
                {"source_frame_index": 786, "processed_frame_index": 262, "timestamp_seconds": 26.2, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True, "seconds_since_detection": 0.2, "last_detection_confidence": 0.8, "validation": {"valid": True}, "object_family": "person"},
                {"source_frame_index": 789, "processed_frame_index": 263, "timestamp_seconds": 26.3, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True, "seconds_since_detection": 0.9, "last_detection_confidence": 0.8, "validation": {"valid": True}, "object_family": "person"},
                {"source_frame_index": 792, "processed_frame_index": 264, "timestamp_seconds": 26.4, "bbox_xyxy": [1235, 2, 1279, 212], "bbox_source": "kcf", "kcf_success": True, "seconds_since_detection": 1.0, "last_detection_confidence": 0.8, "validation": {"valid": True}, "object_family": "person"},
            ],
        }
        drift = {
            "track_id": 1,
            "flags": ["frozen_kcf_box", "boundary_stuck_box", "long_kcf_only_segment"],
            "segments": [
                {
                    "segment_flags": ["frozen_kcf_box", "boundary_stuck_box", "long_kcf_only_segment"],
                    "observation_indexes": [789, 792],
                }
            ],
        }
        sanitized, events = sanitize_track_timeline(rebuilt, drift, config=SanitizationConfig())
        self.assertEqual(sanitized["sanitized_end_timestamp_seconds"], 26.2)
        self.assertGreater(sanitized["trimmed_kcf_duration_seconds"], 0.0)
        self.assertNotEqual(sanitized["track_integrity_status"], "usable")
        self.assertTrue(any(item["event_type"] == "invalid_kcf_tail_trimmed" for item in events))
        quality = evaluate_track_quality(sanitized, frame_width=1280, frame_height=720)
        self.assertIn(quality["downstream_status"], {"manual_review", "invalid"})

    def test_prevents_frozen_kcf_frame_from_becoming_primary_crop_and_decays_support(self) -> None:
        observation = {
            "bbox_xyxy": [1235, 2, 1279, 212],
            "bbox_source": "kcf",
            "observation_validity": "invalid",
            "drift_segment_flags": ["frozen_kcf_box", "boundary_stuck_box"],
            "seconds_since_detection": 0.9,
            "last_detection_confidence": 0.8,
            "object_family": "person",
        }
        result = validate_representative_observation(
            observation,
            frame_width=1280,
            frame_height=720,
            config=RepresentativeFrameValidationConfig(),
        )
        self.assertFalse(result["identity_crop_eligible"])
        self.assertIn("frozen_kcf_box", result["eligibility_reasons"])
        self.assertLess(
            effective_detector_support_score(0.8, 0.9, config=RepresentativeFrameValidationConfig()),
            effective_detector_support_score(0.8, 0.1, config=RepresentativeFrameValidationConfig()),
        )

    def test_rejects_heavily_clipped_crop_and_allows_valid_short_kcf_bridge(self) -> None:
        clipped = validate_representative_observation(
            {
                "bbox_xyxy": [1270, 2, 1400, 40],
                "bbox_source": "yolo",
                "observation_validity": "valid",
                "drift_segment_flags": [],
                "seconds_since_detection": 0.0,
                "last_detection_confidence": 0.9,
                "object_family": "vehicle",
            },
            frame_width=1280,
            frame_height=720,
            config=RepresentativeFrameValidationConfig(),
        )
        self.assertFalse(clipped["identity_crop_eligible"])
        bridge = validate_representative_observation(
            {
                "bbox_xyxy": [300, 200, 420, 420],
                "bbox_source": "kcf",
                "observation_validity": "supported",
                "drift_segment_flags": [],
                "seconds_since_detection": 0.2,
                "last_detection_confidence": 0.85,
                "object_family": "vehicle",
            },
            frame_width=1280,
            frame_height=720,
            config=RepresentativeFrameValidationConfig(),
        )
        self.assertTrue(bridge["identity_crop_eligible"])

    def test_generates_sequential_and_overlap_candidates_and_merges_valid_pair(self) -> None:
        track_a = {
            "track_id": 10,
            "class_name": "car",
            "final_class": "car",
            "object_family": "vehicle",
            "track_integrity_status": "usable",
            "quality_score": 0.9,
            "quality_level": "high",
            "confirmed": True,
            "sanitized_start_timestamp_seconds": 10.0,
            "sanitized_end_timestamp_seconds": 11.0,
            "sanitized_valid_timeline": [
                {"source_frame_index": 300, "processed_frame_index": 100, "timestamp_seconds": 10.0, "bbox_xyxy": [100, 100, 200, 200], "bbox_source": "yolo"},
                {"source_frame_index": 330, "processed_frame_index": 110, "timestamp_seconds": 11.0, "bbox_xyxy": [120, 100, 220, 200], "bbox_source": "yolo"},
            ],
            "sanitized_timeline": [],
            "source_raw_track_ids": [10],
            "class_votes": {"car": 1.0},
            "entry_boundary": None,
            "exit_boundary": "right",
            "is_confirmed": True,
        }
        track_b = {
            "track_id": 11,
            "class_name": "car",
            "final_class": "car",
            "object_family": "vehicle",
            "track_integrity_status": "usable",
            "quality_score": 0.88,
            "quality_level": "high",
            "confirmed": True,
            "sanitized_start_timestamp_seconds": 11.2,
            "sanitized_end_timestamp_seconds": 12.0,
            "sanitized_valid_timeline": [
                {"source_frame_index": 336, "processed_frame_index": 112, "timestamp_seconds": 11.2, "bbox_xyxy": [125, 100, 225, 200], "bbox_source": "yolo"},
                {"source_frame_index": 360, "processed_frame_index": 120, "timestamp_seconds": 12.0, "bbox_xyxy": [145, 100, 245, 200], "bbox_source": "yolo"},
            ],
            "sanitized_timeline": [],
            "source_raw_track_ids": [11],
            "class_votes": {"car": 1.0},
            "entry_boundary": "right",
            "exit_boundary": None,
            "is_confirmed": True,
        }
        duplicate = {
            **track_b,
            "track_id": 12,
            "sanitized_start_timestamp_seconds": 10.7,
            "sanitized_end_timestamp_seconds": 11.1,
            "sanitized_valid_timeline": [
                {"source_frame_index": 321, "processed_frame_index": 107, "timestamp_seconds": 10.7, "bbox_xyxy": [118, 100, 218, 200], "bbox_source": "yolo"},
                {"source_frame_index": 330, "processed_frame_index": 110, "timestamp_seconds": 11.0, "bbox_xyxy": [120, 100, 220, 200], "bbox_source": "yolo"},
            ],
            "source_raw_track_ids": [12],
        }
        far_simultaneous = {
            **track_b,
            "track_id": 13,
            "sanitized_start_timestamp_seconds": 10.8,
            "sanitized_end_timestamp_seconds": 12.0,
            "sanitized_valid_timeline": [
                {"source_frame_index": 324, "processed_frame_index": 108, "timestamp_seconds": 10.8, "bbox_xyxy": [800, 100, 900, 200], "bbox_source": "yolo"},
                {"source_frame_index": 360, "processed_frame_index": 120, "timestamp_seconds": 12.0, "bbox_xyxy": [820, 100, 920, 200], "bbox_source": "yolo"},
            ],
            "source_raw_track_ids": [13],
        }
        candidates, candidate_report = generate_reconciliation_candidates(
            [track_a, track_b, duplicate, far_simultaneous],
            config=CandidateIndexConfig(),
        )
        self.assertTrue(any(item["candidate_type"] == "sequential_fragment" and item["to_track_id"] == 11 for item in candidates))
        self.assertTrue(any(item["candidate_type"] == "overlap_duplicate" and item["to_track_id"] == 12 for item in candidates))
        self.assertFalse(any(item["to_track_id"] == 13 for item in candidates))
        quality_report = build_track_quality_report([track_a, track_b], frame_width=1280, frame_height=720)
        reconciled, merges, _, report = reconcile_track_fragments(
            [track_a, track_b],
            quality_report,
            camera_id="cam1",
            camera_group="grp",
            camera_timezone="Asia/Kolkata",
            scoring_config=MergeScoringConfig(automatic_merge_score=0.7, possible_merge_score=0.62),
            candidate_records=[item for item in candidates if item["to_track_id"] == 11],
        )
        self.assertEqual(report["accepted_merge_count"], 1)
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["source_raw_track_ids"], [10, 11])
        self.assertTrue(merges)


if __name__ == "__main__":
    unittest.main()
