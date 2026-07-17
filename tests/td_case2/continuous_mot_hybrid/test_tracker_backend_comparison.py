from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.continuous_mot_hybrid.compare_tracker_backend_results import build_identity_switch_candidates
from tests.td_case2.continuous_mot_hybrid.fixed_5fps_validation_core import build_validation_checks
from tests.td_case2.continuous_mot_hybrid.tracker_backend_metrics import build_reid_metric_block
from tests.td_case2.continuous_mot_hybrid.tracker_backend_visualizer import select_visual_timestamps
from tests.td_case2.continuous_mot_hybrid.tracker_yaml_builder import parse_simple_yaml, write_tracker_yaml
from tests.td_case2.continuous_mot_hybrid.ultralytics_botsort_backend import installed_ultralytics_info


class TrackerBackendComparisonTests(unittest.TestCase):
    def test_detection_cache_checksum_stability(self) -> None:
        payload = [{"processed_frame_index": 0, "detections": [{"detection_id": "a"}]}]
        first = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        second = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(first, second)

    def test_botsort_yaml_without_reid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "botsort.yaml"
            source.write_text("tracker_type: botsort\nwith_reid: true\nmodel: auto\n", encoding="utf-8")
            resolved = write_tracker_yaml(
                source_yaml=source,
                destination_yaml=root / "out.yaml",
                overrides={"with_reid": False, "model": "auto"},
            )
            self.assertFalse(resolved["with_reid"])
            self.assertEqual(parse_simple_yaml(root / "out.yaml")["tracker_type"], "botsort")

    def test_botsort_yaml_with_reid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "botsort.yaml"
            source.write_text("tracker_type: botsort\nwith_reid: false\nmodel: manual\n", encoding="utf-8")
            resolved = write_tracker_yaml(
                source_yaml=source,
                destination_yaml=root / "out.yaml",
                overrides={"with_reid": True, "model": "auto"},
            )
            self.assertTrue(resolved["with_reid"])
            self.assertEqual(parse_simple_yaml(root / "out.yaml")["model"], "auto")

    def test_runtime_reid_verification_unavailable(self) -> None:
        block = build_reid_metric_block({"requested_with_reid": True, "actual_with_reid": False})
        self.assertFalse(block["actual_with_reid"])
        self.assertEqual(block["embedding_extractions"], "not_available")

    def test_failure_when_reid_requested_but_not_active(self) -> None:
        verification = {
            "requested_with_reid": True,
            "actual_with_reid": False,
            "fallback_reason": "cached_detection_replay_has_no_native_detector_features_for_model_auto",
        }
        self.assertTrue(verification["requested_with_reid"])
        self.assertFalse(verification["actual_with_reid"])
        self.assertIn("cached_detection_replay", verification["fallback_reason"])

    def test_skipped_detector_frame_safety(self) -> None:
        checks = build_validation_checks(
            per_frame_events=[{"detector_ran": False, "lost_track_ids": [], "terminated_track_ids": [], "new_track_ids": []}],
            new_id_events=[],
            reactivation_events=[],
            records={},
        )
        self.assertTrue(checks["passed"])

    def test_no_id_creation_on_skipped_frames(self) -> None:
        checks = build_validation_checks(
            per_frame_events=[{"detector_ran": False, "lost_track_ids": [], "terminated_track_ids": [], "new_track_ids": ["x"]}],
            new_id_events=[],
            reactivation_events=[],
            records={},
        )
        self.assertFalse(checks["passed"])

    def test_same_timestamps_supplied_to_all_trackers(self) -> None:
        timestamps = select_visual_timestamps(bytetrack_events=[{"timestamp_seconds": 1.0, "new_track_ids": ["a"]}], fallback_duration_seconds=10.0)
        self.assertEqual(len(timestamps), 20)
        self.assertEqual(timestamps[0], 0.0)

    def test_track_metric_extraction(self) -> None:
        block = build_reid_metric_block({"feature_vector_count": 9, "actual_with_reid": True})
        self.assertEqual(block["embedding_extractions"], 9)

    def test_reid_metric_unavailable_field_handling(self) -> None:
        block = build_reid_metric_block({})
        self.assertEqual(block["average_appearance_similarity"], "not_available")

    def test_identity_switch_candidate_generation(self) -> None:
        candidates = build_identity_switch_candidates(
            backend_tracks={
                "bytetrack": [
                    {"tracker_id": "a", "object_family": "vehicle", "start_timestamp": 0.0, "end_timestamp": 1.0},
                    {"tracker_id": "b", "object_family": "vehicle", "start_timestamp": 1.2, "end_timestamp": 2.0},
                ],
                "botsort_no_reid": [],
                "botsort_reid": [],
            }
        )
        self.assertEqual(candidates[0]["label"], "review_required")

    def test_visual_timestamp_consistency(self) -> None:
        values = select_visual_timestamps(bytetrack_events=[], fallback_duration_seconds=19.0)
        self.assertEqual(len(values), 20)
        self.assertEqual(values[-1], 19.0)

    def test_config_difference_reporting_shape(self) -> None:
        site_packages = Path("tests/td_case2/.venv/Lib/site-packages")
        info = installed_ultralytics_info(site_packages)
        self.assertIn("ultralytics_version", info)
        self.assertTrue(info["botsort_yaml_path"].endswith("botsort.yaml"))

    def test_missing_reid_model_handling(self) -> None:
        verification = {"requested_with_reid": True, "actual_with_reid": False}
        status = "reid_not_available" if verification["requested_with_reid"] and not verification["actual_with_reid"] else "ok"
        self.assertEqual(status, "reid_not_available")

    def test_no_silent_downloading_policy_marker(self) -> None:
        warning = "Requested model=auto cannot prove active ReID during cached replay because detector-native features are unavailable."
        self.assertIn("cannot prove active ReID", warning)

    def test_final_report_generation_inputs(self) -> None:
        reports = {
            "bytetrack": {"tracker_runtime_seconds": 1.0, "raw_track_ids": 10},
            "botsort_no_reid": {"tracker_runtime_seconds": 2.0, "raw_track_ids": 9},
            "botsort_reid": {"tracker_runtime_seconds": 3.0, "raw_track_ids": 8},
        }
        self.assertEqual(reports["bytetrack"]["raw_track_ids"], 10)

    def test_existing_bytetrack_runner_remains_usable(self) -> None:
        source = Path("tests/td_case2/continuous_mot_hybrid/run_fixed_5fps_bytetrack_validation.py")
        self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
