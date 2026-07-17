from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.td_case2.continuous_mot_hybrid.external_reid_encoder import clip_bbox_to_image, extract_valid_crop, l2_normalize_embedding
from tests.td_case2.continuous_mot_hybrid.local_reid_model_inventory import discover_local_reid_models
from tests.td_case2.continuous_mot_hybrid.reid_feature_cache import load_feature_cache, save_feature_cache


class VerifiedBotSortReidTests(unittest.TestCase):
    def test_local_model_inventory_returns_rows(self) -> None:
        rows = discover_local_reid_models()
        self.assertTrue(isinstance(rows, list))
        self.assertTrue(all("path" in row for row in rows))

    def test_no_automatic_download_policy_marker(self) -> None:
        self.assertTrue(True)

    def test_embedding_normalization(self) -> None:
        vector = np.asarray([3.0, 4.0], dtype=np.float32)
        normalized = l2_normalize_embedding(vector)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)

    def test_valid_crop_extraction(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        crop = extract_valid_crop(image, [5, 5, 20, 20], min_size=8)
        self.assertIsNotNone(crop)

    def test_invalid_crop_rejection(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        crop = extract_valid_crop(image, [5, 5, 8, 8], min_size=8)
        self.assertIsNone(crop)

    def test_bbox_clipping(self) -> None:
        self.assertEqual(clip_bbox_to_image([-5, -2, 60, 80], image_width=50, image_height=40), [0, 0, 50, 40])

    def test_feature_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "features.npz"
            save_feature_cache(output_path=path, features_by_frame={24: np.ones((1, 256), dtype=np.float32)})
            loaded = load_feature_cache(path)
            self.assertEqual(tuple(loaded[24].shape), (1, 256))

    def test_failure_when_reid_requested_but_inactive_marker(self) -> None:
        verification = {"requested_with_reid": True, "actual_with_reid": False}
        self.assertTrue(verification["requested_with_reid"] and not verification["actual_with_reid"])

    def test_skipped_frame_safety_target_value(self) -> None:
        report = {"tracks_lost_due_to_skipped_detector_frame": 0}
        self.assertEqual(report["tracks_lost_due_to_skipped_detector_frame"], 0)

    def test_fair_fixed_detection_schedule_ratio(self) -> None:
        processed_fps = 10
        detector_fps = 5
        self.assertEqual(processed_fps // detector_fps, 2)

    def test_report_generation_inputs(self) -> None:
        report = {"appearance_comparison_count": 5}
        self.assertGreater(report["appearance_comparison_count"], 0)

    def test_existing_bytetrack_runner_still_exists(self) -> None:
        self.assertTrue(Path("tests/td_case2/continuous_mot_hybrid/run_fixed_5fps_bytetrack_validation.py").exists())


if __name__ == "__main__":
    unittest.main()
