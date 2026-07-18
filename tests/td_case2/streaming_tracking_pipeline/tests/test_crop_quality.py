from __future__ import annotations

import unittest

import numpy as np

from tests.td_case2.streaming_tracking_pipeline.config import CropCollectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_quality import compute_crop_quality, extract_crop, score_preliminary_quality
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox


class CropQualityTest(unittest.TestCase):
    def test_extract_crop_clips_padding_and_reports_completeness(self) -> None:
        frame = np.zeros((40, 50, 3), dtype=np.uint8)
        config = CropCollectionConfig(padding_ratio=0.25, minimum_crop_width=4, minimum_crop_height=4)

        extracted = extract_crop(frame, BoundingBox(0, 0, 20, 20), frame_width=50, frame_height=40, config=config)

        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted.crop_bbox.to_xyxy(), [0.0, 0.0, 25, 25])
        self.assertTrue(extracted.padding_clipped)
        self.assertLess(extracted.crop_completeness, 1.0)

    def test_quality_metrics_are_bounded_and_json_ready(self) -> None:
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[10:40, 20:55, :] = 210
        config = CropCollectionConfig(padding_ratio=0.0)
        extracted = extract_crop(frame, BoundingBox(20, 10, 55, 40), frame_width=80, frame_height=60, config=config)
        assert extracted is not None

        quality = compute_crop_quality(
            crop=extracted.crop,
            source_bbox=BoundingBox(20, 10, 55, 40),
            extracted=extracted,
            frame_width=80,
            frame_height=60,
            detection_confidence=0.9,
            config=config,
        )

        self.assertGreater(quality.bbox_area_ratio, 0.0)
        self.assertGreaterEqual(quality.preliminary_score or 0.0, 0.0)
        self.assertLessEqual(quality.preliminary_score or 0.0, 1.0)
        self.assertEqual(quality.crop_width, 35)

    def test_score_penalizes_edge_touching(self) -> None:
        config = CropCollectionConfig()
        clean = score_preliminary_quality(
            detection_confidence=0.9,
            bbox_area_ratio=0.2,
            sharpness=100.0,
            brightness=0.5,
            contrast=30.0,
            edge_touching=False,
            crop_completeness=1.0,
            config=config,
        )
        edge = score_preliminary_quality(
            detection_confidence=0.9,
            bbox_area_ratio=0.2,
            sharpness=100.0,
            brightness=0.5,
            contrast=30.0,
            edge_touching=True,
            crop_completeness=1.0,
            config=config,
        )

        self.assertLess(edge, clean)


if __name__ == "__main__":
    unittest.main()
