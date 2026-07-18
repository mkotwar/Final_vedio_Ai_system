from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.plate_diagnostic_metrics import build_plate_diagnostic_metrics
from tests.td_case2.streaming_tracking_pipeline.plate_diagnostics import (
    PlateAttemptStatus,
    PlateBoxDisposition,
    PlateDiagnosticAttempt,
    RawPlateBoxDiagnostic,
    TrackPlateDiagnosticResult,
)


class PlateDiagnosticMetricTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        metrics = build_plate_diagnostic_metrics([])
        self.assertEqual(metrics["tracks_processed"], 0)
        self.assertEqual(metrics["raw_detector_boxes"], 0)

    def test_counts_rejections_and_groupings(self) -> None:
        raw = RawPlateBoxDiagnostic(
            source_id="cam",
            track_id=1,
            track_generation=0,
            source_track_id=None,
            vehicle_crop_role="primary",
            vehicle_crop_rank=1,
            source_frame_index=1,
            timestamp_sec=0.1,
            attempt_number=1,
            diagnostic_threshold=0.05,
            raw_box_index=1,
            raw_bbox_xyxy=[1, 1, 2, 2],
            clipped_bbox=None,
            raw_confidence=0.04,
            raw_class_id=0,
            raw_class_name="license_plate",
            width=None,
            height=None,
            area=None,
            disposition=PlateBoxDisposition.BELOW_THRESHOLD,
            rejection_reason="below_diagnostic_acceptance_threshold",
            plate_crop_path=None,
        )
        attempt = PlateDiagnosticAttempt(
            source_id="cam",
            track_id=1,
            track_generation=0,
            source_track_id=None,
            attempt_number=1,
            vehicle_crop_role="primary",
            vehicle_crop_rank=1,
            vehicle_crop_path="crop.jpg",
            source_frame_index=1,
            timestamp_sec=0.1,
            configured_detection_threshold=0.2,
            diagnostic_thresholds_used=[0.05],
            raw_box_count=1,
            below_threshold_box_count=1,
            invalid_geometry_count=0,
            empty_after_clipping_count=0,
            too_small_count=0,
            accepted_plate_count=0,
            raw_boxes=[raw],
            accepted_candidates=[],
            ocr_results=[],
            attempt_status=PlateAttemptStatus.ALL_BOXES_BELOW_THRESHOLD,
            stop_reason=None,
            runtime_sec=0.01,
            error_message=None,
            metadata={"vehicle_crop": {"selection_score": 0.6}, "vehicle_crop_size_bucket": "small"},
        )
        result = TrackPlateDiagnosticResult("cam", 1, 0, None, "car", [attempt], None, None, None, "no_plate_candidate", ["below"], True)
        metrics = build_plate_diagnostic_metrics([result])
        self.assertEqual(metrics["boxes_below_threshold"], 1)
        self.assertEqual(metrics["raw_boxes_by_threshold"]["0.050"], 1)
        self.assertEqual(metrics["by_crop_role"]["primary"]["all_boxes_below_threshold"], 1)
        self.assertEqual(metrics["tracks_exhausting_all_crops"], 1)


if __name__ == "__main__":
    unittest.main()
