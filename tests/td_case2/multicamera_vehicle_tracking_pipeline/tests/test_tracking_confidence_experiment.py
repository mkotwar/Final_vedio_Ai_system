from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_config import load_detection_config
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.run_tracking_confidence_experiment import (
    AspectRatioClassRange,
    AspectRatioValidationConfig,
    _build_annotation_label,
    _build_detection_override,
    _build_evidence_index,
    _build_frame_level_report,
    _save_frame_variants,
    _build_track_json,
    _count_class_occurrences,
    calculate_aspect_ratio,
    ensure_unique_run_directory,
    evaluate_aspect_ratio,
    should_gate_new_track,
)
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import ClassObservation, LocalVehicleTrack, TrackObservation
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket


def _observation(frame_number: int, *, class_name: str = "car") -> TrackObservation:
    return TrackObservation(
        camera_code="CAM_002",
        local_track_id=4,
        native_tracker_id=4,
        frame_number=frame_number,
        video_time_seconds=float(frame_number) / 30.0,
        camera_timestamp=datetime(2026, 7, 28, 12, 0, min(frame_number, 59)),
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=(10.0, 10.0, 30.0, 20.0),
        track_uuid="CAM_002:TRACK_4",
        state="active",
        raw_class_name=class_name,
    )


def _class_observation(frame_number: int, *, class_name: str, confidence: float = 0.9) -> ClassObservation:
    return ClassObservation(
        frame_number=frame_number,
        video_time_seconds=float(frame_number) / 30.0,
        camera_timestamp=datetime(2026, 7, 28, 12, 0, min(frame_number, 59)),
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(10.0, 10.0, 30.0, 20.0),
        raw_class_name=class_name,
    )


def _track(*, evidence_path: str | None = None) -> LocalVehicleTrack:
    evidence_package = None
    if evidence_path is not None:
        evidence_package = TrackEvidencePackage(
            run_id="RUN_TEST",
            camera_code="CAM_002",
            local_track_id=4,
            track_uuid="CAM_002:TRACK_4",
            class_name="car",
            output_directory=str(Path(evidence_path).parent),
            candidates={
                "best_overall": EvidenceCandidate(
                    candidate_type="best_overall",
                    frame_number=12,
                    video_time_seconds=0.4,
                    confidence=0.93,
                    original_bbox_xyxy=(10.0, 10.0, 30.0, 20.0),
                    expanded_bbox_xyxy=(8.0, 8.0, 32.0, 22.0),
                    bbox_xyxy=(10.0, 10.0, 30.0, 20.0),
                    crop_width=20,
                    crop_height=10,
                    area=200,
                    sharpness_score=0.8,
                    visibility_score=0.9,
                    centeredness_score=0.85,
                    visible_bbox_ratio=1.0,
                    edge_penalty=0.0,
                    overall_score=0.88,
                    crop_clipped=False,
                    touches_left_edge=False,
                    touches_right_edge=False,
                    touches_top_edge=False,
                    touches_bottom_edge=False,
                    encoded_jpeg=b"",
                    file_path=evidence_path,
                )
            },
        )
    return LocalVehicleTrack(
        track_uuid="CAM_002:TRACK_4",
        camera_code="CAM_002",
        local_track_id=4,
        class_name="car",
        first_frame_number=10,
        last_frame_number=15,
        first_seen_at=datetime(2026, 7, 28, 12, 0, 10),
        last_seen_at=datetime(2026, 7, 28, 12, 0, 15),
        first_video_time_seconds=10 / 30.0,
        last_video_time_seconds=15 / 30.0,
        observation_count=4,
        best_confidence=0.95,
        state="completed",
        observations=[_observation(10), _observation(11), _observation(14), _observation(15)],
        native_tracker_ids_seen=[4, 9],
        reactivation_count=1,
        fragment_relink_count=1,
        maximum_consecutive_missing_frames=2,
        split_from_track_uuid="CAM_002:TRACK_3",
        completion_reason="flush",
        stable_class_name="car",
        provisional_class_name="car",
        class_is_locked=True,
        class_winner_margin=1.25,
        class_observation_count=4,
        class_scores={"car": 3.4, "bus": 0.4},
        class_observation_counts={"car": 3, "bus": 1},
        class_max_confidences={"car": 0.95, "bus": 0.4},
        raw_class_history=[
            _class_observation(10, class_name="car", confidence=0.95),
            _class_observation(11, class_name="car", confidence=0.92),
            _class_observation(14, class_name="bus", confidence=0.40),
            _class_observation(15, class_name="car", confidence=0.88),
        ],
        evidence_package=evidence_package,
    )


class TrackingConfidenceExperimentTests(unittest.TestCase):
    def test_confidence_override_changes_only_detector_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "detection.yaml"
            model_path = Path(tempdir) / "model.pt"
            model_path.write_bytes(b"")
            original = {
                "vehicle_detector": {
                    "model_path": str(model_path),
                    "device": "cpu",
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "image_size": 640,
                    "allowed_classes": ["car", "truck"],
                }
            }
            config_path.write_text(json.dumps(original), encoding="utf-8")
            loaded = load_detection_config(config_path, overrides=_build_detection_override(0.60))

            self.assertEqual(loaded.confidence_threshold, 0.60)
            self.assertEqual(loaded.iou_threshold, 0.45)
            self.assertEqual(loaded.image_size, 640)
            self.assertEqual(config_path.read_text(encoding="utf-8"), json.dumps(original))

    def test_confidence_override_can_enable_class_thresholds_without_mutating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "detection.yaml"
            model_path = Path(tempdir) / "model.pt"
            model_path.write_bytes(b"")
            original = {
                "vehicle_detector": {
                    "model_path": str(model_path),
                    "device": "cpu",
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "image_size": 640,
                    "allowed_classes": ["car", "truck", "bus", "motorcycle", "3wheeler"],
                }
            }
            config_path.write_text(json.dumps(original), encoding="utf-8")
            loaded = load_detection_config(
                config_path,
                overrides=_build_detection_override(
                    0.32,
                    class_confidence_thresholds={
                        "enabled": True,
                        "default": 0.38,
                        "classes": {"car": 0.38, "truck": 0.50, "bus": 0.50, "motorcycle": 0.32, "3wheeler": 0.38},
                    },
                ),
            )

            self.assertEqual(loaded.confidence_threshold, 0.32)
            self.assertTrue(loaded.class_confidence_thresholds.enabled)
            self.assertEqual(loaded.class_confidence_thresholds.classes["motorcycle"], 0.32)
            self.assertEqual(config_path.read_text(encoding="utf-8"), json.dumps(original))

    def test_unique_run_directory_appends_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            first = ensure_unique_run_directory(root, "confidence_025")
            second = ensure_unique_run_directory(root, "confidence_025")

            self.assertEqual(first.name, "confidence_025")
            self.assertEqual(second.name, "confidence_025_02")

    def test_frame_level_report_contains_expected_counts(self) -> None:
        report = _build_frame_level_report(
            frame_number=225,
            timestamp_seconds=7.5,
            raw_rows=[
                {"confidence": 0.32},
                {"confidence": 0.44},
                {"confidence": 0.66},
            ],
            filtered_detection_count=2,
            native_tracker_count=2,
            logical_tracker_count=1,
        )

        self.assertEqual(report["raw_detection_count"], 3)
        self.assertEqual(report["detections_below_035"], 1)
        self.assertEqual(report["detections_below_050"], 2)
        self.assertEqual(report["logical_rows"], 1)

    def test_track_json_contains_class_history_and_persistence_diagnostics(self) -> None:
        payload = _build_track_json(_track())

        self.assertEqual(payload["class_observation_counts"]["car"], 3)
        self.assertEqual(payload["class_persistence_summary"][0]["summary"], "CAR observed in 3/4 frames")
        self.assertEqual(payload["number_of_native_tracker_ids"], 2)
        self.assertEqual(payload["number_of_missed_frames"], 2)
        self.assertEqual(payload["maximum_consecutive_missing_frames"], 2)
        self.assertEqual(payload["longest_consecutive_observation_run"], 2)
        self.assertEqual(payload["number_of_fragment_relinks"], 1)
        self.assertEqual(payload["number_of_logical_splits"], 1)

    def test_class_occurrence_counts_group_by_key(self) -> None:
        counts = _count_class_occurrences(
            [
                {"raw_class_name": "car"},
                {"raw_class_name": "CAR"},
                {"raw_class_name": "bus"},
                {"raw_class_name": ""},
            ],
            key="raw_class_name",
        )

        self.assertEqual(counts, {"car": 2, "bus": 1})

    def test_annotation_label_distinguishes_raw_native_and_logical_context(self) -> None:
        raw_label = _build_annotation_label(
            camera_code="CAM_002",
            frame_number=225,
            timestamp_seconds=7.5,
            class_name="CAR",
            confidence=0.32,
        )
        logical_label = _build_annotation_label(
            camera_code="CAM_002",
            frame_number=225,
            timestamp_seconds=7.5,
            class_name="CAR",
            confidence=0.32,
            native_tracker_id=4,
            logical_track_id="TRACK_4",
        )

        self.assertIn("7.50s", raw_label)
        self.assertNotIn("native", raw_label)
        self.assertIn("native 4", logical_label)
        self.assertIn("logical TRACK_4", logical_label)

    def test_evidence_index_links_to_track_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            run_dir = Path(tempdir)
            image_path = run_dir / "artifacts" / "track_4" / "best_overall.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")

            _build_evidence_index(run_dir, [_track(evidence_path=str(image_path))])
            payload = (run_dir / "evidence_index.html").read_text(encoding="utf-8")

            self.assertIn("CAM_002:TRACK_4", payload)
            self.assertIn("artifacts/track_4/best_overall.jpg", payload)

    def test_save_frame_variants_writes_flat_sample_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            run_dir = Path(tempdir)
            frame_packet = FramePacket(
                camera_code="CAM_002",
                camera_name="CAM_002",
                source_path=Path("camera.mp4"),
                frame_number=220,
                source_fps=30.0,
                source_frame_count=1285,
                video_time_seconds=220 / 30.0,
                camera_timestamp=datetime(2026, 7, 28, 12, 0, 7),
                frame=__import__("numpy").zeros((40, 60, 3), dtype=__import__("numpy").uint8),
            )
            _save_frame_variants(
                frame_packet=frame_packet,
                raw_rows=[{"bbox_xyxy": [1, 2, 10, 12], "normalized_class_name": "car", "raw_class_name": "car", "confidence": 0.7}],
                native_rows=[{"bbox_xyxy": [1, 2, 10, 12], "class_name": "car", "confidence": 0.7, "native_tracker_id": 3}],
                logical_rows=[{"bbox_xyxy": [1, 2, 10, 12], "class_name": "car", "confidence": 0.7, "native_tracker_id": 3, "logical_track_id": "TRACK_3"}],
                frame_dir=run_dir / "diagnostic_frames" / "frame_000220",
                sample_output_root=run_dir / "sample_frames",
                sample_index=1,
            )

            sample_root = run_dir / "sample_frames" / "CAM_002"
            self.assertTrue((sample_root / "yolo_detections" / "sample_000001.jpg").exists())
            self.assertTrue((sample_root / "yolo_detections" / "sample_000001.json").exists())
            self.assertTrue((sample_root / "native_tracker_output" / "sample_000001.jpg").exists())
            self.assertTrue((sample_root / "logical_track_output" / "sample_000001.jpg").exists())
            self.assertTrue((sample_root / "raw_full_frames" / "sample_000001.jpg").exists())

    def test_aspect_ratio_calculation_and_zero_height_rejection(self) -> None:
        self.assertEqual(calculate_aspect_ratio(20.0, 10.0), 2.0)
        with self.assertRaises(ValueError):
            calculate_aspect_ratio(20.0, 0.0)

    def test_per_class_range_lookup_and_missing_range_handling(self) -> None:
        config = AspectRatioValidationConfig(classes={"car": AspectRatioClassRange(min_ratio=1.2, max_ratio=2.4)})

        evaluation = evaluate_aspect_ratio(
            class_name="truck",
            width=20.0,
            height=10.0,
            touches_edge=False,
            partial_visibility_ratio=1.0,
            config=config,
        )

        self.assertEqual(config.range_for("car"), AspectRatioClassRange(min_ratio=1.2, max_ratio=2.4))
        self.assertIsNone(config.range_for("truck"))
        self.assertEqual(evaluation.status, "RANGE_NOT_CONFIGURED")

    def test_edge_tolerance_and_report_only_mode_keep_detections(self) -> None:
        config = AspectRatioValidationConfig(
            classes={"car": AspectRatioClassRange(min_ratio=1.2, max_ratio=1.5)},
            action="report_only",
        )
        evaluation = evaluate_aspect_ratio(
            class_name="car",
            width=3.0,
            height=10.0,
            touches_edge=True,
            partial_visibility_ratio=0.8,
            config=config,
        )

        self.assertEqual(evaluation.status, "EDGE_TOLERATED")
        self.assertFalse(
            should_gate_new_track(
                action=config.action,
                detection_confidence=0.20,
                aspect_ratio_status=evaluation.status,
                is_new_track=True,
                strong_continuation=False,
            )
        )

    def test_quality_penalty_mode_keeps_detections_and_new_track_gate_respects_continuation(self) -> None:
        self.assertFalse(
            should_gate_new_track(
                action="quality_penalty",
                detection_confidence=0.20,
                aspect_ratio_status="BELOW_MINIMUM",
                is_new_track=True,
                strong_continuation=False,
            )
        )
        self.assertFalse(
            should_gate_new_track(
                action="new_track_gate",
                detection_confidence=0.20,
                aspect_ratio_status="BELOW_MINIMUM",
                is_new_track=True,
                strong_continuation=True,
            )
        )
        self.assertTrue(
            should_gate_new_track(
                action="new_track_gate",
                detection_confidence=0.20,
                aspect_ratio_status="ABOVE_MAXIMUM",
                is_new_track=True,
                strong_continuation=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
