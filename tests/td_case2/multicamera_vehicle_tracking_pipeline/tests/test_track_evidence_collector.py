from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_config import EvidenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.track_evidence_collector import TrackEvidenceCollector
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


def _packet(frame_number: int, frame: np.ndarray) -> DetectionPacket:
    return DetectionPacket(
        camera_code="CAM_001",
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 23, 12, 0, min(frame_number, 59)),
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        detections=[],
        inference_time_ms=1.0,
        detector_model="fake.pt",
        detector_device="cpu",
        frame=frame,
    )


def _observation(frame_number: int, *, confidence: float = 0.9, bbox_xyxy=(10.0, 10.0, 80.0, 80.0)) -> TrackObservation:
    return TrackObservation(
        camera_code="CAM_001",
        local_track_id=1,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 23, 12, 0, min(frame_number, 59)),
        class_name="car",
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        track_uuid="RUN_TEST:CAM_001:TRACK_1",
        state="active",
    )


def _track(observation_count: int) -> LocalVehicleTrack:
    observations = [_observation(index) for index in range(observation_count)]
    return LocalVehicleTrack(
        track_uuid="RUN_TEST:CAM_001:TRACK_1",
        camera_code="CAM_001",
        local_track_id=1,
        class_name="car",
        first_frame_number=0,
        last_frame_number=observation_count - 1,
        first_seen_at=observations[0].camera_timestamp,
        last_seen_at=observations[-1].camera_timestamp,
        first_video_time_seconds=0.0,
        last_video_time_seconds=float(observation_count - 1),
        observation_count=observation_count,
        best_confidence=max(item.confidence for item in observations),
        state="completed",
        observations=observations,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
    )


class TrackEvidenceCollectorTests(unittest.TestCase):
    def test_collects_bounded_named_candidates(self) -> None:
        collector = TrackEvidenceCollector(EvidenceConfig(enabled=True, minimum_crop_width=20, minimum_crop_height=20), run_id="RUN_TEST")
        frame = np.full((100, 100, 3), 127, dtype=np.uint8)
        for frame_number, confidence in ((0, 0.6), (1, 0.9), (2, 0.7)):
            collector.update(_packet(frame_number, frame.copy()), [_observation(frame_number, confidence=confidence)])
        evidence = collector.finalize_track(_track(3))
        self.assertIsNotNone(evidence)
        self.assertIn("first", evidence.candidates)
        self.assertIn("last", evidence.candidates)
        self.assertIn("highest_confidence", evidence.candidates)
        self.assertLessEqual(len(evidence.candidates), 7)
        self.assertEqual(evidence.candidates["highest_confidence"].frame_number, 1)

    def test_rejects_too_small_crops(self) -> None:
        collector = TrackEvidenceCollector(EvidenceConfig(enabled=True, minimum_crop_width=50, minimum_crop_height=50), run_id="RUN_TEST")
        frame = np.full((60, 60, 3), 127, dtype=np.uint8)
        collector.update(_packet(0, frame), [_observation(0, bbox_xyxy=(10.0, 10.0, 30.0, 30.0))])
        self.assertIsNone(collector.finalize_track(_track(1)))

    def test_saves_final_selected_crops_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrackEvidenceCollector(
                EvidenceConfig(
                    enabled=True,
                    output_root=tmpdir,
                    minimum_crop_width=20,
                    minimum_crop_height=20,
                    save_final_selected_crops=True,
                ),
                run_id="RUN_TEST",
            )
            frame = np.full((100, 100, 3), 127, dtype=np.uint8)
            collector.update(_packet(0, frame), [_observation(0)])
            evidence = collector.finalize_track(_track(1))
            self.assertIsNotNone(evidence)
            self.assertIsNotNone(evidence.output_directory)
            self.assertTrue(Path(evidence.output_directory).exists())
            self.assertTrue((Path(evidence.output_directory) / "full_frame.jpg").exists())
            self.assertTrue((Path(evidence.output_directory) / "annotated_full_frame.jpg").exists())
            self.assertTrue((Path(evidence.output_directory) / "evidence_manifest.json").exists())
            self.assertTrue((Path(evidence.output_directory) / "vehicle" / "best_overall.jpg").exists())
            self.assertTrue((Path(evidence.output_directory) / "full_frames" / "best_overall.jpg").exists())
            self.assertTrue((Path(evidence.output_directory) / "annotated_frames" / "best_overall.jpg").exists())
            self.assertIsNotNone(evidence.full_frame_path)
            self.assertIsNotNone(evidence.annotated_full_frame_path)
            self.assertIsNotNone(evidence.manifest_path)

    def test_edge_clipped_candidate_records_visibility_diagnostics(self) -> None:
        collector = TrackEvidenceCollector(
            EvidenceConfig(enabled=True, minimum_crop_width=20, minimum_crop_height=20, minimum_padding_pixels=12),
            run_id="RUN_TEST",
        )
        frame = np.full((80, 80, 3), 127, dtype=np.uint8)
        collector.update(_packet(0, frame), [_observation(0, bbox_xyxy=(0.0, 8.0, 20.0, 50.0))])
        evidence = collector.finalize_track(_track(1))
        self.assertIsNotNone(evidence)
        assert evidence is not None
        candidate = evidence.candidates["best_overall"]
        self.assertTrue(candidate.crop_clipped)
        self.assertTrue(candidate.touches_left_edge)
        self.assertLess(candidate.visible_bbox_ratio, 1.0)

    def test_best_overall_prefers_fully_visible_crop_over_edge_crop(self) -> None:
        collector = TrackEvidenceCollector(EvidenceConfig(enabled=True, minimum_crop_width=20, minimum_crop_height=20), run_id="RUN_TEST")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        centered_crop = frame.copy()
        centered_crop[30:90, 50:110] = 255
        edge_crop = frame.copy()
        edge_crop[10:70, 0:60] = 255

        collector.update(_packet(0, edge_crop), [_observation(0, confidence=0.95, bbox_xyxy=(0.0, 10.0, 60.0, 70.0))])
        collector.update(_packet(1, centered_crop), [_observation(1, confidence=0.90, bbox_xyxy=(50.0, 30.0, 110.0, 90.0))])

        evidence = collector.finalize_track(_track(2))
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence.candidates["best_overall"].frame_number, 1)

    def test_selected_annotation_label_contains_raw_and_final_class(self) -> None:
        track = _track(1)
        track.class_name = "truck"
        track.stable_class_name = "truck"
        candidate = EvidenceCandidate(
            candidate_type="best_overall",
            frame_number=220,
            video_time_seconds=7.333333333333333,
            confidence=0.70,
            original_bbox_xyxy=(10.0, 10.0, 80.0, 80.0),
            expanded_bbox_xyxy=(8.0, 8.0, 82.0, 82.0),
            bbox_xyxy=(8.0, 8.0, 82.0, 82.0),
            crop_width=74,
            crop_height=74,
            area=5476,
            sharpness_score=0.9,
            visibility_score=1.0,
            centeredness_score=0.9,
            visible_bbox_ratio=1.0,
            edge_penalty=0.0,
            overall_score=0.9,
            crop_clipped=False,
            touches_left_edge=False,
            touches_right_edge=False,
            touches_top_edge=False,
            touches_bottom_edge=False,
            encoded_jpeg=b"",
        )
        label = TrackEvidenceCollector._build_selected_label(
            track,
            candidate,
            {
                "track_uuid": track.track_uuid,
                "class_name": "car",
                "raw_class_name": "car",
                "confidence": 0.702154815196991,
                "frame_number": 220,
                "video_time_seconds": 7.333333333333333,
            },
        )

        self.assertIn("RAW CAR", label)
        self.assertIn("FINAL TRUCK", label)
        self.assertIn("frame 220", label)

    def test_selected_annotation_label_uses_mixed_identity_when_final_class_is_blocked(self) -> None:
        track = _track(1)
        track.class_name = "truck"
        track.class_status = "MIXED_IDENTITY"
        track.stable_class_name = None
        candidate = EvidenceCandidate(
            candidate_type="best_overall",
            frame_number=220,
            video_time_seconds=7.333333333333333,
            confidence=0.70,
            original_bbox_xyxy=(10.0, 10.0, 80.0, 80.0),
            expanded_bbox_xyxy=(8.0, 8.0, 82.0, 82.0),
            bbox_xyxy=(8.0, 8.0, 82.0, 82.0),
            crop_width=74,
            crop_height=74,
            area=5476,
            sharpness_score=0.9,
            visibility_score=1.0,
            centeredness_score=0.9,
            visible_bbox_ratio=1.0,
            edge_penalty=0.0,
            overall_score=0.9,
            crop_clipped=False,
            touches_left_edge=False,
            touches_right_edge=False,
            touches_top_edge=False,
            touches_bottom_edge=False,
            encoded_jpeg=b"",
        )
        label = TrackEvidenceCollector._build_selected_label(
            track,
            candidate,
            {
                "track_uuid": track.track_uuid,
                "class_name": "car",
                "raw_class_name": "car",
                "confidence": 0.702154815196991,
                "frame_number": 220,
                "video_time_seconds": 7.333333333333333,
            },
        )

        self.assertIn("RAW CAR", label)
        self.assertIn("FINAL MIXED_IDENTITY", label)


if __name__ == "__main__":
    unittest.main()
