from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.evidence_to_track_media_mapper import build_track_media_records


def _candidate(path: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_type="best_overall",
        frame_number=10,
        video_time_seconds=1.2,
        confidence=0.87,
        original_bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        expanded_bbox_xyxy=(0.0, 1.0, 11.0, 13.0),
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        crop_width=128,
        crop_height=64,
        area=8192,
        sharpness_score=787.66,
        visibility_score=0.95,
        centeredness_score=0.88,
        visible_bbox_ratio=0.97,
        edge_penalty=0.1,
        overall_score=0.9,
        crop_clipped=False,
        touches_left_edge=False,
        touches_right_edge=False,
        touches_top_edge=False,
        touches_bottom_edge=False,
        encoded_jpeg=b"x",
        file_path=path,
    )


class EvidenceToTrackMediaMapperTests(unittest.TestCase):
    def test_best_overall_record_generated_with_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "RUN_1" / "CAM_001" / "track_000001" / "RUN_1_CAM_001_TRACK_1" / "best_overall.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"jpeg")
            evidence = TrackEvidencePackage(
                run_id="RUN_1",
                camera_code="CAM_001",
                local_track_id=1,
                track_uuid="RUN_1:CAM_001:TRACK_1",
                class_name="car",
                candidates={"best_overall": _candidate(str(target))},
                output_directory=str(target.parent),
            )
            records = build_track_media_records(
                evidence_package=evidence,
                vehicle_track_id="track-id",
                camera_id="camera-id",
                artifact_root=root,
                persist_roles=["BEST_OVERALL"],
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].storage_uri, "RUN_1/CAM_001/track_000001/RUN_1_CAM_001_TRACK_1/best_overall.jpg")
            self.assertEqual(records[0].media_type, "BEST_VEHICLE_CROP")
            self.assertNotIn("encoded_jpeg", records[0].metadata)

    def test_full_frame_record_generated_when_package_includes_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "RUN_1" / "CAM_001" / "track_000001" / "RUN_1_CAM_001_TRACK_1" / "best_overall.jpg"
            full_frame = root / "RUN_1" / "CAM_001" / "track_000001" / "RUN_1_CAM_001_TRACK_1" / "full_frames" / "best_overall.jpg"
            annotated_frame = root / "RUN_1" / "CAM_001" / "track_000001" / "RUN_1_CAM_001_TRACK_1" / "annotated_frames" / "best_overall.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"jpeg")
            full_frame.parent.mkdir(parents=True, exist_ok=True)
            full_frame.write_bytes(b"fullframe")
            annotated_frame.parent.mkdir(parents=True, exist_ok=True)
            annotated_frame.write_bytes(b"annotated")
            candidate = _candidate(str(target))
            candidate = EvidenceCandidate(
                candidate_type=candidate.candidate_type,
                frame_number=candidate.frame_number,
                video_time_seconds=candidate.video_time_seconds,
                confidence=candidate.confidence,
                original_bbox_xyxy=candidate.original_bbox_xyxy,
                expanded_bbox_xyxy=candidate.expanded_bbox_xyxy,
                bbox_xyxy=candidate.bbox_xyxy,
                crop_width=candidate.crop_width,
                crop_height=candidate.crop_height,
                area=candidate.area,
                sharpness_score=candidate.sharpness_score,
                visibility_score=candidate.visibility_score,
                centeredness_score=candidate.centeredness_score,
                visible_bbox_ratio=candidate.visible_bbox_ratio,
                edge_penalty=candidate.edge_penalty,
                overall_score=candidate.overall_score,
                crop_clipped=candidate.crop_clipped,
                touches_left_edge=candidate.touches_left_edge,
                touches_right_edge=candidate.touches_right_edge,
                touches_top_edge=candidate.touches_top_edge,
                touches_bottom_edge=candidate.touches_bottom_edge,
                encoded_jpeg=candidate.encoded_jpeg,
                file_path=candidate.file_path,
                source_frame_path=str(full_frame),
                annotated_frame_path=str(annotated_frame),
                source_frame_width=1920,
                source_frame_height=1080,
            )
            evidence = TrackEvidencePackage(
                run_id="RUN_1",
                camera_code="CAM_001",
                local_track_id=1,
                track_uuid="RUN_1:CAM_001:TRACK_1",
                class_name="car",
                candidates={"best_overall": candidate},
                output_directory=str(target.parent),
                full_frame_path=str(full_frame),
                full_frame_frame_number=10,
                full_frame_video_time_seconds=1.2,
                full_frame_bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                full_frame_width=1920,
                full_frame_height=1080,
            )
            records = build_track_media_records(
                evidence_package=evidence,
                vehicle_track_id="track-id",
                camera_id="camera-id",
                artifact_root=root,
                persist_roles=["BEST_OVERALL"],
            )
            self.assertEqual(len(records), 3)
            full_frame_record = next(record for record in records if record.media_type == "FULL_FRAME")
            self.assertEqual(full_frame_record.storage_uri, "RUN_1/CAM_001/track_000001/RUN_1_CAM_001_TRACK_1/full_frames/best_overall.jpg")
            self.assertEqual(full_frame_record.width, 1920)
            self.assertEqual(full_frame_record.height, 1080)
            annotated_frame_record = next(record for record in records if record.media_type == "ANNOTATED_FULL_FRAME")
            self.assertEqual(annotated_frame_record.storage_uri, "RUN_1/CAM_001/track_000001/RUN_1_CAM_001_TRACK_1/annotated_frames/best_overall.jpg")

    def test_missing_file_reported(self) -> None:
        evidence = TrackEvidencePackage(
            run_id="RUN_1",
            camera_code="CAM_001",
            local_track_id=1,
            track_uuid="RUN_1:CAM_001:TRACK_1",
            class_name="car",
            candidates={"best_overall": _candidate("missing.jpg")},
        )
        with self.assertRaises(FileNotFoundError):
            build_track_media_records(
                evidence_package=evidence,
                vehicle_track_id="track-id",
                camera_id="camera-id",
                artifact_root=Path("."),
                persist_roles=["BEST_OVERALL"],
            )


if __name__ == "__main__":
    unittest.main()
