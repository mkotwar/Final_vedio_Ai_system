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
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        crop_width=128,
        crop_height=64,
        area=8192,
        sharpness_score=787.66,
        edge_penalty=0.1,
        overall_score=0.9,
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
