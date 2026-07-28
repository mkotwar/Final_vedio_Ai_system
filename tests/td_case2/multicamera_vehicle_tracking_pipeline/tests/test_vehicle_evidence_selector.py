from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_evidence_selector import select_vehicle_evidence_candidates
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


def _track(artifact_file: Path) -> LocalVehicleTrack:
    observation = TrackObservation("CAM_001", 1, 0, 0.0, datetime(2026, 7, 23, 12, 0, 0), "car", 0.9, (1.0, 2.0, 5.0, 6.0), "RUN:CAM_001:TRACK_1", "active")
    track = LocalVehicleTrack(
        track_uuid="RUN:CAM_001:TRACK_1",
        camera_code="CAM_001",
        local_track_id=1,
        class_name="car",
        first_frame_number=0,
        last_frame_number=0,
        first_seen_at=observation.camera_timestamp,
        last_seen_at=observation.camera_timestamp,
        first_video_time_seconds=0.0,
        last_video_time_seconds=0.0,
        observation_count=1,
        best_confidence=0.9,
        state="completed",
        observations=[observation],
        camera_name="Cam",
        source_path=Path("cam.mp4"),
    )
    track.evidence_package = TrackEvidencePackage(
        run_id="RUN_1",
        camera_code="CAM_001",
        local_track_id=1,
        track_uuid=track.track_uuid,
        class_name=track.class_name,
        candidates={
            "best_overall": EvidenceCandidate(
                candidate_type="best_overall",
                frame_number=0,
                video_time_seconds=0.0,
                confidence=0.9,
                original_bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                expanded_bbox_xyxy=(0.0, 1.0, 6.0, 7.0),
                bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                crop_width=100,
                crop_height=80,
                area=8000,
                sharpness_score=0.5,
                visibility_score=0.9,
                centeredness_score=0.8,
                visible_bbox_ratio=0.95,
                edge_penalty=0.1,
                overall_score=0.9,
                crop_clipped=False,
                touches_left_edge=False,
                touches_right_edge=False,
                touches_top_edge=False,
                touches_bottom_edge=False,
                encoded_jpeg=b"",
                file_path=str(artifact_file),
            )
        },
        output_directory=str(artifact_file.parent),
    )
    return track


class VehicleEvidenceSelectorTests(unittest.TestCase):
    def test_preserves_order_and_builds_relative_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            image_path = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "best_overall.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"jpg")
            selected = select_vehicle_evidence_candidates(
                completed_track=_track(image_path),
                configured_roles=("BEST_OVERALL", "SHARPEST"),
                maximum_candidates=5,
                artifact_root=artifact_root,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].source_vehicle_role, "BEST_OVERALL")
            self.assertEqual(selected[0].source_vehicle_storage_uri, "RUN_1/CAM_001/track_000001/best_overall.jpg")


if __name__ == "__main__":
    unittest.main()
