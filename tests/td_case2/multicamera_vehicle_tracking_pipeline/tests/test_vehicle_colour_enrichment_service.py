from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_config import VehicleColourConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_enrichment_service import VehicleColourEnrichmentService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_models import VehicleColourResult
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


class _FakeExtractor:
    def extract(self, image_path: Path, *, track_uuid: str, camera_code: str, source_storage_uri: str) -> VehicleColourResult:
        return VehicleColourResult(
            canonical_colour="WHITE",
            raw_output='{"primary_colour":"WHITE"}',
            confidence=0.8,
            status="SUCCESS",
            source_storage_uri=source_storage_uri,
            metadata={"track_uuid": track_uuid, "camera_code": camera_code},
        )


def _track() -> LocalVehicleTrack:
    observation = TrackObservation("CAM_001", 1, 0, 0.0, datetime(2026, 7, 23, 12, 0, 0), "car", 0.9, (1.0, 2.0, 5.0, 6.0), "RUN:CAM_001:TRACK_1", "active")
    return LocalVehicleTrack(
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


class VehicleColourEnrichmentServiceTests(unittest.TestCase):
    def test_dry_run_uses_evidence_fallback_without_db_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            target = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "RUN_1_CAM_001_TRACK_1" / "best_overall.jpg"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"jpg")
            track = _track()
            from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage

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
                        crop_height=100,
                        area=10000,
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
                        encoded_jpeg=None,
                        file_path=str(target),
                    )
                },
                output_directory=str(target.parent),
            )
            service = VehicleColourEnrichmentService(
                extractor=_FakeExtractor(),
                config=VehicleColourConfig(enabled=True),
                persistence_config=PersistenceConfig(backend="dry_run"),
                artifact_root=artifact_root,
            )
            result = service.enrich_track(completed_track=track, persisted_vehicle_track_id="DRYRUN:TRACK:RUN:CAM_001:TRACK_1")
            self.assertEqual(result.mode, "dry_run")
            self.assertFalse(result.persisted)
            self.assertEqual(result.result.canonical_colour, "WHITE")


if __name__ == "__main__":
    unittest.main()
