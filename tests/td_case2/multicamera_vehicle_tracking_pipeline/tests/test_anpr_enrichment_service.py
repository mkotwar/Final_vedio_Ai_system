from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.anpr_config import AnprConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.anpr_enrichment_service import AnprEnrichmentService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_models import PlateCandidate, PlateOcrResult, VehicleEvidenceInput
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


class _FakeCollector:
    def __init__(self, candidate: PlateCandidate) -> None:
        self.candidate = candidate

    def collect(self, vehicle_evidence):
        return [self.candidate]


class _FakeExtractor:
    def extract(self, candidate: PlateCandidate) -> PlateOcrResult:
        return PlateOcrResult(
            raw_text="MH 12 AB 1234",
            normalized_text="MH12AB1234",
            confidence=0.9,
            status="VERIFIED",
            verification_status="VERIFIED",
            country_profile="INDIA",
            backend="florence",
            model_name="florence",
            adapter_name="adapter",
            source_vehicle_track_id=candidate.track_uuid,
            source_plate_storage_uri=candidate.relative_storage_uri,
            source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
            metadata={"matched_pattern": "STANDARD"},
        )


class _FakeTrackMediaRepository:
    def upsert(self, record):
        return {"id": "media-id", "storage_uri": record.storage_uri}


class _FakePlateDetectionRepository:
    def create_plate_detection(self, record):
        return {"id": "plate-detection-id"}


class _FakePlateReadingRepository:
    def create_plate_reading(self, record):
        return {"id": "plate-reading-id"}


class _FakePlateSummaryRepository:
    def upsert_plate_summary(self, record):
        return {"vehicle_track_id": record.vehicle_track_id}


def _track(evidence_path: Path | None = None) -> LocalVehicleTrack:
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
    if evidence_path is not None:
        track.evidence_package = TrackEvidencePackage(
            run_id="RUN_1",
            camera_code="CAM_001",
            local_track_id=1,
            track_uuid=track.track_uuid,
            class_name=track.class_name,
            candidates={
                "best_overall": EvidenceCandidate(
                    candidate_type="best_overall",
                    frame_number=10,
                    video_time_seconds=0.5,
                    confidence=0.9,
                    bbox_xyxy=(1.0, 2.0, 30.0, 15.0),
                    crop_width=120,
                    crop_height=40,
                    area=4800,
                    sharpness_score=0.8,
                    edge_penalty=0.0,
                    overall_score=0.9,
                    encoded_jpeg=b"",
                    file_path=str(evidence_path),
                )
            },
            output_directory=str(evidence_path.parent),
        )
    return track


class AnprEnrichmentServiceTests(unittest.TestCase):
    def test_dry_run_validates_without_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            plate_path = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "plate_evidence" / "candidate_001.jpg"
            plate_path.parent.mkdir(parents=True)
            plate_path.write_bytes(b"jpg")
            candidate = PlateCandidate(
                track_uuid="RUN:CAM_001:TRACK_1",
                camera_code="CAM_001",
                source_vehicle_role="BEST_OVERALL",
                source_vehicle_storage_uri="RUN_1/CAM_001/track_000001/best_overall.jpg",
                plate_bbox_xyxy=(1.0, 2.0, 30.0, 15.0),
                detector_confidence=0.9,
                crop_width=120,
                crop_height=40,
                area=4800,
                aspect_ratio=3.0,
                sharpness_score=0.8,
                edge_penalty=0.0,
                overall_score=0.9,
                local_file_path=plate_path,
                relative_storage_uri="RUN_1/CAM_001/track_000001/plate_evidence/candidate_001.jpg",
                frame_number=10,
                video_time_seconds=0.5,
            )
            service = AnprEnrichmentService(
                config=AnprConfig(enabled=True),
                persistence_config=PersistenceConfig(backend="dry_run", dry_run=True),
                artifact_root=artifact_root,
                candidate_collector=_FakeCollector(candidate),
                ocr_extractor=_FakeExtractor(),
            )
            evidence_path = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "best_overall.jpg"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(b"jpg")
            result = service.enrich_track(completed_track=_track(evidence_path), persisted_vehicle_track_id="DRYRUN:TRACK:RUN:CAM_001:TRACK_1")
            self.assertEqual(result.mode, "dry_run")
            self.assertFalse(result.persisted)
            self.assertEqual(result.normalized_text, "MH12AB1234")

    def test_analytics_mode_persists_after_detection_id_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            plate_path = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "plate_evidence" / "candidate_001.jpg"
            plate_path.parent.mkdir(parents=True)
            plate_path.write_bytes(b"jpg")
            candidate = PlateCandidate(
                track_uuid="RUN:CAM_001:TRACK_1",
                camera_code="CAM_001",
                source_vehicle_role="BEST_OVERALL",
                source_vehicle_storage_uri="RUN_1/CAM_001/track_000001/best_overall.jpg",
                plate_bbox_xyxy=(1.0, 2.0, 30.0, 15.0),
                detector_confidence=0.9,
                crop_width=120,
                crop_height=40,
                area=4800,
                aspect_ratio=3.0,
                sharpness_score=0.8,
                edge_penalty=0.0,
                overall_score=0.9,
                local_file_path=plate_path,
                relative_storage_uri="RUN_1/CAM_001/track_000001/plate_evidence/candidate_001.jpg",
                frame_number=10,
                video_time_seconds=0.5,
            )
            service = AnprEnrichmentService(
                config=AnprConfig(enabled=True),
                persistence_config=PersistenceConfig(backend="analytics_supabase", dry_run=False),
                artifact_root=artifact_root,
                candidate_collector=_FakeCollector(candidate),
                ocr_extractor=_FakeExtractor(),
                track_media_repository=_FakeTrackMediaRepository(),
                plate_detection_repository=_FakePlateDetectionRepository(),
                plate_reading_repository=_FakePlateReadingRepository(),
                plate_summary_repository=_FakePlateSummaryRepository(),
            )
            evidence_path = artifact_root / "RUN_1" / "CAM_001" / "track_000001" / "best_overall.jpg"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(b"jpg")
            result = service.enrich_track(completed_track=_track(evidence_path), persisted_vehicle_track_id="track-id")
            self.assertTrue(result.persisted)
            self.assertEqual(result.normalized_text, "MH12AB1234")
            self.assertEqual(service.metrics.anpr_readings_inserted, 1)
            self.assertEqual(service.metrics.anpr_summaries_inserted, 1)


if __name__ == "__main__":
    unittest.main()
