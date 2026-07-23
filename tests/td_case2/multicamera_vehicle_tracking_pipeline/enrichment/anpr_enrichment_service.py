from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..persistence.persistence_config import PersistenceConfig
from ..persistence.persistence_models import PlateDetectionRecord, PlateReadingRecord, PlateSummaryRecord, TrackMediaRecord
from ..persistence.track_media_repository import TrackMediaRepository
from ..tracking.tracking_models import LocalVehicleTrack
from .anpr_config import AnprConfig
from .best_plate_selector import select_best_plate_candidates
from .florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from .plate_candidate_collector import PlateCandidateCollector
from .plate_models import PlateCandidate, PlateOcrResult
from .vehicle_evidence_selector import select_vehicle_evidence_candidates


@dataclass(slots=True)
class AnprMetrics:
    anpr_tracks_considered: int = 0
    anpr_tracks_with_vehicle_evidence: int = 0
    anpr_vehicle_crops_examined: int = 0
    anpr_plate_detections: int = 0
    anpr_plate_candidates_saved: int = 0
    anpr_tracks_with_plate: int = 0
    anpr_tracks_without_plate: int = 0
    anpr_ocr_attempts: int = 0
    anpr_ocr_successes: int = 0
    anpr_verified_plates: int = 0
    anpr_unverified_plates: int = 0
    anpr_unknown_results: int = 0
    anpr_plate_media_validated: int = 0
    anpr_plate_media_inserted: int = 0
    anpr_plate_media_failed: int = 0
    anpr_readings_validated: int = 0
    anpr_readings_inserted: int = 0
    anpr_summaries_validated: int = 0
    anpr_summaries_inserted: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.__dict__)


@dataclass(frozen=True, slots=True)
class AnprEnrichmentResult:
    track_uuid: str
    camera_code: str
    mode: str
    status: str
    persisted: bool
    raw_text: str | None
    normalized_text: str | None
    verification_status: str
    plate_detected: bool
    selected_source_vehicle_role: str | None
    plate_storage_uri: str | None
    detector_confidence: float | None
    ocr_confidence: float | None
    error: str | None = None

    def to_report_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "persisted": self.persisted,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "verification_status": self.verification_status,
            "plate_detected": self.plate_detected,
            "source_vehicle_role": self.selected_source_vehicle_role,
            "plate_storage_uri": self.plate_storage_uri,
            "detector_confidence": self.detector_confidence,
            "ocr_confidence": self.ocr_confidence,
            "error": self.error,
        }


class AnprEnrichmentService:
    def __init__(
        self,
        *,
        config: AnprConfig,
        persistence_config: PersistenceConfig,
        artifact_root: Path,
        candidate_collector: PlateCandidateCollector,
        ocr_extractor: FlorencePlateOcrExtractor,
        track_media_repository: TrackMediaRepository | None = None,
        plate_detection_repository=None,
        plate_reading_repository=None,
        plate_summary_repository=None,
    ) -> None:
        self.config = config
        self.persistence_config = persistence_config
        self.artifact_root = artifact_root.resolve()
        self.candidate_collector = candidate_collector
        self.ocr_extractor = ocr_extractor
        self.track_media_repository = track_media_repository
        self.plate_detection_repository = plate_detection_repository
        self.plate_reading_repository = plate_reading_repository
        self.plate_summary_repository = plate_summary_repository
        self.metrics = AnprMetrics()

    def get_metrics(self) -> AnprMetrics:
        return self.metrics

    def enrich_track(self, *, completed_track: LocalVehicleTrack, persisted_vehicle_track_id: str) -> AnprEnrichmentResult:
        self.metrics.anpr_tracks_considered += 1
        if not self.config.enabled:
            return AnprEnrichmentResult(
                track_uuid=completed_track.track_uuid,
                camera_code=completed_track.camera_code,
                mode="disabled",
                status="DISABLED",
                persisted=False,
                raw_text=None,
                normalized_text=None,
                verification_status="DISABLED",
                plate_detected=False,
                selected_source_vehicle_role=None,
                plate_storage_uri=None,
                detector_confidence=None,
                ocr_confidence=None,
            )
        evidence_inputs = select_vehicle_evidence_candidates(
            completed_track=completed_track,
            configured_roles=self.config.vehicle_evidence_roles,
            maximum_candidates=self.config.maximum_vehicle_crops_per_track,
            artifact_root=self.artifact_root,
        )
        if not evidence_inputs:
            self.metrics.anpr_tracks_without_plate += 1
            return AnprEnrichmentResult(
                track_uuid=completed_track.track_uuid,
                camera_code=completed_track.camera_code,
                mode=_mode(self.persistence_config),
                status="NO_VEHICLE_EVIDENCE",
                persisted=False,
                raw_text=None,
                normalized_text=None,
                verification_status="UNKNOWN",
                plate_detected=False,
                selected_source_vehicle_role=None,
                plate_storage_uri=None,
                detector_confidence=None,
                ocr_confidence=None,
            )
        self.metrics.anpr_tracks_with_vehicle_evidence += 1
        self.metrics.anpr_vehicle_crops_examined += len(evidence_inputs)
        candidates = self.candidate_collector.collect(evidence_inputs)
        self.metrics.anpr_plate_detections += len(candidates)
        self.metrics.anpr_plate_candidates_saved += len(candidates)
        if not candidates:
            self.metrics.anpr_tracks_without_plate += 1
            return AnprEnrichmentResult(
                track_uuid=completed_track.track_uuid,
                camera_code=completed_track.camera_code,
                mode=_mode(self.persistence_config),
                status="NO_PLATE_DETECTED",
                persisted=False,
                raw_text=None,
                normalized_text=None,
                verification_status="UNKNOWN",
                plate_detected=False,
                selected_source_vehicle_role=None,
                plate_storage_uri=None,
                detector_confidence=None,
                ocr_confidence=None,
            )
        self.metrics.anpr_tracks_with_plate += 1
        selected = select_best_plate_candidates(
            candidates,
            maximum_for_ocr=self.config.ocr.maximum_plate_candidates_for_ocr,
            config=self.config.plate_selection,
        )
        ocr_result = self._run_ocr(selected)
        if ocr_result is None:
            self.metrics.anpr_unknown_results += 1
            return AnprEnrichmentResult(
                track_uuid=completed_track.track_uuid,
                camera_code=completed_track.camera_code,
                mode=_mode(self.persistence_config),
                status="OCR_EMPTY",
                persisted=False,
                raw_text=None,
                normalized_text=None,
                verification_status="UNKNOWN",
                plate_detected=True,
                selected_source_vehicle_role=selected[0].source_vehicle_role if selected else None,
                plate_storage_uri=selected[0].relative_storage_uri if selected else None,
                detector_confidence=selected[0].detector_confidence if selected else None,
                ocr_confidence=None,
            )
        persisted = self._persist(
            completed_track=completed_track,
            persisted_vehicle_track_id=persisted_vehicle_track_id,
            candidate=selected[0],
            ocr_result=ocr_result,
        )
        if ocr_result.verification_status == "VERIFIED":
            self.metrics.anpr_verified_plates += 1
        elif ocr_result.normalized_text:
            self.metrics.anpr_unverified_plates += 1
        else:
            self.metrics.anpr_unknown_results += 1
        return AnprEnrichmentResult(
            track_uuid=completed_track.track_uuid,
            camera_code=completed_track.camera_code,
            mode=_mode(self.persistence_config),
            status=ocr_result.status,
            persisted=persisted,
            raw_text=ocr_result.raw_text,
            normalized_text=ocr_result.normalized_text,
            verification_status=ocr_result.verification_status,
            plate_detected=True,
            selected_source_vehicle_role=selected[0].source_vehicle_role,
            plate_storage_uri=selected[0].relative_storage_uri,
            detector_confidence=selected[0].detector_confidence,
            ocr_confidence=ocr_result.confidence,
        )

    def _run_ocr(self, candidates: list[PlateCandidate]) -> PlateOcrResult | None:
        if not candidates:
            return None
        best_result: PlateOcrResult | None = None
        for candidate in candidates:
            self.metrics.anpr_ocr_attempts += 1
            result = self.ocr_extractor.extract(candidate)
            if result.raw_text:
                self.metrics.anpr_ocr_successes += 1
            if best_result is None:
                best_result = result
            if result.normalized_text or not self.config.ocr.fallback_to_other_plate_candidates:
                return result
        return best_result

    def _persist(
        self,
        *,
        completed_track: LocalVehicleTrack,
        persisted_vehicle_track_id: str,
        candidate: PlateCandidate,
        ocr_result: PlateOcrResult,
    ) -> bool:
        media_record = TrackMediaRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            media_type="PLATE_CROP",
            storage_uri=candidate.relative_storage_uri,
            frame_number=candidate.frame_number,
            captured_at=datetime.now(timezone.utc),
            video_time_seconds=candidate.video_time_seconds,
            width=candidate.crop_width,
            height=candidate.crop_height,
            quality_score=min(1.0, candidate.overall_score),
            sharpness_score=candidate.sharpness_score,
            selection_rank=1,
            is_primary=True,
            metadata={"source_vehicle_role": candidate.source_vehicle_role, "source_vehicle_storage_uri": candidate.source_vehicle_storage_uri},
        )
        detection_record = PlateDetectionRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            detected_at=datetime.now(timezone.utc),
            frame_number=candidate.frame_number,
            bbox_x1=candidate.plate_bbox_xyxy[0],
            bbox_y1=candidate.plate_bbox_xyxy[1],
            bbox_x2=candidate.plate_bbox_xyxy[2],
            bbox_y2=candidate.plate_bbox_xyxy[3],
            confidence=candidate.detector_confidence,
            detector_name="OCR_MUKUL_YOLO_PLATE",
            detector_version=None,
            metadata={"source_vehicle_role": candidate.source_vehicle_role, "storage_uri": candidate.relative_storage_uri},
        )
        reading_record = PlateReadingRecord(
            plate_detection_id="DRYRUN:PLATE_DETECTION" if self.persistence_config.backend != "analytics_supabase" else "",
            ocr_engine="FLORENCE",
            ocr_version=ocr_result.model_name,
            raw_text=ocr_result.raw_text,
            normalized_text=ocr_result.normalized_text,
            plate_pattern=str(ocr_result.metadata.get("matched_pattern")) if ocr_result.metadata.get("matched_pattern") else None,
            confidence=ocr_result.confidence,
            status=_plate_reading_status(ocr_result),
            is_selected=True,
            metadata=ocr_result.metadata,
        )
        summary_record = PlateSummaryRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            selected_plate_reading_id="DRYRUN:PLATE_READING" if self.persistence_config.backend != "analytics_supabase" else None,
            canonical_plate=ocr_result.normalized_text if (ocr_result.verification_status == "VERIFIED" or not self.config.validation.persist_only_verified_as_primary) else None,
            plate_pattern=str(ocr_result.metadata.get("matched_pattern")) if ocr_result.metadata.get("matched_pattern") else None,
            status=_plate_reading_status(ocr_result),
            confidence=ocr_result.confidence,
            reading_count=1,
        )
        media_record.to_payload()
        self.metrics.anpr_plate_media_validated += 1
        detection_record.to_payload()
        self.metrics.anpr_readings_validated += 1
        reading_record.to_payload()
        self.metrics.anpr_readings_validated += 1
        summary_record.to_payload()
        self.metrics.anpr_summaries_validated += 1
        if self.persistence_config.backend != "analytics_supabase" or not self.config.persist_result:
            return False
        media_row = self.track_media_repository.upsert(media_record) if self.track_media_repository is not None else None
        self.metrics.anpr_plate_media_inserted += 1 if media_row is not None else 0
        if media_row is not None:
            detection_record = PlateDetectionRecord(
                vehicle_track_id=persisted_vehicle_track_id,
                track_media_id=str(media_row["id"]),
                detected_at=detection_record.detected_at,
                frame_number=detection_record.frame_number,
                bbox_x1=detection_record.bbox_x1,
                bbox_y1=detection_record.bbox_y1,
                bbox_x2=detection_record.bbox_x2,
                bbox_y2=detection_record.bbox_y2,
                confidence=detection_record.confidence,
                detector_name=detection_record.detector_name,
                detector_version=detection_record.detector_version,
                metadata=detection_record.metadata,
            )
        detection_row = self.plate_detection_repository.create_plate_detection(detection_record)
        reading_record = PlateReadingRecord(
            plate_detection_id=str(detection_row["id"]),
            ocr_engine=reading_record.ocr_engine,
            ocr_version=reading_record.ocr_version,
            raw_text=reading_record.raw_text,
            normalized_text=reading_record.normalized_text,
            plate_pattern=reading_record.plate_pattern,
            confidence=reading_record.confidence,
            status=reading_record.status,
            is_selected=True,
            metadata=reading_record.metadata,
        )
        reading_row = self.plate_reading_repository.create_plate_reading(reading_record)
        self.metrics.anpr_readings_inserted += 1
        summary_record = PlateSummaryRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            selected_plate_reading_id=str(reading_row["id"]),
            canonical_plate=summary_record.canonical_plate,
            plate_pattern=summary_record.plate_pattern,
            status=summary_record.status,
            confidence=summary_record.confidence,
            reading_count=summary_record.reading_count,
        )
        self.plate_summary_repository.upsert_plate_summary(summary_record)
        self.metrics.anpr_summaries_inserted += 1
        return True


def _mode(persistence_config: PersistenceConfig) -> str:
    return "analytics_supabase" if persistence_config.backend == "analytics_supabase" else "dry_run"


def _plate_reading_status(result: PlateOcrResult) -> str:
    if result.verification_status == "VERIFIED":
        return "VERIFIED"
    if result.normalized_text:
        return "PROBABLE"
    if result.raw_text and result.raw_text != "UNKNOWN":
        return "PARTIAL"
    return "UNKNOWN"
