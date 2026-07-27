from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
from .plate_models import PlateCandidate, PlateOcrAttempt
from .plate_ocr_aggregation import AggregatedPlateResult, aggregate_ocr_attempts
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
    anpr_unreadable_results: int = 0
    anpr_conflicting_results: int = 0
    anpr_plate_media_validated: int = 0
    anpr_plate_media_inserted: int = 0
    anpr_plate_media_failed: int = 0
    anpr_readings_validated: int = 0
    anpr_readings_inserted: int = 0
    anpr_summaries_validated: int = 0
    anpr_summaries_inserted: int = 0
    detector_calls: int = 0
    detector_candidates: int = 0
    heuristic_candidates: int = 0
    padded_crops_examined: int = 0
    original_frame_regions_examined: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return deepcopy(asdict(self))


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
    support_frame_count: int = 0
    support_candidate_count: int = 0
    attempt_count: int = 0
    diagnostic_attempts: tuple[dict[str, object], ...] = ()
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
            "support_frame_count": self.support_frame_count,
            "support_candidate_count": self.support_candidate_count,
            "attempt_count": self.attempt_count,
            "diagnostic_attempts": list(self.diagnostic_attempts),
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
        self._sync_collection_metrics()
        return self.metrics

    def enrich_track(self, *, completed_track: LocalVehicleTrack, persisted_vehicle_track_id: str) -> AnprEnrichmentResult:
        self.metrics.anpr_tracks_considered += 1
        if not self.config.enabled:
            return self._result(
                completed_track=completed_track,
                status="DISABLED",
                persisted=False,
                verification_status="DISABLED",
                plate_detected=False,
            )
        evidence_inputs = select_vehicle_evidence_candidates(
            completed_track=completed_track,
            configured_roles=self.config.vehicle_evidence_roles,
            maximum_candidates=self.config.maximum_vehicle_crops_per_track,
            artifact_root=self.artifact_root,
        )
        if not evidence_inputs:
            self.metrics.anpr_tracks_without_plate += 1
            return self._result(
                completed_track=completed_track,
                status="NO_VEHICLE_EVIDENCE",
                persisted=False,
                verification_status="UNKNOWN",
                plate_detected=False,
            )
        self.metrics.anpr_tracks_with_vehicle_evidence += 1
        self.metrics.anpr_vehicle_crops_examined += len(evidence_inputs)
        candidates = self.candidate_collector.collect(evidence_inputs)
        self._sync_collection_metrics()
        self.metrics.anpr_plate_detections += len(candidates)
        self.metrics.anpr_plate_candidates_saved += len(candidates)
        if not candidates:
            self.metrics.anpr_tracks_without_plate += 1
            return self._result(
                completed_track=completed_track,
                status="NO_PLATE_DETECTED",
                persisted=False,
                verification_status="UNKNOWN",
                plate_detected=False,
            )
        self.metrics.anpr_tracks_with_plate += 1
        selected = select_best_plate_candidates(
            candidates,
            maximum_for_ocr=self.config.ocr.maximum_plate_candidates_for_ocr,
            config=self.config.plate_selection,
        )
        aggregated = self._run_ocr(selected)
        persisted = self._persist(
            completed_track=completed_track,
            persisted_vehicle_track_id=persisted_vehicle_track_id,
            aggregated=aggregated,
        )
        return self._result_from_aggregate(completed_track=completed_track, aggregated=aggregated, persisted=persisted)

    def _run_ocr(self, candidates: list[PlateCandidate]) -> AggregatedPlateResult:
        attempts_by_candidate_uri: dict[str, list[PlateOcrAttempt]] = {}
        for candidate in candidates:
            if hasattr(self.ocr_extractor, "extract_attempts"):
                attempts = self.ocr_extractor.extract_attempts(candidate)
            else:
                legacy_result = self.ocr_extractor.extract(candidate)
                attempts = [
                    PlateOcrAttempt(
                        candidate_storage_uri=candidate.relative_storage_uri,
                        source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
                        source_vehicle_role=candidate.source_vehicle_role,
                        source_image_kind=candidate.source_image_kind,
                        candidate_source=candidate.candidate_source,
                        preprocessing_variant="legacy_single_pass",
                        frame_number=candidate.frame_number,
                        video_time_seconds=candidate.video_time_seconds,
                        detector_confidence=candidate.detector_confidence,
                        raw_text=legacy_result.raw_text,
                        normalized_text=legacy_result.normalized_text,
                        confidence=legacy_result.confidence,
                        status=legacy_result.status,
                        verification_status=legacy_result.verification_status,
                        metadata=legacy_result.metadata,
                    )
                ]
            attempts_by_candidate_uri[candidate.relative_storage_uri] = attempts
            self.metrics.anpr_ocr_attempts += len(attempts)
            self.metrics.anpr_ocr_successes += sum(1 for attempt in attempts if attempt.raw_text and attempt.raw_text != "UNKNOWN")
        aggregated = aggregate_ocr_attempts(candidates=candidates, attempts_by_candidate_uri=attempts_by_candidate_uri, config=self.config)
        if aggregated.status == "VERIFIED":
            self.metrics.anpr_verified_plates += 1
        elif aggregated.status == "UNREADABLE":
            self.metrics.anpr_unreadable_results += 1
        elif aggregated.status == "CONFLICTING_CANDIDATES":
            self.metrics.anpr_conflicting_results += 1
        elif aggregated.normalized_text:
            self.metrics.anpr_unverified_plates += 1
        else:
            self.metrics.anpr_unknown_results += 1
        return aggregated

    def _persist(
        self,
        *,
        completed_track: LocalVehicleTrack,
        persisted_vehicle_track_id: str,
        aggregated: AggregatedPlateResult,
    ) -> bool:
        if aggregated.selected_candidate is None:
            return False
        candidate = aggregated.selected_candidate
        attempt = aggregated.selected_attempt
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
            metadata={
                "source_vehicle_role": candidate.source_vehicle_role,
                "source_vehicle_storage_uri": candidate.source_vehicle_storage_uri,
                "candidate_source": candidate.candidate_source,
                "source_image_kind": candidate.source_image_kind,
                "heuristic_region_name": candidate.heuristic_region_name,
            },
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
            metadata={
                "source_vehicle_role": candidate.source_vehicle_role,
                "storage_uri": candidate.relative_storage_uri,
                "candidate_source": candidate.candidate_source,
                "source_image_kind": candidate.source_image_kind,
                "heuristic_region_name": candidate.heuristic_region_name,
                "detector_imgsz": self.config.plate_detector.inference_image_size,
            },
        )
        reading_payload = {
            "ocr_engine": "FLORENCE",
            "ocr_version": attempt.metadata.get("variant_path") if attempt is not None else None,
            "raw_text": aggregated.raw_text,
            "normalized_text": aggregated.normalized_text,
            "plate_pattern": str(attempt.metadata.get("matched_pattern")) if attempt is not None and attempt.metadata.get("matched_pattern") else None,
            "confidence": aggregated.confidence,
            "status": _plate_reading_status(aggregated.status, aggregated.normalized_text),
            "is_selected": True,
            "metadata": {
                "aggregation_status": aggregated.status,
                "verification_status": aggregated.verification_status,
                "support_frame_count": aggregated.support_frame_count,
                "support_candidate_count": aggregated.support_candidate_count,
                "attempt_count": len(aggregated.attempts),
                "attempts": [_attempt_to_dict(item) for item in aggregated.attempts],
            },
        }
        summary_record = PlateSummaryRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            selected_plate_reading_id="DRYRUN:PLATE_READING" if self.persistence_config.backend != "analytics_supabase" else None,
            canonical_plate=aggregated.normalized_text if (aggregated.verification_status == "VERIFIED" or not self.config.validation.persist_only_verified_as_primary) else None,
            plate_pattern=str(attempt.metadata.get("matched_pattern")) if attempt is not None and attempt.metadata.get("matched_pattern") else None,
            status=_plate_reading_status(aggregated.status, aggregated.normalized_text),
            confidence=aggregated.confidence,
            reading_count=max(1, len(aggregated.attempts)),
        )
        media_record.to_payload()
        self.metrics.anpr_plate_media_validated += 1
        detection_record.to_payload()
        self.metrics.anpr_readings_validated += 1
        if self.persistence_config.backend != "analytics_supabase":
            reading_record = PlateReadingRecord(plate_detection_id="DRYRUN:PLATE_DETECTION", **reading_payload)
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
        reading_record = PlateReadingRecord(plate_detection_id=str(detection_row["id"]), **reading_payload)
        reading_record.to_payload()
        self.metrics.anpr_readings_validated += 1
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
        summary_record.to_payload()
        self.metrics.anpr_summaries_validated += 1
        self.plate_summary_repository.upsert_plate_summary(summary_record)
        self.metrics.anpr_summaries_inserted += 1
        return True

    def _result(
        self,
        *,
        completed_track: LocalVehicleTrack,
        status: str,
        persisted: bool,
        verification_status: str,
        plate_detected: bool,
    ) -> AnprEnrichmentResult:
        return AnprEnrichmentResult(
            track_uuid=completed_track.track_uuid,
            camera_code=completed_track.camera_code,
            mode=_mode(self.persistence_config),
            status=status,
            persisted=persisted,
            raw_text=None,
            normalized_text=None,
            verification_status=verification_status,
            plate_detected=plate_detected,
            selected_source_vehicle_role=None,
            plate_storage_uri=None,
            detector_confidence=None,
            ocr_confidence=None,
        )

    def _result_from_aggregate(
        self,
        *,
        completed_track: LocalVehicleTrack,
        aggregated: AggregatedPlateResult,
        persisted: bool,
    ) -> AnprEnrichmentResult:
        selected_candidate = aggregated.selected_candidate
        return AnprEnrichmentResult(
            track_uuid=completed_track.track_uuid,
            camera_code=completed_track.camera_code,
            mode=_mode(self.persistence_config),
            status=aggregated.status,
            persisted=persisted,
            raw_text=aggregated.raw_text,
            normalized_text=aggregated.normalized_text,
            verification_status=aggregated.verification_status,
            plate_detected=selected_candidate is not None,
            selected_source_vehicle_role=selected_candidate.source_vehicle_role if selected_candidate is not None else None,
            plate_storage_uri=selected_candidate.relative_storage_uri if selected_candidate is not None else None,
            detector_confidence=selected_candidate.detector_confidence if selected_candidate is not None else None,
            ocr_confidence=aggregated.confidence,
            support_frame_count=aggregated.support_frame_count,
            support_candidate_count=aggregated.support_candidate_count,
            attempt_count=len(aggregated.attempts),
            diagnostic_attempts=tuple(_attempt_to_dict(item) for item in aggregated.attempts),
        )

    def _sync_collection_metrics(self) -> None:
        collection_metrics = getattr(self.candidate_collector, "metrics", None)
        if collection_metrics is None:
            return
        self.metrics.detector_calls = collection_metrics.detector_calls
        self.metrics.detector_candidates = collection_metrics.detector_candidates
        self.metrics.heuristic_candidates = collection_metrics.heuristic_candidates
        self.metrics.padded_crops_examined = collection_metrics.padded_crops_examined
        self.metrics.original_frame_regions_examined = collection_metrics.original_frame_regions_examined


def _attempt_to_dict(attempt: PlateOcrAttempt) -> dict[str, object]:
    return {
        "candidate_storage_uri": attempt.candidate_storage_uri,
        "source_vehicle_storage_uri": attempt.source_vehicle_storage_uri,
        "source_vehicle_role": attempt.source_vehicle_role,
        "source_image_kind": attempt.source_image_kind,
        "candidate_source": attempt.candidate_source,
        "preprocessing_variant": attempt.preprocessing_variant,
        "frame_number": attempt.frame_number,
        "video_time_seconds": attempt.video_time_seconds,
        "detector_confidence": attempt.detector_confidence,
        "raw_text": attempt.raw_text,
        "normalized_text": attempt.normalized_text,
        "confidence": attempt.confidence,
        "status": attempt.status,
        "verification_status": attempt.verification_status,
        "metadata": deepcopy(attempt.metadata),
    }


def _mode(persistence_config: PersistenceConfig) -> str:
    return "analytics_supabase" if persistence_config.backend == "analytics_supabase" else "dry_run"


def _plate_reading_status(status: str, normalized_text: str | None) -> str:
    if status == "VERIFIED":
        return "VERIFIED"
    if normalized_text:
        return "PARTIAL" if len(normalized_text) < 6 else "PROBABLE"
    return "UNKNOWN"
