from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..persistence.persistence_config import PersistenceConfig
from ..persistence.persistence_models import TrackMediaRecord, TrackPersistenceResult, VehicleAttributeRecord
from ..persistence.track_media_repository import TrackMediaRepository
from ..persistence.vehicle_colour_repository import VehicleColourRepository
from ..tracking.tracking_models import LocalVehicleTrack
from .florence_vehicle_colour_extractor import FlorenceVehicleColourExtractor
from .media_resolver import MediaResolutionError, resolve_local_media_path
from .vehicle_colour_config import VehicleColourConfig
from .vehicle_colour_models import VehicleColourResult


@dataclass(slots=True)
class VehicleColourMetrics:
    colour_tracks_considered: int = 0
    colour_images_loaded: int = 0
    colour_extractions_attempted: int = 0
    colour_extractions_succeeded: int = 0
    colour_extractions_unknown: int = 0
    colour_extractions_low_confidence: int = 0
    colour_extractions_failed: int = 0
    colour_images_missing: int = 0
    colour_parse_failures: int = 0
    colour_records_validated: int = 0
    colour_records_inserted: int = 0
    colour_records_already_existing: int = 0
    colour_records_failed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.__dict__)


@dataclass(frozen=True, slots=True)
class VehicleColourEnrichmentResult:
    track_uuid: str
    persisted_vehicle_track_id: str | None
    mode: str
    persisted: bool
    result: VehicleColourResult

    def to_report_payload(self) -> dict[str, object]:
        payload = self.result.to_report_payload()
        payload["mode"] = self.mode
        payload["persisted"] = self.persisted
        payload["vehicle_track_id"] = self.persisted_vehicle_track_id
        return payload


class VehicleColourEnrichmentService:
    def __init__(
        self,
        *,
        extractor: FlorenceVehicleColourExtractor | None,
        config: VehicleColourConfig,
        persistence_config: PersistenceConfig,
        artifact_root: Path,
        track_media_repository: TrackMediaRepository | None = None,
        vehicle_colour_repository: VehicleColourRepository | None = None,
    ) -> None:
        self.extractor = extractor
        self.config = config
        self.persistence_config = persistence_config
        self.artifact_root = artifact_root.resolve()
        self.track_media_repository = track_media_repository
        self.vehicle_colour_repository = vehicle_colour_repository
        self.metrics = VehicleColourMetrics()

    def get_metrics(self) -> VehicleColourMetrics:
        return self.metrics

    def enrich_track(
        self,
        *,
        completed_track: LocalVehicleTrack,
        persisted_vehicle_track_id: str,
    ) -> VehicleColourEnrichmentResult:
        self.metrics.colour_tracks_considered += 1
        if not self.config.enabled or self.extractor is None:
            result = VehicleColourResult(
                canonical_colour="UNKNOWN",
                raw_output="disabled",
                confidence=0.0,
                status="DISABLED",
                source_storage_uri=None,
                metadata={"track_uuid": completed_track.track_uuid, "camera_code": completed_track.camera_code},
            )
            return VehicleColourEnrichmentResult(completed_track.track_uuid, persisted_vehicle_track_id, "disabled", False, result)
        try:
            storage_uri = self._resolve_source_storage_uri(completed_track, persisted_vehicle_track_id)
            image_path = resolve_local_media_path(storage_uri=storage_uri, artifact_root=self.artifact_root)
        except (FileNotFoundError, MediaResolutionError) as exc:
            self.metrics.colour_images_missing += 1
            self.metrics.colour_extractions_failed += 1
            result = VehicleColourResult(
                canonical_colour="UNKNOWN",
                raw_output=str(exc),
                confidence=0.0,
                status="IMAGE_MISSING",
                source_storage_uri=None,
                metadata={"track_uuid": completed_track.track_uuid, "camera_code": completed_track.camera_code},
            )
            return VehicleColourEnrichmentResult(completed_track.track_uuid, persisted_vehicle_track_id, "dry_run" if self.persistence_config.backend != "analytics_supabase" else "analytics_supabase", False, result)
        self.metrics.colour_images_loaded += 1
        self.metrics.colour_extractions_attempted += 1
        extracted = self.extractor.extract(
            image_path,
            track_uuid=completed_track.track_uuid,
            camera_code=completed_track.camera_code,
            source_storage_uri=storage_uri,
        )
        self._apply_extraction_metrics(extracted)
        record = self._build_record(
            completed_track=completed_track,
            persisted_vehicle_track_id=persisted_vehicle_track_id,
            result=extracted,
        )
        record.to_payload()
        self.metrics.colour_records_validated += 1
        if self.persistence_config.backend != "analytics_supabase" or not self.config.persist_result:
            return VehicleColourEnrichmentResult(completed_track.track_uuid, persisted_vehicle_track_id, "dry_run", False, extracted)
        if self.vehicle_colour_repository is None:
            raise RuntimeError("VehicleColourRepository is required for analytics_supabase enrichment.")
        row = self.vehicle_colour_repository.upsert_vehicle_colour(record)
        self.metrics.colour_records_inserted += 1
        return VehicleColourEnrichmentResult(completed_track.track_uuid, str(row.get("vehicle_track_id", persisted_vehicle_track_id)), "analytics_supabase", True, extracted)

    def _resolve_source_storage_uri(self, completed_track: LocalVehicleTrack, persisted_vehicle_track_id: str) -> str:
        if self.persistence_config.backend == "analytics_supabase" and self.track_media_repository is not None:
            row = self.track_media_repository.get_existing(
                vehicle_track_id=persisted_vehicle_track_id,
                media_type=self.config.source_media_type,
            )
            if row is not None and row.get("storage_uri"):
                return str(row["storage_uri"])
        evidence = completed_track.evidence_package
        if evidence is None or "best_overall" not in evidence.candidates:
            raise FileNotFoundError(f"No BEST_OVERALL evidence found for track {completed_track.track_uuid}")
        file_path = Path(evidence.candidates["best_overall"].file_path or "")
        relative = file_path.resolve().relative_to(self.artifact_root)
        return relative.as_posix()

    def _build_record(
        self,
        *,
        completed_track: LocalVehicleTrack,
        persisted_vehicle_track_id: str,
        result: VehicleColourResult,
    ) -> VehicleAttributeRecord:
        return VehicleAttributeRecord(
            vehicle_track_id=persisted_vehicle_track_id,
            primary_color=result.canonical_colour,
            secondary_color=result.secondary_colour,
            color_confidence=result.confidence,
            vehicle_class=completed_track.class_name,
            attribute_source="FLORENCE_VEHICLE_COLOUR",
            attribute_status="CURRENT",
            observation_count=1,
            metadata={
                "backend": result.backend,
                "status": result.status,
                "source_media_type": result.source_media_type,
                "source_storage_uri": result.source_storage_uri,
                "track_uuid": completed_track.track_uuid,
                "camera_code": completed_track.camera_code,
                "raw_output": result.raw_output[:1000],
            },
        )

    def _apply_extraction_metrics(self, result: VehicleColourResult) -> None:
        if result.status == "SUCCESS":
            self.metrics.colour_extractions_succeeded += 1
        elif result.status == "LOW_CONFIDENCE":
            self.metrics.colour_extractions_low_confidence += 1
        elif result.status == "IMAGE_MISSING":
            self.metrics.colour_images_missing += 1
            self.metrics.colour_extractions_failed += 1
        elif result.status == "PARSE_ERROR":
            self.metrics.colour_parse_failures += 1
            self.metrics.colour_extractions_failed += 1
        elif result.status in {"MODEL_ERROR", "IMAGE_INVALID"}:
            self.metrics.colour_extractions_failed += 1
        else:
            self.metrics.colour_extractions_unknown += 1
