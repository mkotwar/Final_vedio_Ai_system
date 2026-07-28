from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase, AnalyticsRepositoryError
from .persistence_models import TrackMediaRecord


@dataclass(frozen=True, slots=True)
class TrackMediaBatchResult:
    attempted: int
    inserted: int
    already_existing: int
    failed: int
    validated: int = 0
    missing_files: int = 0
    failed_records: list[dict[str, Any]] | None = None


class TrackMediaRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="track_media")

    def get_existing(self, *, vehicle_track_id: str, media_type: str, storage_uri: str | None = None) -> dict[str, Any] | None:
        try:
            query = self._table().select("*").eq("vehicle_track_id", vehicle_track_id).eq("media_type", media_type)
            response = query.execute()
        except Exception as exc:
            raise self._wrap_error(operation="get_existing", message=f"Failed to query analytics.track_media: {exc}", cause=exc) from exc
        rows = self._extract_rows(response)
        if storage_uri is None:
            return rows[0] if rows else None
        for row in rows:
            if str(row.get("storage_uri")) == storage_uri:
                return row
        return None

    def upsert(self, record: TrackMediaRecord) -> dict[str, Any]:
        existing = self.get_existing(vehicle_track_id=record.vehicle_track_id, media_type=record.media_type, storage_uri=record.storage_uri)
        if existing is not None:
            return existing
        try:
            response = self._table().insert(record.to_payload()).execute()
        except Exception as exc:
            raise self._wrap_error(operation="upsert", message=f"Failed to insert analytics.track_media: {exc}", cause=exc) from exc
        return self._expect_one(response, operation="upsert")

    def bulk_upsert(self, records: Sequence[TrackMediaRecord]) -> TrackMediaBatchResult:
        validated = [record.to_payload() for record in records]
        if not validated:
            return TrackMediaBatchResult(attempted=0, inserted=0, already_existing=0, failed=0)
        vehicle_track_ids = sorted({str(record["vehicle_track_id"]) for record in validated})
        media_types = sorted({str(record["media_type"]) for record in validated})
        try:
            query = self._table().select("*")
            if hasattr(query, "in_"):
                response = query.in_("vehicle_track_id", vehicle_track_ids).in_("media_type", media_types).execute()
            else:
                response = query.execute()
        except Exception as exc:
            raise self._wrap_error(operation="bulk_upsert_prefetch", message=f"Failed to prefetch analytics.track_media: {exc}", cause=exc) from exc
        existing_rows = self._extract_rows(response)
        existing_keys = {
            (str(row.get("vehicle_track_id")), str(row.get("media_type")), str(row.get("storage_uri")))
            for row in existing_rows
        }
        to_insert = [
            payload
            for payload in validated
            if (str(payload["vehicle_track_id"]), str(payload["media_type"]), str(payload["storage_uri"])) not in existing_keys
        ]
        already_existing = len(validated) - len(to_insert)
        if not to_insert:
            return TrackMediaBatchResult(attempted=len(validated), inserted=0, already_existing=already_existing, failed=0)
        try:
            response = self._table().insert(to_insert).execute()
        except Exception as exc:
            raise self._wrap_error(operation="bulk_upsert_insert", message=f"Failed to insert analytics.track_media batch: {exc}", cause=exc) from exc
        inserted_rows = self._extract_rows(response)
        inserted = len(inserted_rows)
        failed = len(to_insert) - inserted
        return TrackMediaBatchResult(
            attempted=len(validated),
            inserted=inserted,
            already_existing=already_existing,
            failed=failed,
            failed_records=[],
        )
