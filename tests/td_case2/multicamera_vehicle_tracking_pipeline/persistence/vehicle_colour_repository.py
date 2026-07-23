from __future__ import annotations

from typing import Any

from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase
from .persistence_models import VehicleAttributeRecord


class VehicleColourRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="vehicle_attribute")

    def get_current_by_track_and_source(self, *, vehicle_track_id: str, attribute_source: str) -> dict[str, Any] | None:
        try:
            response = (
                self._table()
                .select("*")
                .eq("vehicle_track_id", vehicle_track_id)
                .eq("attribute_source", attribute_source)
                .eq("attribute_status", "CURRENT")
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise self._wrap_error(
                operation="get_current_by_track_and_source",
                message=f"Failed to query analytics.vehicle_attribute: {exc}",
                cause=exc,
            ) from exc
        rows = self._extract_rows(response)
        return rows[0] if rows else None

    def upsert_vehicle_colour(self, record: VehicleAttributeRecord) -> dict[str, Any]:
        existing = None
        if record.attribute_source is not None:
            existing = self.get_current_by_track_and_source(
                vehicle_track_id=record.vehicle_track_id,
                attribute_source=record.attribute_source,
            )
        if existing is None:
            try:
                response = self._table().insert(record.to_payload()).execute()
            except Exception as exc:
                raise self._wrap_error(
                    operation="insert_vehicle_colour",
                    message=f"Failed to insert analytics.vehicle_attribute: {exc}",
                    cause=exc,
                ) from exc
            return self._expect_one(response, operation="insert_vehicle_colour")
        try:
            response = self._table().update(record.to_payload()).eq("id", str(existing["id"])).execute()
        except Exception as exc:
            raise self._wrap_error(
                operation="update_vehicle_colour",
                message=f"Failed to update analytics.vehicle_attribute: {exc}",
                cause=exc,
            ) from exc
        return self._expect_one(response, operation="update_vehicle_colour")

