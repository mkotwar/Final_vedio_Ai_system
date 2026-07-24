from __future__ import annotations

from datetime import datetime
from typing import Any

from ..cross_camera.global_match_models import GlobalObjectMembership, GlobalVehicleObjectProposal
from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase


def _response_rows(response: object) -> list[dict[str, Any]]:
    rows = getattr(response, "data", None)
    if rows is None:
        return []
    return [dict(item) for item in rows]


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


class GlobalVehicleObjectRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="global_vehicle")

    def find_global_object_for_track(self, track_id: str) -> dict[str, Any] | None:
        membership_rows = _response_rows(
            self.client.table("global_vehicle_track").select("*").eq("vehicle_track_id", track_id).eq("is_current", True).limit(1).execute()
        )
        if not membership_rows:
            return None
        global_vehicle_id = membership_rows[0].get("global_vehicle_id")
        if not global_vehicle_id:
            return None
        rows = _response_rows(self._table().select("*").eq("id", global_vehicle_id).limit(1).execute())
        return rows[0] if rows else None

    def list_global_objects_for_run(self, processing_run_id: str) -> list[dict[str, Any]]:
        return _response_rows(self._table().select("*").eq("processing_run_id", processing_run_id).execute())

    def find_objects_by_verified_plate(self, processing_run_id: str, plate: str) -> list[dict[str, Any]]:
        return _response_rows(self._table().select("*").eq("processing_run_id", processing_run_id).eq("canonical_plate", plate).execute())

    def create_or_get_global_object(self, proposal: GlobalVehicleObjectProposal) -> dict[str, Any]:
        existing = _response_rows(
            self._table().select("*").eq("processing_run_id", proposal.processing_run_id).eq("global_vehicle_code", proposal.global_object_code).limit(1).execute()
        )
        if existing:
            return existing[0]
        payload = {
            "processing_run_id": proposal.processing_run_id,
            "global_vehicle_code": proposal.global_object_code,
            "object_type": "VEHICLE",
            "canonical_plate": proposal.canonical_plate,
            "canonical_color": proposal.canonical_colour,
            "canonical_vehicle_class": proposal.canonical_vehicle_class,
            "first_seen_at": _serialize_datetime(proposal.first_seen_at),
            "last_seen_at": _serialize_datetime(proposal.last_seen_at),
            "identity_confidence": proposal.confidence,
            "status": proposal.status,
            "camera_count": proposal.camera_count,
            "track_count": proposal.track_count,
            "creation_method": proposal.creation_method,
            "metadata": proposal.metadata,
        }
        response = self._table().insert(payload).execute()
        rows = _response_rows(response)
        return rows[0] if rows else payload

    def add_or_update_member(self, global_vehicle_id: str, membership: GlobalObjectMembership) -> dict[str, Any]:
        current_rows = _response_rows(
            self.client.table("global_vehicle_track").select("*").eq("vehicle_track_id", membership.vehicle_track_id).eq("is_current", True).execute()
        )
        for row in current_rows:
            if str(row.get("global_vehicle_id")) != global_vehicle_id:
                raise RuntimeError(f"Track {membership.track_uuid} already belongs to another active global object: {row.get('global_vehicle_id')}")
        existing = _response_rows(
            self.client.table("global_vehicle_track")
            .select("*")
            .eq("global_vehicle_id", global_vehicle_id)
            .eq("vehicle_track_id", membership.vehicle_track_id)
            .limit(1)
            .execute()
        )
        payload = {
            "global_vehicle_id": global_vehicle_id,
            "vehicle_track_id": membership.vehicle_track_id,
            "association_score": membership.membership_confidence,
            "association_method": membership.match_method,
            "association_status": membership.membership_status,
            "is_current": membership.membership_status != "REJECTED",
            "metadata": membership.metadata,
        }
        if existing:
            response = self.client.table("global_vehicle_track").update(payload).eq("id", existing[0].get("id")).execute()
        else:
            response = self.client.table("global_vehicle_track").insert(payload).execute()
        rows = _response_rows(response)
        return rows[0] if rows else payload

    def list_memberships_for_run(self, processing_run_id: str) -> list[dict[str, Any]]:
        object_ids = [row.get("id") for row in self.list_global_objects_for_run(processing_run_id) if row.get("id")]
        if not object_ids:
            return []
        return _response_rows(self.client.table("global_vehicle_track").select("*").in_("global_vehicle_id", object_ids).execute())
