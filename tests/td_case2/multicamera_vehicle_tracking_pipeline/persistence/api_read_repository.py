from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase, AnalyticsRepositoryError
from .track_media_types import (
    ANNOTATED_FULL_FRAME_MEDIA_TYPE_PRIORITY,
    FULL_FRAME_MEDIA_TYPE_PRIORITY,
    PLATE_MEDIA_TYPE_PRIORITY,
    VEHICLE_MEDIA_TYPE_PRIORITY,
)


@dataclass(frozen=True, slots=True)
class Page:
    items: list[dict[str, Any]]
    page: int
    page_size: int
    total: int

    @property
    def has_next(self) -> bool:
        return (self.page * self.page_size) < self.total


def _response_rows(response: object) -> list[dict[str, Any]]:
    rows = getattr(response, "data", None)
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise AnalyticsRepositoryError(
            operation="extract_rows",
            table_name="unknown",
            message="Expected list response from analytics query.",
        )
    return [dict(item) for item in rows]


def _maybe_upper(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized.upper() or None


def _plate_display_text(text: str | None, status: str | None) -> str:
    normalized_status = _maybe_upper(status) or "UNKNOWN"
    normalized_text = str(text or "").strip().upper() or None
    if normalized_status == "PARTIAL":
        if not normalized_text:
            return "Partial plate"
        return f"...{normalized_text[-4:]}" if len(normalized_text) > 4 else normalized_text
    if normalized_text:
        return normalized_text
    if normalized_status == "UNREADABLE":
        return "Unreadable"
    if normalized_status == "NO_PLATE_DETECTED":
        return "Not detected"
    if normalized_status == "PARTIAL":
        return "Partial plate"
    return "No plate result"


PLATE_STATUS_PRIORITY = {
    "VERIFIED": 0,
    "PROBABLE": 1,
    "PARTIAL": 2,
    "UNREADABLE": 3,
    "NO_PLATE_DETECTED": 4,
    "UNKNOWN": 5,
}


class AnalyticsReadRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="processing_run")

    def health(self) -> dict[str, str]:
        try:
            self.client.table("processing_run").select("id", count="exact").limit(1).execute()
        except (AnalyticsRepositoryError, httpx.HTTPError):
            return {
                "status": "degraded",
                "service": "multicamera-vehicle-api",
                "database": "unreachable",
                "schema": self.client.schema_name,
            }
        return {
            "status": "ok",
            "service": "multicamera-vehicle-api",
            "database": "reachable",
            "schema": self.client.schema_name,
        }

    def find_run_by_code(self, run_code: str) -> dict[str, Any] | None:
        rows = _response_rows(
            self.client.table("processing_run")
            .select("*")
            .eq("run_code", run_code)
            .limit(1)
            .execute()
        )
        return rows[0] if rows else None

    def list_runs(
        self,
        *,
        page: int,
        page_size: int,
        status: str | None = None,
        run_code: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Page:
        query = self.client.table("processing_run").select(
            "id,run_code,status,started_at,completed_at,created_at,active_camera_count,total_tracks",
            count="exact",
        )
        if status:
            query = query.eq("status", status)
        if run_code:
            query = query.ilike("run_code", f"%{run_code}%")
        query = query.order(sort_by, desc=sort_order == "desc").range((page - 1) * page_size, page * page_size - 1)
        response = query.execute()
        rows = _response_rows(response)
        totals = self._count_related("processing_error", "processing_run_id", [str(row["id"]) for row in rows])
        global_totals = self._count_related("global_vehicle", "processing_run_id", [str(row["id"]) for row in rows])
        items = [
            {
                "id": row.get("id"),
                "run_code": row.get("run_code"),
                "status": row.get("status"),
                "started_at": row.get("started_at"),
                "completed_at": row.get("completed_at"),
                "created_at": row.get("created_at"),
                "camera_count": row.get("active_camera_count") or 0,
                "track_count": row.get("total_tracks") or 0,
                "global_vehicle_count": global_totals.get(str(row.get("id")), 0),
                "processing_error_count": totals.get(str(row.get("id")), 0),
            }
            for row in rows
        ]
        return Page(items=items, page=page, page_size=page_size, total=int(getattr(response, "count", 0) or 0))

    def get_run_detail(self, run_code: str) -> dict[str, Any] | None:
        run_row = self.find_run_by_code(run_code)
        if run_row is None:
            return None
        run_id = str(run_row["id"])
        camera_runs = _response_rows(
            self.client.table("camera_run")
            .select("id,camera_id,status,frames_read,frames_processed,detections_count,completed_tracks_count,discarded_tracks_count")
            .eq("processing_run_id", run_id)
            .execute()
        )
        tracks = _response_rows(
            self.client.table("vehicle_track")
            .select("id,observation_count")
            .eq("processing_run_id", run_id)
            .execute()
        )
        track_ids = [str(track["id"]) for track in tracks]
        attributes = self._count_for_ids("vehicle_attribute", "vehicle_track_id", track_ids)
        summaries = self._count_for_ids("plate_summary", "vehicle_track_id", track_ids)
        media = self._count_for_ids("track_media", "vehicle_track_id", track_ids)
        errors = self._count_for_ids("processing_error", "processing_run_id", [run_id], target_key="processing_run_id")
        globals_count = self._count_for_ids("global_vehicle", "processing_run_id", [run_id], target_key="processing_run_id")
        return {
            "id": run_row.get("id"),
            "run_code": run_row.get("run_code"),
            "status": run_row.get("status"),
            "started_at": run_row.get("started_at"),
            "completed_at": run_row.get("completed_at"),
            "created_at": run_row.get("created_at"),
            "pipeline_name": run_row.get("pipeline_name"),
            "pipeline_version": run_row.get("pipeline_version"),
            "execution_mode": run_row.get("execution_mode"),
            "runtime_device": run_row.get("runtime_device"),
            "camera_summary": {
                "configured_camera_count": run_row.get("configured_camera_count") or 0,
                "active_camera_count": run_row.get("active_camera_count") or 0,
                "camera_run_count": len(camera_runs),
                "completed_camera_runs": len([row for row in camera_runs if row.get("status") == "COMPLETED"]),
            },
            "track_summary": {
                "track_count": run_row.get("total_tracks") or len(tracks),
                "total_track_observations": run_row.get("total_track_observations") or 0,
            },
            "enrichment_summary": {
                "tracks_with_colour": attributes.get(run_id, 0) if False else len(attributes),
                "tracks_with_plate_summary": len(summaries),
                "tracks_with_media": len(media),
            },
            "global_object_summary": {
                "global_vehicle_count": globals_count.get(run_id, 0),
            },
            "processing_error_summary": {
                "processing_error_count": errors.get(run_id, 0),
            },
        }

    def list_run_cameras(
        self,
        *,
        run_code: str,
        page: int,
        page_size: int,
        status: str | None = None,
        camera_code: str | None = None,
        sort_by: str = "camera_code",
        sort_order: str = "asc",
    ) -> tuple[dict[str, Any] | None, Page]:
        run_row = self.find_run_by_code(run_code)
        if run_row is None:
            return None, Page(items=[], page=page, page_size=page_size, total=0)
        camera_rows = _response_rows(self.client.table("camera").select("id,camera_code,camera_name,location_name").execute())
        camera_by_id = {str(row["id"]): row for row in camera_rows if row.get("id")}
        query = self.client.table("camera_run").select(
            "id,camera_id,status,frames_read,frames_processed,detections_count,completed_tracks_count,discarded_tracks_count",
            count="exact",
        ).eq("processing_run_id", str(run_row["id"]))
        if status:
            query = query.eq("status", status)
        response = query.execute()
        rows = _response_rows(response)
        items: list[dict[str, Any]] = []
        for row in rows:
            camera = camera_by_id.get(str(row.get("camera_id")), {})
            item = {
                "id": camera.get("id"),
                "camera_code": camera.get("camera_code"),
                "camera_name": camera.get("camera_name"),
                "location": camera.get("location_name"),
                "camera_run_status": row.get("status"),
                "frames_read": row.get("frames_read") or 0,
                "frames_processed": row.get("frames_processed") or 0,
                "detection_count": row.get("detections_count") or 0,
                "completed_track_count": row.get("completed_tracks_count") or 0,
                "discarded_track_count": row.get("discarded_tracks_count") or 0,
            }
            if camera_code and item["camera_code"] != camera_code:
                continue
            items.append(item)
        reverse = sort_order == "desc"
        items.sort(key=lambda item: (item.get(sort_by) is None, item.get(sort_by)), reverse=reverse)
        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        return run_row, Page(items=items[start:end], page=page, page_size=page_size, total=total)

    def get_camera_in_run(self, run_code: str, camera_code: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        run_row = self.find_run_by_code(run_code)
        if run_row is None:
            return None, None
        camera_rows = _response_rows(
            self.client.table("camera").select("*").eq("camera_code", camera_code).limit(1).execute()
        )
        if not camera_rows:
            return run_row, None
        camera_row = camera_rows[0]
        camera_run_rows = _response_rows(
            self.client.table("camera_run")
            .select("*")
            .eq("processing_run_id", str(run_row["id"]))
            .eq("camera_id", str(camera_row["id"]))
            .limit(1)
            .execute()
        )
        if not camera_run_rows:
            return run_row, None
        camera_run = camera_run_rows[0]
        tracks = _response_rows(
            self.client.table("vehicle_track")
            .select("id")
            .eq("processing_run_id", str(run_row["id"]))
            .eq("camera_id", str(camera_row["id"]))
            .execute()
        )
        track_ids = [str(track["id"]) for track in tracks]
        attribute_ids = self._count_for_ids("vehicle_attribute", "vehicle_track_id", track_ids)
        plate_ids = self._count_for_ids("plate_summary", "vehicle_track_id", track_ids)
        media_ids = self._count_for_ids("track_media", "vehicle_track_id", track_ids)
        errors = _response_rows(
            self.client.table("processing_error")
            .select("id,stage_name,error_code,message,created_at")
            .eq("processing_run_id", str(run_row["id"]))
            .eq("camera_run_id", str(camera_run["id"]))
            .execute()
        )
        return run_row, {
            "id": camera_row.get("id"),
            "camera_code": camera_row.get("camera_code"),
            "camera_name": camera_row.get("camera_name"),
            "location": camera_row.get("location_name"),
            "camera_run_status": camera_run.get("status"),
            "frames_read": camera_run.get("frames_read") or 0,
            "frames_processed": camera_run.get("frames_processed") or 0,
            "detection_count": camera_run.get("detections_count") or 0,
            "completed_track_count": camera_run.get("completed_tracks_count") or 0,
            "discarded_track_count": camera_run.get("discarded_tracks_count") or 0,
            "track_count": len(track_ids),
            "colour_coverage": len(attribute_ids),
            "plate_coverage": len(plate_ids),
            "media_coverage": len(media_ids),
            "processing_errors": errors,
        }

    def list_tracks(
        self,
        *,
        run_code: str,
        page: int,
        page_size: int,
        camera_code: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        plate: str | None = None,
        plate_status: str | None = None,
        lifecycle_state: str | None = None,
        minimum_confidence: float | None = None,
        has_media: bool | None = None,
        sort_by: str = "first_seen_at",
        sort_order: str = "desc",
    ) -> tuple[dict[str, Any] | None, Page]:
        run_row = self.find_run_by_code(run_code)
        if run_row is None:
            return None, Page(items=[], page=page, page_size=page_size, total=0)
        run_id = str(run_row["id"])
        query = self.client.table("vehicle_track").select("*", count="exact").eq("processing_run_id", run_id)
        if vehicle_class:
            query = query.eq("vehicle_class", vehicle_class)
        if lifecycle_state:
            query = query.eq("lifecycle_state", lifecycle_state)
        if minimum_confidence is not None:
            query = query.gte("best_detection_confidence", minimum_confidence)

        if camera_code:
            camera_rows = _response_rows(self.client.table("camera").select("id").eq("camera_code", camera_code).limit(1).execute())
            if not camera_rows:
                return run_row, Page(items=[], page=page, page_size=page_size, total=0)
            query = query.eq("camera_id", str(camera_rows[0]["id"]))

        filtered_track_ids = self._filtered_track_ids_by_enrichment(
            run_id=run_id,
            colour=colour,
            plate=plate,
            plate_status=plate_status,
            has_media=has_media,
        )
        if filtered_track_ids is not None:
            if not filtered_track_ids:
                return run_row, Page(items=[], page=page, page_size=page_size, total=0)
            query = query.in_("id", filtered_track_ids)

        query = query.order(sort_by, desc=sort_order == "desc").range((page - 1) * page_size, page * page_size - 1)
        response = query.execute()
        rows = _response_rows(response)
        page_items = self._decorate_tracks(rows)
        return run_row, Page(page=page, page_size=page_size, total=int(getattr(response, "count", 0) or 0), items=page_items)

    def get_track_by_uuid(self, track_uuid: str) -> dict[str, Any] | None:
        rows = _response_rows(self.client.table("vehicle_track").select("*").eq("track_uuid", track_uuid).limit(1).execute())
        if not rows:
            return None
        track = rows[0]
        track_id = str(track["id"])
        decorated = self._decorate_tracks([track])[0]
        camera = self._camera_by_id([str(track.get("camera_id"))]).get(str(track.get("camera_id")), {})
        observations = _response_rows(
            self.client.table("track_observation")
            .select("id,frame_number,observed_at,video_time_seconds,detection_confidence,tracker_confidence,is_key_observation,metadata,bbox_x1,bbox_y1,bbox_x2,bbox_y2")
            .eq("vehicle_track_id", track_id)
            .order("frame_number")
            .execute()
        )
        media = self.list_track_media(track_uuid)
        memberships = _response_rows(
            self.client.table("global_vehicle_track")
            .select("*")
            .eq("vehicle_track_id", track_id)
            .eq("is_current", True)
            .execute()
        )
        global_membership = self._to_global_membership(memberships[0]) if memberships else None
        errors = _response_rows(
            self.client.table("processing_error")
            .select("id,stage_name,error_code,message,severity,created_at")
            .eq("vehicle_track_id", track_id)
            .execute()
        )
        match_rows = _response_rows(
            self.client.table("cross_camera_match")
            .select("*")
            .or_(f"source_track_id.eq.{track_id},candidate_track_id.eq.{track_id}")
            .execute()
        )
        return {
            "track": decorated,
            "camera": {
                "camera_code": camera.get("camera_code"),
                "camera_name": camera.get("camera_name"),
                "location": camera.get("location_name"),
            },
            "colour": {
                "primary_colour": decorated.get("primary_colour"),
                "colour_confidence": decorated.get("colour_confidence"),
            },
            "plate": {
                "plate_result": decorated.get("plate_result"),
                "canonical_plate": decorated.get("canonical_plate"),
                "plate_status": decorated.get("plate_status"),
                "plate_confidence": decorated.get("plate_confidence"),
            },
            "media": media,
            "observation_summary": {
                "count": len(observations),
                "first_frame": observations[0]["frame_number"] if observations else None,
                "last_frame": observations[-1]["frame_number"] if observations else None,
                "key_observation_count": len([row for row in observations if row.get("is_key_observation")]),
            },
            "class_diagnostics": self._class_diagnostics_from_track(track),
            "global_membership": global_membership,
            "cross_camera_matches": match_rows,
            "errors": errors,
        }

    def list_track_observations(
        self,
        *,
        track_uuid: str,
        page: int,
        page_size: int,
        key_only: bool = False,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sort_order: str = "asc",
    ) -> tuple[dict[str, Any] | None, Page]:
        track = self.get_track_row(track_uuid)
        if track is None:
            return None, Page(items=[], page=page, page_size=page_size, total=0)
        query = self.client.table("track_observation").select(
            "id,frame_number,observed_at,video_time_seconds,bbox_x1,bbox_y1,bbox_x2,bbox_y2,detection_confidence,tracker_confidence,is_key_observation,metadata",
            count="exact",
        ).eq("vehicle_track_id", str(track["id"]))
        if key_only:
            query = query.eq("is_key_observation", True)
        if start_frame is not None:
            query = query.gte("frame_number", start_frame)
        if end_frame is not None:
            query = query.lte("frame_number", end_frame)
        query = query.order("frame_number", desc=sort_order == "desc").range((page - 1) * page_size, page * page_size - 1)
        response = query.execute()
        rows = _response_rows(response)
        items = [
            {
                "frame_number": row.get("frame_number"),
                "timestamp": row.get("observed_at"),
                "video_time_seconds": row.get("video_time_seconds"),
                "bbox": {
                    "x1": row.get("bbox_x1"),
                    "y1": row.get("bbox_y1"),
                    "x2": row.get("bbox_x2"),
                    "y2": row.get("bbox_y2"),
                },
                "detection_confidence": row.get("detection_confidence"),
                "tracker_confidence": row.get("tracker_confidence"),
                "is_key_observation": row.get("is_key_observation"),
                "class_name": self._observation_class_name(row),
                "raw_class_name": self._observation_raw_class_name(row),
            }
            for row in rows
        ]
        return track, Page(items=items, page=page, page_size=page_size, total=int(getattr(response, "count", 0) or 0))

    def list_global_vehicles(
        self,
        *,
        page: int,
        page_size: int,
        run_code: str | None = None,
        status: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        plate: str | None = None,
        minimum_confidence: float | None = None,
        minimum_camera_count: int | None = None,
        sort_by: str = "first_seen_at",
        sort_order: str = "desc",
    ) -> Page:
        query = self.client.table("global_vehicle").select("*", count="exact")
        run_lookup: dict[str, str] = {}
        if run_code:
            run = self.find_run_by_code(run_code)
            if run is None:
                return Page(items=[], page=page, page_size=page_size, total=0)
            query = query.eq("processing_run_id", str(run["id"]))
        if status:
            query = query.eq("status", status)
        if vehicle_class:
            query = query.eq("canonical_vehicle_class", vehicle_class)
        if colour:
            query = query.eq("canonical_color", colour)
        if plate:
            query = query.ilike("canonical_plate", f"%{plate}%")
        if minimum_confidence is not None:
            query = query.gte("identity_confidence", minimum_confidence)
        if minimum_camera_count is not None:
            query = query.gte("camera_count", minimum_camera_count)
        query = query.order(sort_by, desc=sort_order == "desc").range((page - 1) * page_size, page * page_size - 1)
        response = query.execute()
        rows = _response_rows(response)
        run_ids = [str(row.get("processing_run_id")) for row in rows if row.get("processing_run_id")]
        if run_ids:
            run_rows = _response_rows(self.client.table("processing_run").select("id,run_code").in_("id", run_ids).execute())
            run_lookup = {str(row["id"]): str(row.get("run_code")) for row in run_rows if row.get("id")}
        evidence_lookup = self._global_vehicle_primary_evidence([str(row["id"]) for row in rows if row.get("id")])
        member_lookup = self._global_vehicle_members_by_id([str(row["id"]) for row in rows if row.get("id")])
        plate_context = self._plate_context_by_track_id(
            [
                str(member["vehicle_track_id"])
                for members in member_lookup.values()
                for member in members
                if member.get("vehicle_track_id")
            ]
        )
        items = [
            {
                "global_vehicle_code": row.get("global_vehicle_code"),
                "run_code": run_lookup.get(str(row.get("processing_run_id"))),
                "status": row.get("status"),
                "plate_result": self._global_plate_result(
                    global_row=row,
                    members=member_lookup.get(str(row.get("id")), []),
                    plate_context_by_track_id=plate_context,
                ),
                "canonical_plate": row.get("canonical_plate"),
                "canonical_colour": row.get("canonical_color"),
                "canonical_vehicle_class": row.get("canonical_vehicle_class"),
                "confidence": row.get("identity_confidence"),
                "camera_count": row.get("camera_count"),
                "track_count": row.get("track_count"),
                "creation_method": row.get("creation_method"),
                "first_seen_at": row.get("first_seen_at"),
                "last_seen_at": row.get("last_seen_at"),
                "primary_evidence_reference": evidence_lookup.get(str(row.get("id")), {}).get("primary_vehicle_media"),
                "primary_vehicle_media": evidence_lookup.get(str(row.get("id")), {}).get("primary_vehicle_media"),
                "primary_plate_media": evidence_lookup.get(str(row.get("id")), {}).get("primary_plate_media"),
            }
            for row in rows
        ]
        return Page(items=items, page=page, page_size=page_size, total=int(getattr(response, "count", 0) or 0))

    def get_global_vehicle_by_code(self, global_vehicle_code: str) -> dict[str, Any] | None:
        rows = _response_rows(
            self.client.table("global_vehicle").select("*").eq("global_vehicle_code", global_vehicle_code).limit(1).execute()
        )
        if not rows:
            return None
        vehicle = rows[0]
        members = self.list_global_vehicle_members(global_vehicle_code)
        member_track_ids = [str(member["vehicle_track_id"]) for member in members if member.get("vehicle_track_id")]
        plate_context = self._plate_context_by_track_id(member_track_ids)
        matches = _response_rows(
            self.client.table("cross_camera_match").select("*").eq("created_global_vehicle_id", str(vehicle["id"])).execute()
        )
        evidence = self._media_for_tracks(member_track_ids)
        evidence_lookup = self._global_vehicle_primary_evidence([str(vehicle["id"])])
        primary_media_bundle = evidence_lookup.get(str(vehicle["id"]), {})
        return {
            "global_vehicle": {
                "global_vehicle_code": vehicle.get("global_vehicle_code"),
                "run_code": self._run_code_for_id(str(vehicle.get("processing_run_id"))),
                "status": vehicle.get("status"),
                "plate_result": self._global_plate_result(
                    global_row=vehicle,
                    members=members,
                    plate_context_by_track_id=plate_context,
                ),
                "canonical_plate": vehicle.get("canonical_plate"),
                "canonical_colour": vehicle.get("canonical_color"),
                "canonical_vehicle_class": vehicle.get("canonical_vehicle_class"),
                "confidence": vehicle.get("identity_confidence"),
                "camera_count": vehicle.get("camera_count"),
                "track_count": vehicle.get("track_count"),
                "creation_method": vehicle.get("creation_method"),
                "first_seen_at": vehicle.get("first_seen_at"),
                "last_seen_at": vehicle.get("last_seen_at"),
                "primary_vehicle_media": primary_media_bundle.get("primary_vehicle_media"),
                "primary_plate_media": primary_media_bundle.get("primary_plate_media"),
            },
            "members": members,
            "camera_sequence": [
                {
                    "camera_code": member.get("camera_code"),
                    "track_uuid": member.get("track_uuid"),
                    "first_seen_at": member.get("first_seen_at"),
                    "last_seen_at": member.get("last_seen_at"),
                }
                for member in members
            ],
            "confirmed_matches": [match for match in matches if _maybe_upper(match.get("decision")) == "CONFIRMED"],
            "possible_matches": [match for match in matches if _maybe_upper(match.get("decision")) != "CONFIRMED"],
            "evidence": evidence,
        }

    def list_global_vehicle_members(self, global_vehicle_code: str) -> list[dict[str, Any]]:
        vehicle_rows = _response_rows(
            self.client.table("global_vehicle").select("id").eq("global_vehicle_code", global_vehicle_code).limit(1).execute()
        )
        if not vehicle_rows:
            return []
        memberships = _response_rows(
            self.client.table("global_vehicle_track")
            .select("*")
            .eq("global_vehicle_id", str(vehicle_rows[0]["id"]))
            .order("attached_at")
            .execute()
        )
        track_ids = [str(row["vehicle_track_id"]) for row in memberships if row.get("vehicle_track_id")]
        tracks = self._track_rows_by_id(track_ids)
        cameras = self._camera_by_id([str(track.get("camera_id")) for track in tracks.values()])
        attributes = self._current_attributes_by_track_id(track_ids)
        plate_context = self._plate_context_by_track_id(track_ids)
        media_lookup = self._primary_media_bundle_by_track_id(track_ids)
        rows: list[dict[str, Any]] = []
        for membership in memberships:
            track_id = str(membership.get("vehicle_track_id"))
            track = tracks.get(track_id, {})
            camera = cameras.get(str(track.get("camera_id")), {})
            attribute = attributes.get(track_id, {})
            plate_info = plate_context.get(track_id, {})
            summary = plate_info.get("summary", {})
            media = media_lookup.get(track_id, {})
            rows.append(
                {
                    "vehicle_track_id": membership.get("vehicle_track_id"),
                    "track_uuid": track.get("track_uuid"),
                    "camera_code": camera.get("camera_code"),
                    "vehicle_class": track.get("vehicle_class"),
                    "primary_colour": attribute.get("primary_color"),
                    "plate_result": plate_info.get("plate_result"),
                    "canonical_plate": summary.get("canonical_plate"),
                    "plate_status": summary.get("status"),
                    "plate_confidence": (plate_info.get("plate_result") or {}).get("ocr_confidence", summary.get("confidence")),
                    "best_detection_confidence": track.get("best_detection_confidence"),
                    "first_seen_at": track.get("first_seen_at"),
                    "last_seen_at": track.get("last_seen_at"),
                    "association_score": membership.get("association_score"),
                    "association_method": membership.get("association_method"),
                    "association_status": membership.get("association_status"),
                    "is_current": membership.get("is_current"),
                    "attached_at": membership.get("attached_at"),
                    "primary_vehicle_media": self._to_media_reference(media.get("primary_vehicle_media")) if media.get("primary_vehicle_media") else None,
                    "primary_plate_media": self._to_media_reference(media.get("primary_plate_media")) if media.get("primary_plate_media") else None,
                }
            )
        return rows

    def list_cross_camera_matches(
        self,
        *,
        page: int,
        page_size: int,
        run_code: str | None = None,
        decision: str | None = None,
        rule_version: str | None = None,
        minimum_score: float | None = None,
        camera_code: str | None = None,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> Page:
        query = self.client.table("cross_camera_match").select("*", count="exact")
        if run_code:
            run = self.find_run_by_code(run_code)
            if run is None:
                return Page(items=[], page=page, page_size=page_size, total=0)
            query = query.eq("processing_run_id", str(run["id"]))
        if decision:
            query = query.eq("decision", decision)
        if rule_version:
            query = query.eq("rule_version", rule_version)
        if minimum_score is not None:
            query = query.gte("overall_score", minimum_score)
        if camera_code:
            track_rows = _response_rows(
                self.client.table("vehicle_track")
                .select("id,camera_id")
                .execute()
            )
            camera_rows = _response_rows(self.client.table("camera").select("id,camera_code").eq("camera_code", camera_code).limit(1).execute())
            if not camera_rows:
                return Page(items=[], page=page, page_size=page_size, total=0)
            camera_id = str(camera_rows[0]["id"])
            track_ids = [str(row["id"]) for row in track_rows if str(row.get("camera_id")) == camera_id]
            if not track_ids:
                return Page(items=[], page=page, page_size=page_size, total=0)
            query = query.or_(f"source_track_id.in.({','.join(track_ids)}),candidate_track_id.in.({','.join(track_ids)})")
        query = query.order(sort_by, desc=sort_order == "desc").range((page - 1) * page_size, page * page_size - 1)
        response = query.execute()
        rows = _response_rows(response)
        items = self._decorate_matches(rows)
        return Page(items=items, page=page, page_size=page_size, total=int(getattr(response, "count", 0) or 0))

    def get_cross_camera_match(self, match_id: str) -> dict[str, Any] | None:
        rows = _response_rows(
            self.client.table("cross_camera_match").select("*").eq("id", match_id).limit(1).execute()
        )
        if not rows:
            return None
        return self._decorate_matches(rows)[0]

    def search_local_tracks(
        self,
        *,
        run_code: str | None,
        vehicle_class: str | None,
        colour: str | None,
        plate: str | None,
        plate_match_type: str,
        camera_codes: list[str],
        window_start: str | None,
        window_end: str | None,
        minimum_confidence: float | None,
        verified_plate_only: bool,
        sort_by: str,
        sort_order: str,
        fetch_limit: int,
    ) -> Page:
        run_id: str | None = None
        if run_code:
            run = self.find_run_by_code(run_code)
            if run is None:
                return Page(items=[], page=1, page_size=fetch_limit, total=0)
            run_id = str(run["id"])

        query = self.client.table("vehicle_track").select("*", count="exact")
        if run_id:
            query = query.eq("processing_run_id", run_id)
        if vehicle_class:
            query = query.eq("vehicle_class", vehicle_class)
        if minimum_confidence is not None:
            query = query.gte("best_detection_confidence", minimum_confidence)
        if window_start:
            query = query.gte("last_seen_at", window_start)
        if window_end:
            query = query.lte("first_seen_at", window_end)

        if camera_codes:
            camera_rows = self._camera_rows_by_code(camera_codes)
            camera_ids = [str(row["id"]) for row in camera_rows.values() if row.get("id")]
            if not camera_ids:
                return Page(items=[], page=1, page_size=fetch_limit, total=0)
            query = query.in_("camera_id", camera_ids)

        candidate_track_ids = self._search_track_ids_by_enrichment(
            colour=colour,
            plate=plate,
            plate_match_type=plate_match_type,
            verified_plate_only=verified_plate_only,
        )
        if candidate_track_ids is not None:
            if not candidate_track_ids:
                return Page(items=[], page=1, page_size=fetch_limit, total=0)
            query = query.in_("id", candidate_track_ids)

        repository_sort_by = _repository_sort_field(sort_by, result_kind="LOCAL_TRACK")
        query = query.order(repository_sort_by, desc=sort_order == "DESC").range(0, max(fetch_limit - 1, 0))
        response = query.execute()
        decorated = self._decorate_tracks(_response_rows(response))
        items = [
            {
                "result_type": "LOCAL_TRACK",
                "global_vehicle_code": None,
                "track_uuid": item.get("track_uuid"),
                "class_name": item.get("vehicle_class"),
                "colour": item.get("primary_colour"),
                "plate_result": item.get("plate_result"),
                "plate": (item.get("plate_result") or {}).get("normalized_text") or item.get("canonical_plate"),
                "plate_status": (item.get("plate_result") or {}).get("status") or item.get("plate_status"),
                "camera_codes": [item.get("camera_code")] if item.get("camera_code") else [],
                "first_seen_at": item.get("first_seen_at"),
                "last_seen_at": item.get("last_seen_at"),
                "confidence": item.get("best_detection_confidence"),
                "class_confidence": item.get("class_confidence"),
                "class_is_stable": item.get("class_is_stable"),
                "class_observation_count": item.get("class_observation_count"),
                "member_track_count": 1,
                "primary_media": item.get("primary_media"),
                "primary_vehicle_media": item.get("primary_vehicle_media"),
                "primary_plate_media": item.get("primary_plate_media"),
            }
            for item in decorated
        ]
        return Page(items=items, page=1, page_size=fetch_limit, total=int(getattr(response, "count", 0) or 0))

    def search_global_vehicles(
        self,
        *,
        run_code: str | None,
        vehicle_class: str | None,
        colour: str | None,
        plate: str | None,
        plate_match_type: str,
        camera_codes: list[str],
        window_start: str | None,
        window_end: str | None,
        minimum_confidence: float | None,
        multi_camera_only: bool,
        verified_plate_only: bool,
        sort_by: str,
        sort_order: str,
        fetch_limit: int,
    ) -> Page:
        query = self.client.table("global_vehicle").select("*", count="exact")
        if run_code:
            run = self.find_run_by_code(run_code)
            if run is None:
                return Page(items=[], page=1, page_size=fetch_limit, total=0)
            query = query.eq("processing_run_id", str(run["id"]))
        if vehicle_class:
            query = query.eq("canonical_vehicle_class", vehicle_class)
        if colour:
            query = query.eq("canonical_color", colour)
        if plate:
            query = _apply_plate_filter(query, plate, plate_match_type, field_name="canonical_plate")
        if minimum_confidence is not None:
            query = query.gte("identity_confidence", minimum_confidence)
        if multi_camera_only:
            query = query.gte("camera_count", 2)
        if window_start:
            query = query.gte("last_seen_at", window_start)
        if window_end:
            query = query.lte("first_seen_at", window_end)

        repository_sort_by = _repository_sort_field(sort_by, result_kind="GLOBAL_VEHICLE")
        response = query.order(repository_sort_by, desc=sort_order == "DESC").range(0, max(fetch_limit - 1, 0)).execute()
        rows = _response_rows(response)
        if not rows:
            return Page(items=[], page=1, page_size=fetch_limit, total=0)

        member_lookup = self._global_vehicle_members_by_id([str(row["id"]) for row in rows if row.get("id")])
        plate_context = self._plate_context_by_track_id(
            [
                str(member["vehicle_track_id"])
                for members in member_lookup.values()
                for member in members
                if member.get("vehicle_track_id")
            ]
        )

        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            global_id = str(row.get("id"))
            members = member_lookup.get(global_id, [])
            member_cameras = {str(member.get("camera_code")) for member in members if member.get("camera_code")}
            if camera_codes and not member_cameras.intersection(camera_codes):
                continue
            if camera_codes and multi_camera_only and len(camera_codes) > 1 and not set(camera_codes).issubset(member_cameras):
                continue
            if verified_plate_only:
                has_verified = any(
                    str((plate_context.get(str(member.get("vehicle_track_id")), {}).get("plate_result") or {}).get("status") or "").upper() == "VERIFIED"
                    for member in members
                    if member.get("vehicle_track_id")
                )
                if not has_verified:
                    continue
            filtered_rows.append(row)

        evidence_lookup = self._global_vehicle_primary_evidence([str(row["id"]) for row in filtered_rows if row.get("id")])
        items = []
        for row in filtered_rows:
            global_id = str(row.get("id"))
            members = member_lookup.get(global_id, [])
            items.append(
                {
                    "result_type": "GLOBAL_VEHICLE",
                    "global_vehicle_code": row.get("global_vehicle_code"),
                    "track_uuid": None,
                    "class_name": row.get("canonical_vehicle_class"),
                    "colour": row.get("canonical_color"),
                    "plate_result": self._global_plate_result(
                        global_row=row,
                        members=members,
                        plate_context_by_track_id=plate_context,
                    ),
                    "plate": (self._global_plate_result(global_row=row, members=members, plate_context_by_track_id=plate_context) or {}).get("normalized_text") or row.get("canonical_plate"),
                    "plate_status": (self._global_plate_result(global_row=row, members=members, plate_context_by_track_id=plate_context) or {}).get("status") or _global_plate_status(members, plate_context),
                    "camera_codes": sorted({str(member.get("camera_code")) for member in members if member.get("camera_code")}),
                    "first_seen_at": row.get("first_seen_at"),
                    "last_seen_at": row.get("last_seen_at"),
                    "confidence": row.get("identity_confidence"),
                    "member_track_count": len(members),
                    "primary_media": evidence_lookup.get(global_id, {}).get("primary_vehicle_media"),
                    "primary_vehicle_media": evidence_lookup.get(global_id, {}).get("primary_vehicle_media"),
                    "primary_plate_media": evidence_lookup.get(global_id, {}).get("primary_plate_media"),
                }
            )
        return Page(items=items, page=1, page_size=fetch_limit, total=len(filtered_rows))

    def list_track_media(self, track_uuid: str) -> list[dict[str, Any]]:
        track = self.get_track_row(track_uuid)
        if track is None:
            return []
        return self._media_for_tracks([str(track["id"])])

    def get_media_by_id(self, media_id: str) -> dict[str, Any] | None:
        rows = _response_rows(self.client.table("track_media").select("*").eq("id", media_id).limit(1).execute())
        return rows[0] if rows else None

    def get_track_row(self, track_uuid: str) -> dict[str, Any] | None:
        rows = _response_rows(
            self.client.table("vehicle_track").select("*").eq("track_uuid", track_uuid).limit(1).execute()
        )
        return rows[0] if rows else None

    def _decorate_tracks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        track_ids = [str(row["id"]) for row in rows if row.get("id")]
        cameras = self._camera_by_id([str(row.get("camera_id")) for row in rows if row.get("camera_id")])
        attributes = self._current_attributes_by_track_id(track_ids)
        plate_context = self._plate_context_by_track_id(track_ids)
        primary_media = self._primary_media_bundle_by_track_id(track_ids)
        items: list[dict[str, Any]] = []
        for row in rows:
            track_id = str(row.get("id"))
            camera = cameras.get(str(row.get("camera_id")), {})
            attribute = attributes.get(track_id, {})
            plate_info = plate_context.get(track_id, {})
            summary = plate_info.get("summary", {})
            plate_result = plate_info.get("plate_result")
            media = primary_media.get(track_id, {})
            items.append(
                {
                    "track_uuid": row.get("track_uuid"),
                    "camera_code": camera.get("camera_code"),
                    "local_track_id": row.get("local_track_id"),
                    "vehicle_class": row.get("vehicle_class"),
                    "class_confidence": self._class_confidence_from_track(row, attribute),
                    "class_is_stable": self._class_is_stable_from_track(row),
                    "class_observation_count": self._class_observation_count_from_track(row),
                    "lifecycle_state": row.get("lifecycle_state"),
                    "first_seen_at": row.get("first_seen_at"),
                    "last_seen_at": row.get("last_seen_at"),
                    "first_video_time_seconds": row.get("first_video_time_seconds"),
                    "last_video_time_seconds": row.get("last_video_time_seconds"),
                    "observation_count": row.get("observation_count"),
                    "best_detection_confidence": row.get("best_detection_confidence"),
                    "average_detection_confidence": row.get("average_detection_confidence"),
                    "primary_colour": attribute.get("primary_color"),
                    "colour_confidence": attribute.get("color_confidence"),
                    "plate_result": plate_result,
                    "canonical_plate": (plate_result or {}).get("normalized_text") or summary.get("canonical_plate"),
                    "plate_status": (plate_result or {}).get("status") or summary.get("status"),
                    "plate_confidence": (plate_result or {}).get("ocr_confidence") or summary.get("confidence"),
                    "primary_media": self._to_media_reference(media.get("primary_vehicle_media")) if media.get("primary_vehicle_media") else None,
                    "primary_vehicle_media": self._to_media_reference(media.get("primary_vehicle_media")) if media.get("primary_vehicle_media") else None,
                    "primary_plate_media": self._to_media_reference(media.get("primary_plate_media")) if media.get("primary_plate_media") else None,
                    "class_diagnostics": self._class_diagnostics_from_track(row),
                }
            )
        return items

    @staticmethod
    def _class_diagnostics_from_track(track_row: dict[str, Any]) -> dict[str, Any] | None:
        metadata = track_row.get("metadata")
        if not isinstance(metadata, dict):
            return None
        diagnostics = metadata.get("class_diagnostics")
        return dict(diagnostics) if isinstance(diagnostics, dict) else None

    @classmethod
    def _class_confidence_from_track(cls, track_row: dict[str, Any], attribute_row: dict[str, Any]) -> float | None:
        diagnostics = cls._class_diagnostics_from_track(track_row) or {}
        if diagnostics.get("class_confidence") is not None:
            return diagnostics.get("class_confidence")
        return attribute_row.get("class_confidence")

    @classmethod
    def _class_is_stable_from_track(cls, track_row: dict[str, Any]) -> bool | None:
        diagnostics = cls._class_diagnostics_from_track(track_row) or {}
        if "class_is_locked" in diagnostics:
            return bool(diagnostics.get("class_is_locked"))
        return None

    @classmethod
    def _class_observation_count_from_track(cls, track_row: dict[str, Any]) -> int | None:
        diagnostics = cls._class_diagnostics_from_track(track_row) or {}
        value = diagnostics.get("class_observation_count")
        return int(value) if value is not None else None

    @staticmethod
    def _observation_class_name(observation_row: dict[str, Any]) -> str | None:
        metadata = observation_row.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("class_name")
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _observation_raw_class_name(observation_row: dict[str, Any]) -> str | None:
        metadata = observation_row.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("raw_class_name")
            if value not in (None, ""):
                return str(value)
        return None

    def _decorate_matches(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        track_ids = []
        global_ids = []
        for row in rows:
            if row.get("source_track_id"):
                track_ids.append(str(row["source_track_id"]))
            if row.get("candidate_track_id"):
                track_ids.append(str(row["candidate_track_id"]))
            if row.get("created_global_vehicle_id"):
                global_ids.append(str(row["created_global_vehicle_id"]))
        tracks = self._track_rows_by_id(track_ids)
        cameras = self._camera_by_id([str(track.get("camera_id")) for track in tracks.values() if track.get("camera_id")])
        attributes = self._current_attributes_by_track_id(track_ids)
        plate_context = self._plate_context_by_track_id(track_ids)
        media_lookup = self._primary_media_bundle_by_track_id(track_ids)
        globals_map: dict[str, str] = {}
        if global_ids:
            global_rows = _response_rows(self.client.table("global_vehicle").select("id,global_vehicle_code").in_("id", global_ids).execute())
            globals_map = {str(row["id"]): str(row.get("global_vehicle_code")) for row in global_rows if row.get("id")}
        items: list[dict[str, Any]] = []
        for row in rows:
            source_track = tracks.get(str(row.get("source_track_id")), {})
            candidate_track = tracks.get(str(row.get("candidate_track_id")), {})
            source_camera = cameras.get(str(source_track.get("camera_id")), {})
            candidate_camera = cameras.get(str(candidate_track.get("camera_id")), {})
            source_track_id = str(row.get("source_track_id") or "")
            candidate_track_id = str(row.get("candidate_track_id") or "")
            items.append(
                {
                    "id": row.get("id"),
                    "source_track_uuid": source_track.get("track_uuid"),
                    "candidate_track_uuid": candidate_track.get("track_uuid"),
                    "source_camera_code": source_camera.get("camera_code"),
                    "candidate_camera_code": candidate_camera.get("camera_code"),
                    "decision": row.get("decision"),
                    "overall_score": row.get("overall_score"),
                    "plate_score": row.get("plate_score"),
                    "route_score": row.get("route_score"),
                    "time_score": row.get("temporal_score"),
                    "class_score": row.get("class_score"),
                    "colour_score": row.get("color_score"),
                    "visual_score": row.get("appearance_score"),
                    "decision_reasons": self._split_reasons(row.get("decision_reason"), row.get("metadata")),
                    "rule_version": row.get("rule_version"),
                    "linked_global_vehicle_code": globals_map.get(str(row.get("created_global_vehicle_id"))),
                    "source_track": self._match_track_summary(
                        track_id=source_track_id,
                        track=source_track,
                        camera=source_camera,
                        attribute=attributes.get(source_track_id, {}),
                        summary=(plate_context.get(source_track_id, {}) or {}).get("summary", {}),
                        plate_result=(plate_context.get(source_track_id, {}) or {}).get("plate_result"),
                        media=media_lookup.get(source_track_id, {}),
                    ),
                    "candidate_track": self._match_track_summary(
                        track_id=candidate_track_id,
                        track=candidate_track,
                        camera=candidate_camera,
                        attribute=attributes.get(candidate_track_id, {}),
                        summary=(plate_context.get(candidate_track_id, {}) or {}).get("summary", {}),
                        plate_result=(plate_context.get(candidate_track_id, {}) or {}).get("plate_result"),
                        media=media_lookup.get(candidate_track_id, {}),
                    ),
                    "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                }
            )
        return items

    def _count_related(self, table_name: str, column: str, ids: list[str]) -> dict[str, int]:
        if not ids:
            return {}
        rows = _response_rows(self.client.table(table_name).select(column).in_(column, ids).execute())
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            value = row.get(column)
            if value is not None:
                counts[str(value)] += 1
        return dict(counts)

    def _count_for_ids(self, table_name: str, column: str, ids: list[str], *, target_key: str | None = None) -> dict[str, int]:
        key = target_key or column
        if not ids:
            return {}
        rows = _response_rows(self.client.table(table_name).select(key).in_(key, ids).execute())
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            value = row.get(key)
            if value is not None:
                counts[str(value)] += 1
        return dict(counts)

    def _filtered_track_ids_by_enrichment(
        self,
        *,
        run_id: str,
        colour: str | None,
        plate: str | None,
        plate_status: str | None,
        has_media: bool | None,
    ) -> list[str] | None:
        filters_active = any(value is not None for value in (colour, plate, plate_status, has_media))
        if not filters_active:
            return None
        candidate_ids: set[str] | None = None
        if colour:
            rows = _response_rows(
                self.client.table("vehicle_attribute")
                .select("vehicle_track_id")
                .eq("attribute_status", "CURRENT")
                .eq("primary_color", colour)
                .execute()
            )
            candidate_ids = {str(row["vehicle_track_id"]) for row in rows if row.get("vehicle_track_id")}
        if plate or plate_status:
            plate_ids = set(
                self._track_ids_matching_plate_selection(
                    plate=plate,
                    plate_match_type="CONTAINS",
                    required_status=plate_status,
                )
            )
            candidate_ids = plate_ids if candidate_ids is None else candidate_ids & plate_ids
        if has_media is not None:
            rows = _response_rows(self.client.table("track_media").select("vehicle_track_id").execute())
            media_ids = {str(row["vehicle_track_id"]) for row in rows if row.get("vehicle_track_id")}
            current = media_ids if has_media else set()
            if not has_media:
                track_rows = _response_rows(self.client.table("vehicle_track").select("id").eq("processing_run_id", run_id).execute())
                all_ids = {str(row["id"]) for row in track_rows if row.get("id")}
                current = all_ids - media_ids
            candidate_ids = current if candidate_ids is None else candidate_ids & current
        return sorted(candidate_ids or [])

    def _camera_by_id(self, camera_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = [camera_id for camera_id in camera_ids if camera_id]
        if not ids:
            return {}
        rows = _response_rows(self.client.table("camera").select("id,camera_code,camera_name,location_name").in_("id", ids).execute())
        return {str(row["id"]): row for row in rows if row.get("id")}

    def _track_rows_by_id(self, track_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = [track_id for track_id in track_ids if track_id]
        if not ids:
            return {}
        rows = _response_rows(self.client.table("vehicle_track").select("*").in_("id", ids).execute())
        return {str(row["id"]): row for row in rows if row.get("id")}

    def _current_attributes_by_track_id(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not track_ids:
            return {}
        rows = _response_rows(
            self.client.table("vehicle_attribute")
            .select("vehicle_track_id,primary_color,color_confidence,vehicle_class,class_confidence,observation_count,attribute_status,metadata")
            .in_("vehicle_track_id", track_ids)
            .eq("attribute_status", "CURRENT")
            .execute()
        )
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result.setdefault(str(row["vehicle_track_id"]), row)
        return result

    def _plate_summaries_by_track_id(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not track_ids:
            return {}
        rows = _response_rows(
            self.client.table("plate_summary")
            .select("vehicle_track_id,selected_plate_reading_id,canonical_plate,plate_pattern,status,confidence,reading_count")
            .in_("vehicle_track_id", track_ids)
            .execute()
        )
        return {str(row["vehicle_track_id"]): row for row in rows if row.get("vehicle_track_id")}

    def _plate_readings_by_id(self, reading_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not reading_ids:
            return {}
        rows = _response_rows(
            self.client.table("plate_reading")
            .select("id,plate_detection_id,status,raw_text,normalized_text,plate_pattern,confidence,is_selected,metadata")
            .in_("id", reading_ids)
            .execute()
        )
        return {str(row["id"]): row for row in rows if row.get("id")}

    def _plate_detections_by_id(self, detection_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not detection_ids:
            return {}
        rows = _response_rows(
            self.client.table("plate_detection")
            .select("id,track_media_id,confidence,frame_number")
            .in_("id", detection_ids)
            .execute()
        )
        return {str(row["id"]): row for row in rows if row.get("id")}

    def _plate_context_by_track_id(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        summaries = self._plate_summaries_by_track_id(track_ids)
        reading_ids = [
            str(summary["selected_plate_reading_id"])
            for summary in summaries.values()
            if summary.get("selected_plate_reading_id")
        ]
        readings = self._plate_readings_by_id(reading_ids)
        detection_ids = [
            str(reading["plate_detection_id"])
            for reading in readings.values()
            if reading.get("plate_detection_id")
        ]
        detections = self._plate_detections_by_id(detection_ids)
        return {
            track_id: {
                "summary": summary,
                "reading": readings.get(str(summary.get("selected_plate_reading_id"))),
                "detection": detections.get(str((readings.get(str(summary.get("selected_plate_reading_id"))) or {}).get("plate_detection_id"))),
                "plate_result": self._build_plate_result(
                    summary=summary,
                    reading=readings.get(str(summary.get("selected_plate_reading_id"))),
                    detection=detections.get(str((readings.get(str(summary.get("selected_plate_reading_id"))) or {}).get("plate_detection_id"))),
                ),
            }
            for track_id, summary in summaries.items()
        }

    def _primary_media_bundle_by_track_id(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not track_ids:
            return {}
        rows = _response_rows(
            self.client.table("track_media")
            .select("id,vehicle_track_id,media_type,storage_provider,storage_uri,frame_number,captured_at,video_time_seconds,width,height,quality_score,sharpness_score,visibility_score,selection_rank,is_primary")
            .in_("vehicle_track_id", track_ids)
            .execute()
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("vehicle_track_id"):
                grouped[str(row["vehicle_track_id"])].append(row)
        result: dict[str, dict[str, Any]] = {}
        for track_id, media_rows in grouped.items():
            primary_vehicle_media = self._best_media_row(media_rows, VEHICLE_MEDIA_TYPE_PRIORITY)
            result[track_id] = {
                "primary_vehicle_media": primary_vehicle_media,
                "primary_plate_media": self._best_media_row(
                    media_rows,
                    PLATE_MEDIA_TYPE_PRIORITY,
                    excluded_media_ids={str(primary_vehicle_media.get("id"))} if primary_vehicle_media and primary_vehicle_media.get("id") else None,
                ),
                "primary_full_frame_media": self._best_media_row(media_rows, FULL_FRAME_MEDIA_TYPE_PRIORITY),
                "primary_annotated_full_frame_media": self._best_media_row(media_rows, ANNOTATED_FULL_FRAME_MEDIA_TYPE_PRIORITY),
            }
        return result

    def _media_for_tracks(self, track_ids: list[str]) -> list[dict[str, Any]]:
        if not track_ids:
            return []
        tracks = self._track_rows_by_id(track_ids)
        rows = _response_rows(
            self.client.table("track_media")
            .select("id,vehicle_track_id,media_type,storage_provider,storage_uri,frame_number,captured_at,video_time_seconds,width,height,quality_score,sharpness_score,visibility_score,selection_rank,is_primary")
            .in_("vehicle_track_id", track_ids)
            .order("selection_rank")
            .execute()
        )
        items: list[dict[str, Any]] = []
        for row in rows:
            track = tracks.get(str(row.get("vehicle_track_id")), {})
            enriched = dict(row)
            enriched["track_uuid"] = track.get("track_uuid")
            items.append(self._to_media_reference(enriched))
        return items

    def _camera_rows_by_code(self, camera_codes: list[str]) -> dict[str, dict[str, Any]]:
        if not camera_codes:
            return {}
        rows = _response_rows(
            self.client.table("camera").select("id,camera_code,camera_name,location_name").in_("camera_code", camera_codes).execute()
        )
        return {str(row.get("camera_code")): row for row in rows if row.get("camera_code")}

    def _global_vehicle_primary_evidence(self, global_vehicle_ids: list[str]) -> dict[str, dict[str, Any] | None]:
        if not global_vehicle_ids:
            return {}
        member_rows = _response_rows(
            self.client.table("global_vehicle_track")
            .select("global_vehicle_id,vehicle_track_id")
            .in_("global_vehicle_id", global_vehicle_ids)
            .eq("is_current", True)
            .execute()
        )
        track_to_global: dict[str, str] = {}
        for row in member_rows:
            if row.get("global_vehicle_id") and row.get("vehicle_track_id"):
                track_to_global.setdefault(str(row["vehicle_track_id"]), str(row["global_vehicle_id"]))
        primary_media = self._primary_media_bundle_by_track_id(list(track_to_global.keys()))
        result: dict[str, dict[str, Any] | None] = {}
        for track_id, global_id in track_to_global.items():
            media_bundle = primary_media.get(track_id) or {}
            if global_id in result:
                continue
            result[global_id] = {
                "primary_vehicle_media": self._to_media_reference(media_bundle.get("primary_vehicle_media")) if media_bundle.get("primary_vehicle_media") else None,
                "primary_plate_media": self._to_media_reference(media_bundle.get("primary_plate_media")) if media_bundle.get("primary_plate_media") else None,
                "primary_full_frame_media": self._to_media_reference(media_bundle.get("primary_full_frame_media")) if media_bundle.get("primary_full_frame_media") else None,
                "primary_annotated_full_frame_media": self._to_media_reference(media_bundle.get("primary_annotated_full_frame_media")) if media_bundle.get("primary_annotated_full_frame_media") else None,
            }
        return result

    def _global_vehicle_members_by_id(self, global_vehicle_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not global_vehicle_ids:
            return {}
        memberships = _response_rows(
            self.client.table("global_vehicle_track")
            .select("global_vehicle_id,vehicle_track_id,association_score,association_method,association_status,is_current,attached_at")
            .in_("global_vehicle_id", global_vehicle_ids)
            .eq("is_current", True)
            .execute()
        )
        track_ids = [str(row["vehicle_track_id"]) for row in memberships if row.get("vehicle_track_id")]
        tracks = self._track_rows_by_id(track_ids)
        cameras = self._camera_by_id([str(track.get("camera_id")) for track in tracks.values() if track.get("camera_id")])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for membership in memberships:
            track = tracks.get(str(membership.get("vehicle_track_id")), {})
            camera = cameras.get(str(track.get("camera_id")), {})
            grouped[str(membership.get("global_vehicle_id"))].append(
                {
                    "vehicle_track_id": membership.get("vehicle_track_id"),
                    "track_uuid": track.get("track_uuid"),
                    "camera_code": camera.get("camera_code"),
                    "first_seen_at": track.get("first_seen_at"),
                    "last_seen_at": track.get("last_seen_at"),
                    "association_score": membership.get("association_score"),
                    "association_method": membership.get("association_method"),
                    "association_status": membership.get("association_status"),
                    "is_current": membership.get("is_current"),
                    "attached_at": membership.get("attached_at"),
                }
            )
        return grouped

    def _global_vehicle_codes_by_id(self, global_vehicle_ids: Iterable[str]) -> dict[str, str]:
        ids = [global_vehicle_id for global_vehicle_id in global_vehicle_ids if global_vehicle_id]
        if not ids:
            return {}
        rows = _response_rows(
            self.client.table("global_vehicle").select("id,global_vehicle_code").in_("id", ids).execute()
        )
        return {
            str(row["id"]): str(row.get("global_vehicle_code"))
            for row in rows
            if row.get("id") and row.get("global_vehicle_code")
        }

    def _global_vehicle_member_counts(self, global_vehicle_ids: Iterable[str]) -> dict[str, int]:
        ids = [global_vehicle_id for global_vehicle_id in global_vehicle_ids if global_vehicle_id]
        if not ids:
            return {}
        rows = _response_rows(
            self.client.table("global_vehicle_track")
            .select("global_vehicle_id")
            .in_("global_vehicle_id", ids)
            .eq("is_current", True)
            .execute()
        )
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            value = row.get("global_vehicle_id")
            if value is not None:
                counts[str(value)] += 1
        return dict(counts)

    def _run_code_for_id(self, run_id: str) -> str | None:
        rows = _response_rows(self.client.table("processing_run").select("run_code").eq("id", run_id).limit(1).execute())
        return str(rows[0].get("run_code")) if rows else None

    def _split_reasons(self, decision_reason: Any, metadata: Any) -> list[str]:
        if isinstance(metadata, dict) and isinstance(metadata.get("reasons"), list):
            return [str(item) for item in metadata.get("reasons") if str(item).strip()]
        text = str(decision_reason or "").strip()
        return [part.strip() for part in text.split(";") if part.strip()]

    def _to_global_membership(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        global_vehicle_id = str(row.get("global_vehicle_id") or "").strip()
        codes = self._global_vehicle_codes_by_id([global_vehicle_id])
        counts = self._global_vehicle_member_counts([global_vehicle_id])
        return {
            "linked": bool(global_vehicle_id),
            "global_vehicle_id": global_vehicle_id or None,
            "global_vehicle_code": codes.get(global_vehicle_id),
            "membership_confidence": row.get("association_score"),
            "membership_status": row.get("association_status"),
            "member_track_count": counts.get(global_vehicle_id),
        }

    def _build_plate_result(
        self,
        *,
        summary: dict[str, Any] | None,
        reading: dict[str, Any] | None,
        detection: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        summary_row = summary or {}
        reading_row = reading or {}
        detection_row = detection or {}
        status = _maybe_upper(reading_row.get("status")) or _maybe_upper(summary_row.get("status")) or "UNKNOWN"
        raw_text = str(reading_row.get("raw_text") or "").strip() or None
        normalized_text = str(reading_row.get("normalized_text") or summary_row.get("canonical_plate") or "").strip().upper() or None
        verification_status = _maybe_upper(reading_row.get("status")) or _maybe_upper((reading_row.get("metadata") or {}).get("verification_status")) or status
        plate_pattern = reading_row.get("plate_pattern") or summary_row.get("plate_pattern")
        ocr_confidence = reading_row.get("confidence")
        if ocr_confidence is None:
            ocr_confidence = summary_row.get("confidence")
        return {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "display_text": _plate_display_text(normalized_text or raw_text, status),
            "status": status,
            "verification_status": verification_status,
            "plate_pattern": plate_pattern,
            "ocr_confidence": ocr_confidence,
            "detector_confidence": detection_row.get("confidence"),
            "source_media_id": detection_row.get("track_media_id"),
        }

    def _best_plate_result(self, plate_results: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
        candidates = [result for result in plate_results if result]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda result: (
                PLATE_STATUS_PRIORITY.get(_maybe_upper(result.get("status")) or "UNKNOWN", 999),
                -float(result.get("ocr_confidence") or 0.0),
                -float(result.get("detector_confidence") or 0.0),
                result.get("display_text") or "",
            ),
        )

    def _global_plate_result(
        self,
        *,
        global_row: dict[str, Any],
        members: list[dict[str, Any]],
        plate_context_by_track_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        best_member_result = self._best_plate_result(
            (plate_context_by_track_id.get(str(member.get("vehicle_track_id")), {}).get("plate_result") for member in members)
        )
        canonical_plate = str(global_row.get("canonical_plate") or "").strip().upper() or None
        if best_member_result is None and canonical_plate is None:
            return None
        if best_member_result is None:
            status = "UNKNOWN"
            return {
                "raw_text": canonical_plate,
                "normalized_text": canonical_plate,
                "display_text": _plate_display_text(canonical_plate, status),
                "status": status,
                "verification_status": status,
                "plate_pattern": None,
                "ocr_confidence": None,
                "detector_confidence": None,
                "source_media_id": None,
            }
        if canonical_plate:
            result = dict(best_member_result)
            result["raw_text"] = result.get("raw_text") or canonical_plate
            result["normalized_text"] = canonical_plate
            result["display_text"] = _plate_display_text(canonical_plate, result.get("status"))
            return result
        return best_member_result

    def _to_media_reference(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "media_id": row.get("id"),
            "media_type": row.get("media_type"),
            "storage_provider": row.get("storage_provider"),
            "storage_uri": row.get("storage_uri"),
            "track_uuid": row.get("track_uuid"),
            "frame_number": row.get("frame_number"),
            "captured_at": row.get("captured_at"),
            "video_time_seconds": row.get("video_time_seconds"),
            "width": row.get("width"),
            "height": row.get("height"),
            "quality_score": row.get("quality_score"),
            "sharpness_score": row.get("sharpness_score"),
            "visibility_score": row.get("visibility_score"),
            "selection_rank": row.get("selection_rank"),
            "is_primary": row.get("is_primary"),
        }

    def _best_media_row(
        self,
        rows: list[dict[str, Any]],
        priority_map: dict[str, int],
        excluded_media_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded_ids = excluded_media_ids or set()
        candidates = [
            row
            for row in rows
            if str(row.get("media_type") or "").upper() in priority_map and str(row.get("id") or "") not in excluded_ids
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (
                priority_map.get(str(row.get("media_type") or "").upper(), 999),
                not bool(row.get("is_primary")),
                int(row.get("selection_rank") or 999999),
            ),
        )

    def _match_track_summary(
        self,
        *,
        track_id: str,
        track: dict[str, Any],
        camera: dict[str, Any],
        attribute: dict[str, Any],
        summary: dict[str, Any],
        plate_result: dict[str, Any] | None,
        media: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not track:
            return None
        return {
            "track_uuid": track.get("track_uuid"),
            "camera_code": camera.get("camera_code"),
            "vehicle_class": track.get("vehicle_class"),
            "lifecycle_state": track.get("lifecycle_state"),
            "first_seen_at": track.get("first_seen_at"),
            "last_seen_at": track.get("last_seen_at"),
            "best_detection_confidence": track.get("best_detection_confidence"),
            "primary_colour": attribute.get("primary_color"),
            "colour_confidence": attribute.get("color_confidence"),
            "plate_result": plate_result,
            "canonical_plate": (plate_result or {}).get("normalized_text") or summary.get("canonical_plate"),
            "plate_status": (plate_result or {}).get("status") or summary.get("status"),
            "plate_confidence": (plate_result or {}).get("ocr_confidence") or summary.get("confidence"),
            "primary_media": self._to_media_reference(media.get("primary_vehicle_media")) if media.get("primary_vehicle_media") else None,
            "primary_vehicle_media": self._to_media_reference(media.get("primary_vehicle_media")) if media.get("primary_vehicle_media") else None,
            "primary_plate_media": self._to_media_reference(media.get("primary_plate_media")) if media.get("primary_plate_media") else None,
            "primary_full_frame_media": self._to_media_reference(media.get("primary_full_frame_media")) if media.get("primary_full_frame_media") else None,
            "primary_annotated_full_frame_media": self._to_media_reference(media.get("primary_annotated_full_frame_media")) if media.get("primary_annotated_full_frame_media") else None,
        }

    def _search_track_ids_by_enrichment(
        self,
        *,
        colour: str | None,
        plate: str | None,
        plate_match_type: str,
        verified_plate_only: bool,
    ) -> list[str] | None:
        filters_active = any((colour, plate, verified_plate_only))
        if not filters_active:
            return None
        candidate_ids: set[str] | None = None
        if colour:
            rows = _response_rows(
                self.client.table("vehicle_attribute")
                .select("vehicle_track_id")
                .eq("attribute_status", "CURRENT")
                .eq("primary_color", colour)
                .execute()
            )
            candidate_ids = {str(row["vehicle_track_id"]) for row in rows if row.get("vehicle_track_id")}
        if plate or verified_plate_only:
            plate_ids = set(
                self._track_ids_matching_plate_selection(
                    plate=plate,
                    plate_match_type=plate_match_type,
                    required_status="VERIFIED" if verified_plate_only else None,
                )
            )
            candidate_ids = plate_ids if candidate_ids is None else candidate_ids & plate_ids
        return sorted(candidate_ids or [])

    def _track_ids_matching_plate_selection(
        self,
        *,
        plate: str | None,
        plate_match_type: str,
        required_status: str | None,
    ) -> list[str]:
        summaries = _response_rows(
            self.client.table("plate_summary")
            .select("vehicle_track_id,selected_plate_reading_id,canonical_plate,status")
            .execute()
        )
        reading_ids = [str(row["selected_plate_reading_id"]) for row in summaries if row.get("selected_plate_reading_id")]
        readings = self._plate_readings_by_id(reading_ids)
        matches: list[str] = []
        normalized_required_status = _maybe_upper(required_status)
        normalized_plate = str(plate or "").strip().upper() or None
        for summary in summaries:
            track_id = str(summary.get("vehicle_track_id") or "").strip()
            if not track_id:
                continue
            reading = readings.get(str(summary.get("selected_plate_reading_id"))) or {}
            status = _maybe_upper(reading.get("status")) or _maybe_upper(summary.get("status"))
            if normalized_required_status and status != normalized_required_status:
                continue
            candidate_plate = str(reading.get("normalized_text") or summary.get("canonical_plate") or "").strip().upper()
            if normalized_plate and not candidate_plate:
                continue
            if normalized_plate and not _plate_text_matches(candidate_plate, normalized_plate, plate_match_type):
                continue
            matches.append(track_id)
        return matches


def _apply_plate_filter(query: Any, plate: str, match_type: str, *, field_name: str) -> Any:
    if match_type == "EXACT":
        return query.eq(field_name, plate)
    if match_type == "STARTS_WITH":
        return query.ilike(field_name, f"{plate}%")
    if match_type == "ENDS_WITH":
        return query.ilike(field_name, f"%{plate}")
    return query.ilike(field_name, f"%{plate}%")


def _plate_text_matches(candidate: str, requested: str, match_type: str) -> bool:
    if match_type == "EXACT":
        return candidate == requested
    if match_type == "STARTS_WITH":
        return candidate.startswith(requested)
    if match_type == "ENDS_WITH":
        return candidate.endswith(requested)
    return requested in candidate


def _repository_sort_field(sort_by: str, *, result_kind: str) -> str:
    if sort_by == "FIRST_SEEN":
        return "first_seen_at"
    if sort_by == "LAST_SEEN":
        return "last_seen_at"
    if sort_by == "CONFIDENCE":
        return "identity_confidence" if result_kind == "GLOBAL_VEHICLE" else "best_detection_confidence"
    if sort_by == "PLATE":
        return "canonical_plate"
    return "first_seen_at"


def _global_plate_status(members: list[dict[str, Any]], plate_lookup: dict[str, dict[str, Any]]) -> str | None:
    statuses = [
        str(
            (plate_lookup.get(str(member.get("vehicle_track_id")), {}).get("plate_result") or {}).get("status")
            or plate_lookup.get(str(member.get("vehicle_track_id")), {}).get("status")
            or ""
        ).upper()
        for member in members
        if member.get("vehicle_track_id")
    ]
    for status in ("VERIFIED", "PROBABLE", "PARTIAL", "UNKNOWN"):
        if status in statuses:
            return status
    return None
