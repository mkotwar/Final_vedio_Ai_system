from __future__ import annotations

from datetime import datetime
from typing import Any

from ..cross_camera.global_match_models import CrossCameraMatchResult, TrackIdentityFeatures
from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repository_base import AnalyticsRepositoryBase


def _response_rows(response: object) -> list[dict[str, Any]]:
    rows = getattr(response, "data", None)
    if rows is None:
        return []
    return [dict(item) for item in rows]


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


class CrossCameraMatchRepository(AnalyticsRepositoryBase):
    def __init__(self, client: AnalyticsDatabaseClient) -> None:
        super().__init__(client, table_name="cross_camera_match")

    def find_run_by_code(self, run_code: str) -> dict[str, Any] | None:
        response = self.client.table("processing_run").select("id,run_code,status").eq("run_code", run_code).limit(1).execute()
        rows = _response_rows(response)
        return rows[0] if rows else None

    def find_tracks_for_run(self, processing_run_id: str) -> list[TrackIdentityFeatures]:
        tracks = _response_rows(
            self.client.table("vehicle_track")
            .select(
                "id,track_uuid,processing_run_id,camera_id,vehicle_class,first_seen_at,last_seen_at,"
                "first_video_time_seconds,last_video_time_seconds"
            )
            .eq("processing_run_id", processing_run_id)
            .execute()
        )
        camera_ids = [row["camera_id"] for row in tracks if row.get("camera_id")]
        track_ids = [row["id"] for row in tracks if row.get("id")]
        cameras = _response_rows(self.client.table("camera").select("id,camera_code").in_("id", camera_ids).execute()) if camera_ids else []
        attributes = (
            _response_rows(
                self.client.table("vehicle_attribute")
                .select("vehicle_track_id,primary_color,color_confidence,vehicle_class,attribute_status,metadata")
                .in_("vehicle_track_id", track_ids)
                .execute()
            )
            if track_ids
            else []
        )
        summaries = (
            _response_rows(
                self.client.table("plate_summary").select("vehicle_track_id,canonical_plate,status,confidence").in_("vehicle_track_id", track_ids).execute()
            )
            if track_ids
            else []
        )
        media = (
            _response_rows(
                self.client.table("track_media")
                .select("vehicle_track_id,storage_uri,media_type,is_primary,selection_rank")
                .in_("vehicle_track_id", track_ids)
                .execute()
            )
            if track_ids
            else []
        )
        camera_by_id = {str(row["id"]): row for row in cameras if row.get("id") is not None}
        summary_by_track_id = {str(row["vehicle_track_id"]): row for row in summaries if row.get("vehicle_track_id") is not None}
        attribute_by_track_id: dict[str, dict[str, Any]] = {}
        for row in attributes:
            if str(row.get("attribute_status") or "").upper() not in {"", "CURRENT"}:
                continue
            attribute_by_track_id.setdefault(str(row.get("vehicle_track_id")), row)
        primary_media_by_track_id: dict[str, dict[str, Any]] = {}
        for row in sorted(media, key=lambda item: (not bool(item.get("is_primary")), int(item.get("selection_rank") or 999999))):
            primary_media_by_track_id.setdefault(str(row.get("vehicle_track_id")), row)
        features: list[TrackIdentityFeatures] = []
        for track in tracks:
            track_id = str(track["id"])
            camera = camera_by_id.get(str(track.get("camera_id")), {})
            attribute = attribute_by_track_id.get(track_id, {})
            summary = summary_by_track_id.get(track_id, {})
            primary_media = primary_media_by_track_id.get(track_id, {})
            features.append(
                TrackIdentityFeatures(
                    vehicle_track_id=track_id,
                    track_uuid=str(track.get("track_uuid") or ""),
                    processing_run_id=str(track.get("processing_run_id") or ""),
                    camera_id=str(track.get("camera_id") or ""),
                    camera_code=str(camera.get("camera_code") or str(track.get("camera_id") or "")),
                    canonical_class=attribute.get("vehicle_class") or track.get("vehicle_class"),
                    canonical_colour=attribute.get("primary_color"),
                    colour_confidence=_parse_float(attribute.get("color_confidence")),
                    normalized_plate=summary.get("canonical_plate"),
                    plate_status=summary.get("status"),
                    plate_confidence=_parse_float(summary.get("confidence")),
                    first_seen_at=_parse_datetime(track.get("first_seen_at")),
                    last_seen_at=_parse_datetime(track.get("last_seen_at")),
                    first_video_time_seconds=_parse_float(track.get("first_video_time_seconds")),
                    last_video_time_seconds=_parse_float(track.get("last_video_time_seconds")),
                    primary_media_uri=primary_media.get("storage_uri"),
                    metadata={},
                )
            )
        return features

    def find_existing_match(self, left_track_id: str, right_track_id: str) -> dict[str, Any] | None:
        source_track_id, candidate_track_id = sorted((left_track_id, right_track_id))
        response = self._table().select("*").eq("source_track_id", source_track_id).eq("candidate_track_id", candidate_track_id).limit(1).execute()
        rows = _response_rows(response)
        return rows[0] if rows else None

    def list_matches_for_run(self, processing_run_id: str) -> list[dict[str, Any]]:
        return _response_rows(self._table().select("*").eq("processing_run_id", processing_run_id).execute())

    def upsert_match(self, processing_run_id: str, result: CrossCameraMatchResult, *, global_vehicle_id: str | None = None) -> dict[str, Any]:
        source_track_id, candidate_track_id = sorted((result.left_vehicle_track_id, result.right_vehicle_track_id))
        payload = {
            "processing_run_id": processing_run_id,
            "source_track_id": source_track_id,
            "candidate_track_id": candidate_track_id,
            "plate_score": result.plate_score,
            "temporal_score": result.time_score,
            "route_score": result.camera_route_score,
            "class_score": result.class_score,
            "color_score": result.colour_score,
            "appearance_score": result.visual_score,
            "overall_score": result.score,
            "decision": result.decision,
            "decision_reason": "; ".join(result.reasons),
            "matcher_version": result.rule_version,
            "rule_version": result.rule_version,
            "created_global_vehicle_id": global_vehicle_id,
            "metadata": {
                "reasons": list(result.reasons),
                "left_track_uuid": result.left_track_uuid,
                "right_track_uuid": result.right_track_uuid,
            },
        }
        response = self._table().upsert(payload, on_conflict="source_track_id,candidate_track_id").execute()
        rows = _response_rows(response)
        return rows[0] if rows else payload
