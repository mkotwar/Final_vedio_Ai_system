from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from fnmatch import fnmatchcase
from typing import Protocol
from uuid import UUID

from .client import Client
from .models import (
    CameraRecord,
    VehicleAttributeRecord,
    VehicleMatchRecord,
    VehicleObservationRecord,
    VehicleSearchFilters,
    VehicleTrackRecord,
    utc_now,
)


class RepositoryConstraintError(ValueError):
    """Raised when repository constraints are violated."""


class VehicleRepository(Protocol):
    def create_camera(self, camera: CameraRecord) -> CameraRecord: ...
    def list_cameras(self) -> list[CameraRecord]: ...
    def get_camera_by_code(self, camera_code: str) -> CameraRecord | None: ...
    def create_vehicle_track(self, track: VehicleTrackRecord) -> VehicleTrackRecord: ...
    def get_track(self, track_id: UUID) -> VehicleTrackRecord: ...
    def get_track_by_uuid(self, track_uuid: str) -> VehicleTrackRecord | None: ...
    def add_vehicle_observations(self, observations: list[VehicleObservationRecord]) -> list[VehicleObservationRecord]: ...
    def get_track_observations(self, vehicle_track_id: UUID) -> list[VehicleObservationRecord]: ...


class SimpleVehicleRepository:
    """Small in-memory repository for proof-of-concept testing and demos."""

    def __init__(self) -> None:
        self._cameras: dict[UUID, CameraRecord] = {}
        self._tracks: dict[UUID, VehicleTrackRecord] = {}
        self._attributes: dict[UUID, VehicleAttributeRecord] = {}
        self._observations: list[VehicleObservationRecord] = []
        self._matches: dict[UUID, VehicleMatchRecord] = {}
        self._observation_identity = 0

    def create_camera(self, camera: CameraRecord) -> CameraRecord:
        if any(existing.camera_code == camera.camera_code for existing in self._cameras.values()):
            raise RepositoryConstraintError("Duplicate camera_code")
        self._cameras[camera.id] = camera
        return camera

    def list_cameras(self) -> list[CameraRecord]:
        return sorted(self._cameras.values(), key=lambda item: item.camera_code)

    def get_camera_by_code(self, camera_code: str) -> CameraRecord | None:
        for camera in self._cameras.values():
            if camera.camera_code == camera_code:
                return camera
        return None

    def create_vehicle_track(self, track: VehicleTrackRecord) -> VehicleTrackRecord:
        if track.camera_id not in self._cameras:
            raise RepositoryConstraintError("Unknown camera_id")
        if track.vehicle_class not in {"car", "bus", "truck", "motorcycle", "unknown"}:
            raise RepositoryConstraintError("Invalid vehicle_class")
        if track.last_seen_at < track.first_seen_at:
            raise RepositoryConstraintError("last_seen_at must be after first_seen_at")
        for existing in self._tracks.values():
            if existing.track_uuid == track.track_uuid:
                raise RepositoryConstraintError("Duplicate track_uuid")
            if (
                existing.camera_id == track.camera_id
                and existing.local_track_id == track.local_track_id
                and existing.first_seen_at == track.first_seen_at
            ):
                raise RepositoryConstraintError("Duplicate camera/local_track_id/first_seen_at")
        self._tracks[track.id] = track
        return track

    def get_track_by_uuid(self, track_uuid: str) -> VehicleTrackRecord | None:
        for track in self._tracks.values():
            if track.track_uuid == track_uuid:
                return track
        return None

    def add_vehicle_observations(self, observations: list[VehicleObservationRecord]) -> list[VehicleObservationRecord]:
        inserted: list[VehicleObservationRecord] = []
        for observation in observations:
            if observation.vehicle_track_id not in self._tracks:
                raise RepositoryConstraintError("Unknown vehicle_track_id for observation")
            self._observation_identity += 1
            inserted_observation = replace(observation, id=self._observation_identity)
            self._observations.append(inserted_observation)
            inserted.append(inserted_observation)
        self._observations.sort(key=lambda item: (item.vehicle_track_id, item.observed_at, item.frame_number))
        return inserted

    def upsert_vehicle_attributes(self, attributes: VehicleAttributeRecord) -> VehicleAttributeRecord:
        if attributes.vehicle_track_id not in self._tracks:
            raise RepositoryConstraintError("Unknown vehicle_track_id for attributes")
        stored = replace(attributes, updated_at=utc_now())
        self._attributes[attributes.vehicle_track_id] = stored
        return stored

    def create_vehicle_match(self, match: VehicleMatchRecord) -> VehicleMatchRecord:
        if match.match_status not in {"confirmed", "probable", "ambiguous", "rejected"}:
            raise RepositoryConstraintError("Invalid match_status")
        if match.source_track_id == match.candidate_track_id:
            raise RepositoryConstraintError("A vehicle cannot be matched to itself")
        source_track = self.get_track(match.source_track_id)
        candidate_track = self.get_track(match.candidate_track_id)
        if source_track.camera_id == candidate_track.camera_id:
            raise RepositoryConstraintError("Vehicle matches must be across different cameras")
        if any(
            existing.source_track_id == match.source_track_id and existing.candidate_track_id == match.candidate_track_id
            for existing in self._matches.values()
        ):
            raise RepositoryConstraintError("Duplicate source/candidate pair")
        self._matches[match.id] = match
        return match

    def search_vehicles(self, filters: VehicleSearchFilters) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for track in self._tracks.values():
            if filters.camera_id and track.camera_id != filters.camera_id:
                continue
            if filters.start_time and track.last_seen_at < filters.start_time:
                continue
            if filters.end_time and track.first_seen_at > filters.end_time:
                continue
            if filters.vehicle_class and track.vehicle_class != filters.vehicle_class:
                continue
            attrs = self._attributes.get(track.id)
            if filters.vehicle_colour and (attrs is None or attrs.vehicle_colour != filters.vehicle_colour):
                continue
            if filters.exact_plate and (attrs is None or attrs.plate_text != filters.exact_plate):
                continue
            if filters.partial_plate and (attrs is None or filters.partial_plate not in (attrs.plate_text or "")):
                continue
            if filters.plate_pattern and (attrs is None or not fnmatchcase(attrs.plate_pattern or "", filters.plate_pattern)):
                continue
            camera = self._cameras[track.camera_id]
            rows.append(
                {
                    "track_id": track.id,
                    "track_uuid": track.track_uuid,
                    "camera_id": camera.id,
                    "camera_code": camera.camera_code,
                    "camera_name": camera.camera_name,
                    "vehicle_class": track.vehicle_class,
                    "first_seen_at": track.first_seen_at,
                    "last_seen_at": track.last_seen_at,
                    "vehicle_colour": attrs.vehicle_colour if attrs else None,
                    "plate_text": attrs.plate_text if attrs else None,
                    "plate_pattern": attrs.plate_pattern if attrs else None,
                    "plate_confidence": attrs.plate_confidence if attrs else None,
                    "best_frame_path": track.best_frame_path,
                    "best_crop_path": track.best_crop_path,
                }
            )
        rows.sort(key=lambda row: (row["camera_code"], row["first_seen_at"], str(row["track_id"])))
        return rows

    def get_track(self, track_id: UUID) -> VehicleTrackRecord:
        try:
            return self._tracks[track_id]
        except KeyError as exc:
            raise RepositoryConstraintError("Unknown vehicle track") from exc

    def get_track_observations(self, vehicle_track_id: UUID) -> list[VehicleObservationRecord]:
        return [item for item in self._observations if item.vehicle_track_id == vehicle_track_id]

    def get_vehicle_matches(self, track_id: UUID, statuses: tuple[str, ...] = ()) -> list[VehicleMatchRecord]:
        results = [
            item
            for item in self._matches.values()
            if item.source_track_id == track_id or item.candidate_track_id == track_id
        ]
        if statuses:
            allowed = set(statuses)
            results = [item for item in results if item.match_status in allowed]
        return sorted(results, key=lambda item: (item.created_at, str(item.id)))


class SupabaseVehicleRepository:
    """Small repository wrapper over the simplified Supabase schema."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def create_camera(self, camera: CameraRecord) -> CameraRecord:
        payload = {
            "camera_code": camera.camera_code,
            "camera_name": camera.camera_name,
            "source_path": camera.source_path,
            "enabled": camera.enabled,
        }
        try:
            response = self.client.table("cameras").insert(payload).execute()
        except Exception as exc:
            raise RepositoryConstraintError(f"Supabase create_camera failed: {exc}") from exc
        return _camera_from_row(_single_row(response))

    def list_cameras(self) -> list[CameraRecord]:
        response = self.client.table("cameras").select("*").order("camera_code").execute()
        return [_camera_from_row(row) for row in list(response.data or [])]

    def get_camera_by_code(self, camera_code: str) -> CameraRecord | None:
        response = self.client.table("cameras").select("*").eq("camera_code", camera_code).limit(1).execute()
        rows = list(response.data or [])
        if not rows:
            return None
        return _camera_from_row(rows[0])

    def create_vehicle_track(self, track: VehicleTrackRecord) -> VehicleTrackRecord:
        payload = {
            "track_uuid": track.track_uuid,
            "camera_id": str(track.camera_id),
            "local_track_id": track.local_track_id,
            "vehicle_class": track.vehicle_class,
            "first_seen_at": track.first_seen_at.isoformat(),
            "last_seen_at": track.last_seen_at.isoformat(),
            "first_frame_number": track.first_frame_number,
            "last_frame_number": track.last_frame_number,
            "observation_count": track.observation_count,
            "best_confidence": track.best_confidence,
            "best_frame_path": track.best_frame_path,
            "best_crop_path": track.best_crop_path,
        }
        try:
            response = self.client.table("vehicle_tracks").insert(payload).execute()
        except Exception as exc:
            raise RepositoryConstraintError(f"Supabase create_vehicle_track failed: {exc}") from exc
        return _track_from_row(_single_row(response))

    def get_track(self, track_id: UUID) -> VehicleTrackRecord:
        response = self.client.table("vehicle_tracks").select("*").eq("id", str(track_id)).limit(1).execute()
        rows = list(response.data or [])
        if not rows:
            raise RepositoryConstraintError("Unknown vehicle track")
        return _track_from_row(rows[0])

    def get_track_by_uuid(self, track_uuid: str) -> VehicleTrackRecord | None:
        response = self.client.table("vehicle_tracks").select("*").eq("track_uuid", track_uuid).limit(1).execute()
        rows = list(response.data or [])
        if not rows:
            return None
        return _track_from_row(rows[0])

    def add_vehicle_observations(self, observations: list[VehicleObservationRecord]) -> list[VehicleObservationRecord]:
        if not observations:
            return []
        payload = [
            {
                "vehicle_track_id": str(item.vehicle_track_id),
                "frame_number": item.frame_number,
                "observed_at": item.observed_at.isoformat(),
                "bbox_x1": item.bbox_x1,
                "bbox_y1": item.bbox_y1,
                "bbox_x2": item.bbox_x2,
                "bbox_y2": item.bbox_y2,
                "confidence": item.confidence,
            }
            for item in observations
        ]
        try:
            response = self.client.table("vehicle_observations").insert(payload).execute()
        except Exception as exc:
            raise RepositoryConstraintError(f"Supabase add_vehicle_observations failed: {exc}") from exc
        return [_observation_from_row(row) for row in list(response.data or [])]

    def upsert_vehicle_attributes(self, attributes: VehicleAttributeRecord) -> VehicleAttributeRecord:
        payload = {
            "vehicle_track_id": str(attributes.vehicle_track_id),
            "vehicle_colour": attributes.vehicle_colour,
            "colour_confidence": attributes.colour_confidence,
            "plate_text": attributes.plate_text,
            "plate_pattern": attributes.plate_pattern,
            "plate_confidence": attributes.plate_confidence,
            "plate_verified": attributes.plate_verified,
            "plate_readings": attributes.plate_readings,
        }
        try:
            response = self.client.table("vehicle_attributes").upsert(payload, on_conflict="vehicle_track_id").execute()
        except Exception as exc:
            raise RepositoryConstraintError(f"Supabase upsert_vehicle_attributes failed: {exc}") from exc
        return _attribute_from_row(_single_row(response))

    def create_vehicle_match(self, match: VehicleMatchRecord) -> VehicleMatchRecord:
        payload = {
            "source_track_id": str(match.source_track_id),
            "candidate_track_id": str(match.candidate_track_id),
            "plate_similarity": match.plate_similarity,
            "colour_match": match.colour_match,
            "class_match": match.class_match,
            "time_gap_seconds": match.time_gap_seconds,
            "match_score": match.match_score,
            "match_status": match.match_status,
        }
        try:
            response = self.client.table("vehicle_matches").insert(payload).execute()
        except Exception as exc:
            raise RepositoryConstraintError(f"Supabase create_vehicle_match failed: {exc}") from exc
        return _match_from_row(_single_row(response))

    def search_vehicles(self, filters: VehicleSearchFilters) -> list[dict[str, object]]:
        query = self.client.table("searchable_vehicles").select("*")
        if filters.vehicle_class:
            query = query.eq("vehicle_class", filters.vehicle_class)
        if filters.exact_plate:
            query = query.eq("plate_text", filters.exact_plate)
        if filters.vehicle_colour:
            query = query.eq("vehicle_colour", filters.vehicle_colour)
        if filters.start_time:
            query = query.gte("last_seen_at", filters.start_time.isoformat())
        if filters.end_time:
            query = query.lte("first_seen_at", filters.end_time.isoformat())
        response = query.execute()
        return list(response.data or [])

    def get_track_observations(self, vehicle_track_id: UUID) -> list[VehicleObservationRecord]:
        response = (
            self.client.table("vehicle_observations")
            .select("*")
            .eq("vehicle_track_id", str(vehicle_track_id))
            .order("observed_at")
            .order("frame_number")
            .execute()
        )
        return [_observation_from_row(row) for row in list(response.data or [])]

    def get_vehicle_matches(self, track_id: UUID, statuses: tuple[str, ...] = ()) -> list[VehicleMatchRecord]:
        response = self.client.table("vehicle_matches").select("*").execute()
        rows = [_match_from_row(row) for row in list(response.data or [])]
        results = [row for row in rows if row.source_track_id == track_id or row.candidate_track_id == track_id]
        if statuses:
            allowed = set(statuses)
            results = [row for row in results if row.match_status in allowed]
        return sorted(results, key=lambda item: (item.created_at, str(item.id)))


def _single_row(response: object) -> dict[str, object]:
    rows = list(getattr(response, "data", None) or [])
    if not rows:
        raise RepositoryConstraintError("Supabase response did not include a row.")
    return rows[0]


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _camera_from_row(row: dict[str, object]) -> CameraRecord:
    return CameraRecord(
        id=UUID(str(row["id"])),
        camera_code=str(row["camera_code"]),
        camera_name=str(row["camera_name"]) if row.get("camera_name") is not None else None,
        source_path=str(row["source_path"]) if row.get("source_path") is not None else None,
        enabled=bool(row.get("enabled", True)),
        created_at=_parse_datetime(row["created_at"]),
    )


def _track_from_row(row: dict[str, object]) -> VehicleTrackRecord:
    return VehicleTrackRecord(
        id=UUID(str(row["id"])),
        camera_id=UUID(str(row["camera_id"])),
        local_track_id=int(row["local_track_id"]),
        vehicle_class=str(row["vehicle_class"]),
        first_seen_at=_parse_datetime(row["first_seen_at"]),
        last_seen_at=_parse_datetime(row["last_seen_at"]),
        track_uuid=str(row["track_uuid"]),
        first_frame_number=int(row["first_frame_number"]) if row.get("first_frame_number") is not None else None,
        last_frame_number=int(row["last_frame_number"]) if row.get("last_frame_number") is not None else None,
        observation_count=int(row.get("observation_count", 0)),
        best_confidence=float(row["best_confidence"]) if row.get("best_confidence") is not None else None,
        best_frame_path=str(row["best_frame_path"]) if row.get("best_frame_path") is not None else None,
        best_crop_path=str(row["best_crop_path"]) if row.get("best_crop_path") is not None else None,
        created_at=_parse_datetime(row["created_at"]),
    )


def _attribute_from_row(row: dict[str, object]) -> VehicleAttributeRecord:
    return VehicleAttributeRecord(
        id=UUID(str(row["id"])),
        vehicle_track_id=UUID(str(row["vehicle_track_id"])),
        vehicle_colour=str(row["vehicle_colour"]) if row.get("vehicle_colour") is not None else None,
        colour_confidence=float(row["colour_confidence"]) if row.get("colour_confidence") is not None else None,
        plate_text=str(row["plate_text"]) if row.get("plate_text") is not None else None,
        plate_pattern=str(row["plate_pattern"]) if row.get("plate_pattern") is not None else None,
        plate_confidence=float(row["plate_confidence"]) if row.get("plate_confidence") is not None else None,
        plate_verified=bool(row.get("plate_verified", False)),
        plate_readings=list(row.get("plate_readings", [])),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _observation_from_row(row: dict[str, object]) -> VehicleObservationRecord:
    return VehicleObservationRecord(
        id=int(row["id"]) if row.get("id") is not None else None,
        vehicle_track_id=UUID(str(row["vehicle_track_id"])),
        frame_number=int(row["frame_number"]),
        observed_at=_parse_datetime(row["observed_at"]),
        bbox_x1=float(row["bbox_x1"]),
        bbox_y1=float(row["bbox_y1"]),
        bbox_x2=float(row["bbox_x2"]),
        bbox_y2=float(row["bbox_y2"]),
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        created_at=_parse_datetime(row["created_at"]),
    )


def _match_from_row(row: dict[str, object]) -> VehicleMatchRecord:
    return VehicleMatchRecord(
        id=UUID(str(row["id"])),
        source_track_id=UUID(str(row["source_track_id"])),
        candidate_track_id=UUID(str(row["candidate_track_id"])),
        match_status=str(row["match_status"]),
        plate_similarity=float(row["plate_similarity"]) if row.get("plate_similarity") is not None else None,
        colour_match=bool(row.get("colour_match", False)),
        class_match=bool(row.get("class_match", False)),
        time_gap_seconds=float(row["time_gap_seconds"]) if row.get("time_gap_seconds") is not None else None,
        match_score=float(row["match_score"]) if row.get("match_score") is not None else None,
        created_at=_parse_datetime(row["created_at"]),
    )
