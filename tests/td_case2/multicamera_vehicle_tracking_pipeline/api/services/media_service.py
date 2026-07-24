from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

from ..errors import BadRequestError, NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository


class MediaService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_media(self, track_uuid: str):
        track = self.repository.get_track_row(track_uuid)
        if track is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        return self.repository.list_track_media(track_uuid)

    def get_media_reference(self, media_id: str):
        row = self.repository.get_media_by_id(media_id)
        if row is None:
            raise NotFoundError("MEDIA_NOT_FOUND", "Media was not found.")
        storage_uri = str(row.get("storage_uri") or "").strip().replace("\\", "/")
        if self._is_unsafe_local_reference(storage_uri):
            raise BadRequestError("UNSAFE_MEDIA_REFERENCE", "Media reference is not safe to expose.")
        return {
            "media_id": str(row.get("id")),
            "availability": "REFERENCE_ONLY",
            "storage_uri": storage_uri or None,
            "media_type": row.get("media_type"),
        }

    def _is_unsafe_local_reference(self, storage_uri: str) -> bool:
        if not storage_uri:
            return False
        if storage_uri.startswith("/") or storage_uri.startswith("//"):
            return True
        if PureWindowsPath(storage_uri).drive:
            return True
        return any(part == ".." for part in PurePosixPath(storage_uri).parts)
