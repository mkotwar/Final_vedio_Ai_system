from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository
from .media_service import MediaService


class MatchService:
    def __init__(self, repository: AnalyticsReadRepository, *, media_service: MediaService) -> None:
        self.repository = repository
        self.media_service = media_service

    def list_matches(self, **kwargs):
        page = self.repository.list_cross_camera_matches(**kwargs)
        for item in page.items:
            self._decorate_match(item)
        return page

    def get_match(self, match_id: str):
        item = self.repository.get_cross_camera_match(match_id)
        if item is None:
            raise NotFoundError("MATCH_NOT_FOUND", "Cross-camera match was not found.")
        self._decorate_match(item)
        return item

    def _decorate_match(self, item: dict) -> None:
        for field_name in ("source_track", "candidate_track"):
            track = item.get(field_name)
            if not isinstance(track, dict):
                continue
            track["primary_media"] = self.media_service.decorate_media_reference(track.get("primary_media"))
            track["primary_vehicle_media"] = self.media_service.decorate_media_reference(track.get("primary_vehicle_media"))
            track["primary_plate_media"] = self.media_service.decorate_media_reference(track.get("primary_plate_media"))
