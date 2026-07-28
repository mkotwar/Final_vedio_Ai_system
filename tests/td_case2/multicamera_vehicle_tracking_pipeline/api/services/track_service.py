from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository
from .media_service import MediaService


class TrackService:
    def __init__(self, repository: AnalyticsReadRepository, *, media_service: MediaService) -> None:
        self.repository = repository
        self.media_service = media_service

    def list_tracks(self, run_code: str, **kwargs):
        run, page = self.repository.list_tracks(run_code=run_code, **kwargs)
        if run is None:
            raise NotFoundError("RUN_NOT_FOUND", "Run was not found.")
        for item in page.items:
            item["primary_media"] = self.media_service.decorate_media_reference(item.get("primary_media"))
            item["primary_vehicle_media"] = self.media_service.decorate_media_reference(item.get("primary_vehicle_media"))
            item["primary_plate_media"] = self.media_service.decorate_media_reference(item.get("primary_plate_media"))
            item["primary_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_full_frame_media"))
            item["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_annotated_full_frame_media"))
        return page

    def get_track(self, track_uuid: str):
        item = self.repository.get_track_by_uuid(track_uuid)
        if item is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        item["track"]["primary_media"] = self.media_service.decorate_media_reference(item["track"].get("primary_media"))
        item["track"]["primary_vehicle_media"] = self.media_service.decorate_media_reference(item["track"].get("primary_vehicle_media"))
        item["track"]["primary_plate_media"] = self.media_service.decorate_media_reference(item["track"].get("primary_plate_media"))
        item["track"]["primary_full_frame_media"] = self.media_service.decorate_media_reference(item["track"].get("primary_full_frame_media"))
        item["track"]["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(item["track"].get("primary_annotated_full_frame_media"))
        item["media"] = [self.media_service.decorate_media_reference(media) for media in item.get("media", [])]
        return item

    def list_observations(self, track_uuid: str, **kwargs):
        track, page = self.repository.list_track_observations(track_uuid=track_uuid, **kwargs)
        if track is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        return page
