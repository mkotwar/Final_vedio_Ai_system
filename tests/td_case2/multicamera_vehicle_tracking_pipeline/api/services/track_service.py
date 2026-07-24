from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository


class TrackService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_tracks(self, run_code: str, **kwargs):
        run, page = self.repository.list_tracks(run_code=run_code, **kwargs)
        if run is None:
            raise NotFoundError("RUN_NOT_FOUND", "Run was not found.")
        return page

    def get_track(self, track_uuid: str):
        item = self.repository.get_track_by_uuid(track_uuid)
        if item is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        return item

    def list_observations(self, track_uuid: str, **kwargs):
        track, page = self.repository.list_track_observations(track_uuid=track_uuid, **kwargs)
        if track is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        return page
