from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository


class MatchService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_matches(self, **kwargs):
        return self.repository.list_cross_camera_matches(**kwargs)

    def get_match(self, match_id: str):
        item = self.repository.get_cross_camera_match(match_id)
        if item is None:
            raise NotFoundError("MATCH_NOT_FOUND", "Cross-camera match was not found.")
        return item
