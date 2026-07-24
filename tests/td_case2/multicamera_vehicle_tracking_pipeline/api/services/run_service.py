from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository, Page


class RunService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_runs(self, **kwargs):
        return self.repository.list_runs(**kwargs)

    def get_run(self, run_code: str):
        item = self.repository.get_run_detail(run_code)
        if item is None:
            raise NotFoundError("RUN_NOT_FOUND", "Run was not found.")
        return item
