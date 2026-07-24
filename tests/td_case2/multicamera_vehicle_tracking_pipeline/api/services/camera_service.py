from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository


class CameraService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_cameras(self, run_code: str, **kwargs):
        run, page = self.repository.list_run_cameras(run_code=run_code, **kwargs)
        if run is None:
            raise NotFoundError("RUN_NOT_FOUND", "Run was not found.")
        return page

    def get_camera(self, run_code: str, camera_code: str):
        run, item = self.repository.get_camera_in_run(run_code, camera_code)
        if run is None:
            raise NotFoundError("RUN_NOT_FOUND", "Run was not found.")
        if item is None:
            raise NotFoundError("CAMERA_NOT_FOUND", "Camera was not found in the run.")
        return item
