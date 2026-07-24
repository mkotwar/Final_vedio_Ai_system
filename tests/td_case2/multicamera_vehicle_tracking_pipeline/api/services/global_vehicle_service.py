from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository


class GlobalVehicleService:
    def __init__(self, repository: AnalyticsReadRepository) -> None:
        self.repository = repository

    def list_global_vehicles(self, **kwargs):
        return self.repository.list_global_vehicles(**kwargs)

    def get_global_vehicle(self, global_vehicle_code: str):
        item = self.repository.get_global_vehicle_by_code(global_vehicle_code)
        if item is None:
            raise NotFoundError("GLOBAL_VEHICLE_NOT_FOUND", "Global vehicle was not found.")
        return item

    def list_member_tracks(self, global_vehicle_code: str):
        members = self.repository.list_global_vehicle_members(global_vehicle_code)
        if not members:
            item = self.repository.get_global_vehicle_by_code(global_vehicle_code)
            if item is None:
                raise NotFoundError("GLOBAL_VEHICLE_NOT_FOUND", "Global vehicle was not found.")
        return members
