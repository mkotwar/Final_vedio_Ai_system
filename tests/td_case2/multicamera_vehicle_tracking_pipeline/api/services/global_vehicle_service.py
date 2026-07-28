from __future__ import annotations

from ..errors import NotFoundError
from ...persistence.api_read_repository import AnalyticsReadRepository
from .media_service import MediaService


class GlobalVehicleService:
    def __init__(self, repository: AnalyticsReadRepository, *, media_service: MediaService) -> None:
        self.repository = repository
        self.media_service = media_service

    def list_global_vehicles(self, **kwargs):
        page = self.repository.list_global_vehicles(**kwargs)
        for item in page.items:
            item["primary_evidence_reference"] = self.media_service.decorate_media_reference(item.get("primary_evidence_reference"))
            item["primary_vehicle_media"] = self.media_service.decorate_media_reference(item.get("primary_vehicle_media"))
            item["primary_plate_media"] = self.media_service.decorate_media_reference(item.get("primary_plate_media"))
            item["primary_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_full_frame_media"))
            item["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_annotated_full_frame_media"))
        return page

    def get_global_vehicle(self, global_vehicle_code: str):
        item = self.repository.get_global_vehicle_by_code(global_vehicle_code)
        if item is None:
            raise NotFoundError("GLOBAL_VEHICLE_NOT_FOUND", "Global vehicle was not found.")
        item["global_vehicle"]["primary_vehicle_media"] = self.media_service.decorate_media_reference(item["global_vehicle"].get("primary_vehicle_media"))
        item["global_vehicle"]["primary_plate_media"] = self.media_service.decorate_media_reference(item["global_vehicle"].get("primary_plate_media"))
        item["global_vehicle"]["primary_full_frame_media"] = self.media_service.decorate_media_reference(item["global_vehicle"].get("primary_full_frame_media"))
        item["global_vehicle"]["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(item["global_vehicle"].get("primary_annotated_full_frame_media"))
        for member in item.get("members", []):
            member["primary_vehicle_media"] = self.media_service.decorate_media_reference(member.get("primary_vehicle_media"))
            member["primary_plate_media"] = self.media_service.decorate_media_reference(member.get("primary_plate_media"))
            member["primary_full_frame_media"] = self.media_service.decorate_media_reference(member.get("primary_full_frame_media"))
            member["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(member.get("primary_annotated_full_frame_media"))
        item["evidence"] = [self.media_service.decorate_media_reference(media) for media in item.get("evidence", [])]
        return item

    def list_member_tracks(self, global_vehicle_code: str):
        members = self.repository.list_global_vehicle_members(global_vehicle_code)
        if not members:
            item = self.repository.get_global_vehicle_by_code(global_vehicle_code)
            if item is None:
                raise NotFoundError("GLOBAL_VEHICLE_NOT_FOUND", "Global vehicle was not found.")
        for member in members:
            member["primary_vehicle_media"] = self.media_service.decorate_media_reference(member.get("primary_vehicle_media"))
            member["primary_plate_media"] = self.media_service.decorate_media_reference(member.get("primary_plate_media"))
            member["primary_full_frame_media"] = self.media_service.decorate_media_reference(member.get("primary_full_frame_media"))
            member["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(member.get("primary_annotated_full_frame_media"))
        return members
