from __future__ import annotations

from typing import Any

from ...persistence.api_read_repository import AnalyticsReadRepository
from ..search_models import PlateMatchType, VehicleSearchQuery, VehicleSearchSortBy
from .media_service import MediaService


class VehicleSearchService:
    def __init__(self, repository: AnalyticsReadRepository, *, media_service: MediaService) -> None:
        self.repository = repository
        self.media_service = media_service

    def search(self, request: VehicleSearchQuery) -> dict[str, Any]:
        window_start, window_end = request.resolved_window()
        fetch_limit = min(max(request.offset + request.limit, request.limit * 4), 500)

        local_page = None
        global_page = None
        if request.result_scope in {"LOCAL_TRACKS", "ALL"}:
            local_page = self.repository.search_local_tracks(
                run_code=request.run_code,
                vehicle_class=request.vehicle_class,
                colour=request.colour,
                plate=request.plate,
                plate_match_type=request.plate_match_type,
                camera_codes=list(request.camera_codes),
                window_start=window_start,
                window_end=window_end,
                minimum_confidence=request.minimum_confidence,
                verified_plate_only=request.verified_plate_only,
                sort_by=request.sort_by,
                sort_order=request.sort_order,
                fetch_limit=fetch_limit,
            )
        if request.result_scope in {"GLOBAL_VEHICLES", "ALL"}:
            global_page = self.repository.search_global_vehicles(
                run_code=request.run_code,
                vehicle_class=request.vehicle_class,
                colour=request.colour,
                plate=request.plate,
                plate_match_type=request.plate_match_type,
                camera_codes=list(request.camera_codes),
                window_start=window_start,
                window_end=window_end,
                minimum_confidence=request.minimum_confidence,
                multi_camera_only=request.multi_camera_only,
                verified_plate_only=request.verified_plate_only,
                sort_by=request.sort_by,
                sort_order=request.sort_order,
                fetch_limit=fetch_limit,
            )

        combined_items: list[dict[str, Any]] = []
        total = 0
        if local_page is not None:
            combined_items.extend(local_page.items)
            total += local_page.total
        if global_page is not None:
            combined_items.extend(global_page.items)
            total += global_page.total

        for item in combined_items:
            score, reasons = _score_result(item, request, window_start, window_end)
            item["relevance_score"] = score
            item["match_reasons"] = reasons
            item["primary_media"] = self.media_service.decorate_media_reference(item.get("primary_media"))
            item["primary_vehicle_media"] = self.media_service.decorate_media_reference(item.get("primary_vehicle_media"))
            item["primary_plate_media"] = self.media_service.decorate_media_reference(item.get("primary_plate_media"))
            item["primary_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_full_frame_media"))
            item["primary_annotated_full_frame_media"] = self.media_service.decorate_media_reference(item.get("primary_annotated_full_frame_media"))

        sorted_items = _sort_results(combined_items, request)
        page_items = sorted_items[request.offset : request.offset + request.limit]

        return {
            "filters": request.applied_filters(),
            "pagination": {
                "limit": request.limit,
                "offset": request.offset,
                "returned": len(page_items),
                "total": total,
                "has_more": request.offset + len(page_items) < total,
            },
            "results": page_items,
        }


def _score_result(
    item: dict[str, Any],
    request: VehicleSearchQuery,
    window_start: str | None,
    window_end: str | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    candidate_plate = str(item.get("plate") or "").upper()
    plate_status = str(item.get("plate_status") or "").upper()

    if request.plate and candidate_plate:
        matched = _plate_matches(candidate_plate, request.plate, request.plate_match_type)
        if matched:
            verified = plate_status == "VERIFIED"
            if candidate_plate == request.plate:
                score += 100.0 if verified else 90.0
                reasons.append("exact verified plate" if verified else "exact plate")
            else:
                score += 80.0 if verified else 70.0
                reasons.append("partial verified plate" if verified else "partial plate match")

    if request.vehicle_class and request.vehicle_class == item.get("class_name"):
        score += 20.0
        reasons.append("class match")

    if request.colour and request.colour == item.get("colour"):
        score += 20.0
        reasons.append("colour match")

    if request.camera_codes:
        requested = set(request.camera_codes)
        candidate_cameras = {str(code) for code in item.get("camera_codes", []) if code}
        if requested.issubset(candidate_cameras):
            score += 15.0
            reasons.append("all requested cameras present")
        elif requested & candidate_cameras:
            score += 8.0
            reasons.append("camera match")

    if window_start or window_end:
        score += 10.0
        reasons.append("time window overlap")

    confidence = item.get("confidence")
    if isinstance(confidence, (int, float)):
        score += float(confidence) * 5.0

    if item.get("result_type") == "GLOBAL_VEHICLE" and (item.get("member_track_count") or 0) > 1:
        score += 6.0
        if request.multi_camera_only:
            reasons.append("multi-camera object")

    media = item.get("primary_vehicle_media") or item.get("primary_media") or {}
    if media.get("media_id"):
        score += 2.0
        reasons.append("evidence available")

    if not reasons:
        reasons.append("matches selected filters")
    return round(score, 3), reasons


def _plate_matches(candidate: str, requested: str, match_type: PlateMatchType) -> bool:
    if match_type == PlateMatchType.EXACT:
        return candidate == requested
    if match_type == PlateMatchType.STARTS_WITH:
        return candidate.startswith(requested)
    if match_type == PlateMatchType.ENDS_WITH:
        return candidate.endswith(requested)
    return requested in candidate


def _sort_results(items: list[dict[str, Any]], request: VehicleSearchQuery) -> list[dict[str, Any]]:
    reverse = request.sort_order == "DESC"

    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        if request.sort_by == VehicleSearchSortBy.FIRST_SEEN:
            return (item.get("first_seen_at") or "", item.get("last_seen_at") or "", item.get("track_uuid") or item.get("global_vehicle_code") or "")
        if request.sort_by == VehicleSearchSortBy.LAST_SEEN:
            return (item.get("last_seen_at") or "", item.get("first_seen_at") or "", item.get("track_uuid") or item.get("global_vehicle_code") or "")
        if request.sort_by == VehicleSearchSortBy.CONFIDENCE:
            return (float(item.get("confidence") or 0.0), item.get("last_seen_at") or "", item.get("track_uuid") or item.get("global_vehicle_code") or "")
        if request.sort_by == VehicleSearchSortBy.PLATE:
            return (item.get("plate") or "", item.get("last_seen_at") or "", item.get("track_uuid") or item.get("global_vehicle_code") or "")
        return (float(item.get("relevance_score") or 0.0), item.get("last_seen_at") or "", item.get("track_uuid") or item.get("global_vehicle_code") or "")

    return sorted(items, key=key, reverse=reverse)
