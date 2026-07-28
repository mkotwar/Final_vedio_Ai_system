from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    schema_name: str = Field(validation_alias="schema", serialization_alias="schema")


class RunListItem(BaseModel):
    id: str | None = None
    run_code: str
    status: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None
    camera_count: int
    track_count: int
    global_vehicle_count: int
    processing_error_count: int


class RunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class CameraListItem(BaseModel):
    id: str | None = None
    camera_code: str | None = None
    camera_name: str | None = None
    location: str | None = None
    camera_run_status: str | None = None
    frames_read: int
    frames_processed: int
    detection_count: int
    completed_track_count: int
    discarded_track_count: int


class CameraDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class MediaReference(BaseModel):
    media_id: str | None = None
    media_type: str | None = None
    availability: str | None = None
    content_url: str | None = None
    thumbnail_url: str | None = None
    track_uuid: str | None = None
    frame_number: int | None = None
    captured_at: str | None = None
    video_time_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    quality_score: float | None = None
    sharpness_score: float | None = None
    visibility_score: float | None = None
    selection_rank: int | None = None
    is_primary: bool | None = None


class PlateResult(BaseModel):
    raw_text: str | None = None
    normalized_text: str | None = None
    display_text: str | None = None
    status: str | None = None
    verification_status: str | None = None
    plate_pattern: str | None = None
    ocr_confidence: float | None = None
    detector_confidence: float | None = None
    source_media_id: str | None = None


class TrackGlobalMembership(BaseModel):
    linked: bool
    global_vehicle_id: str | None = None
    global_vehicle_code: str | None = None
    membership_confidence: float | None = None
    membership_status: str | None = None
    member_track_count: int | None = None


class TrackListItem(BaseModel):
    track_uuid: str
    camera_code: str | None = None
    local_track_id: int | None = None
    vehicle_class: str | None = None
    lifecycle_state: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    first_video_time_seconds: float | None = None
    last_video_time_seconds: float | None = None
    observation_count: int | None = None
    best_detection_confidence: float | None = None
    average_detection_confidence: float | None = None
    class_confidence: float | None = None
    class_is_stable: bool | None = None
    class_observation_count: int | None = None
    primary_colour: str | None = None
    colour_confidence: float | None = None
    plate_result: PlateResult | None = None
    canonical_plate: str | None = None
    plate_status: str | None = None
    plate_confidence: float | None = None
    primary_media: MediaReference | None = None
    primary_vehicle_media: MediaReference | None = None
    primary_plate_media: MediaReference | None = None
    primary_full_frame_media: MediaReference | None = None
    primary_annotated_full_frame_media: MediaReference | None = None


class TrackDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ObservationItem(BaseModel):
    frame_number: int
    timestamp: str | None = None
    video_time_seconds: float | None = None
    bbox: dict[str, float | None]
    detection_confidence: float | None = None
    tracker_confidence: float | None = None
    is_key_observation: bool
    class_name: str | None = None
    raw_class_name: str | None = None


class GlobalVehicleListItem(BaseModel):
    global_vehicle_code: str
    run_code: str | None = None
    status: str | None = None
    canonical_plate: str | None = None
    canonical_colour: str | None = None
    canonical_vehicle_class: str | None = None
    plate_result: PlateResult | None = None
    confidence: float | None = None
    camera_count: int | None = None
    track_count: int | None = None
    creation_method: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    primary_evidence_reference: MediaReference | None = None
    primary_vehicle_media: MediaReference | None = None
    primary_plate_media: MediaReference | None = None
    primary_full_frame_media: MediaReference | None = None
    primary_annotated_full_frame_media: MediaReference | None = None


class GlobalVehicleMember(BaseModel):
    vehicle_track_id: str | None = None
    track_uuid: str | None = None
    camera_code: str | None = None
    vehicle_class: str | None = None
    primary_colour: str | None = None
    plate_result: PlateResult | None = None
    canonical_plate: str | None = None
    plate_status: str | None = None
    plate_confidence: float | None = None
    best_detection_confidence: float | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    association_score: float | None = None
    association_method: str | None = None
    association_status: str | None = None
    is_current: bool | None = None
    attached_at: str | None = None
    primary_vehicle_media: MediaReference | None = None
    primary_plate_media: MediaReference | None = None
    primary_full_frame_media: MediaReference | None = None
    primary_annotated_full_frame_media: MediaReference | None = None


class GlobalVehicleDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class MatchListItem(BaseModel):
    id: str | None = None
    source_track_uuid: str | None = None
    candidate_track_uuid: str | None = None
    source_camera_code: str | None = None
    candidate_camera_code: str | None = None
    decision: str | None = None
    overall_score: float | None = None
    plate_score: float | None = None
    route_score: float | None = None
    time_score: float | None = None
    class_score: float | None = None
    colour_score: float | None = None
    visual_score: float | None = None
    decision_reasons: list[str]
    rule_version: str | None = None
    linked_global_vehicle_code: str | None = None
    source_track: TrackListItem | None = None
    candidate_track: TrackListItem | None = None


class MatchDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class MediaDeliveryResponse(BaseModel):
    media_id: str
    availability: str
    media_type: str | None = None
    content_url: str | None = None
    thumbnail_url: str | None = None
    frame_number: int | None = None
    width: int | None = None
    height: int | None = None
    quality_score: float | None = None
    sharpness_score: float | None = None
    visibility_score: float | None = None
    selection_rank: int | None = None
    is_primary: bool | None = None
    error_detail: str | None = None


class MediaSignedUrlResponse(BaseModel):
    media_id: str
    availability: str
    url: str | None = None
    expires_in: int | None = None
