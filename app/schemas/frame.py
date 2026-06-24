"""Pydantic schemas for VLM-extracted frame rich metadata."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventMetadata(BaseModel):
    """Represents a single observable interaction or visual state detected in a frame by the VLM."""

    event_type: str = ""
    """Category of interaction: interaction, observation, none."""

    description: str = ""
    """Precise, objective natural-language sentence describing exactly what is visually observable."""

    actors: List[str] = Field(default_factory=list)
    """List of object IDs (from ObjectMetadata.id) involved in the interaction, e.g. ['car_1', 'person_2']."""

    severity: str = "low"
    """Severity level: low (always default to low for objective frame parsing)."""


class ObjectMetadata(BaseModel):
    """Represents a single detected object within a frame."""

    id: str = ""
    """Unique identifier assigned by the VLM, e.g. 'car_1', 'person_2'. Used to link objects to events."""

    type: str = "unknown"
    """Broad object category, e.g. 'vehicle', 'person', 'furniture'."""

    subtype: str = ""
    """Specific type within the category, e.g. 'sedan', 'pedestrian', 'chair'."""

    color: str = ""
    """Dominant color of the object."""

    condition: str = "normal"
    """Physical state of the object: normal / damaged / displaced / moving / stationary / fallen."""

    attributes: List[str] = Field(default_factory=list)
    """Additional descriptive attributes, e.g. ['carrying bag', 'parked', 'damaged front end']."""


class LocationContextMetadata(BaseModel):
    """Represents the spatial location of a specific object."""
    object_id: str = ""
    location: str = ""


class RelationshipMetadata(BaseModel):
    """Represents a physical or semantic interaction between two objects."""
    subject_id: str = ""
    target_id: str = ""
    relation: str = ""


class OCRMetadata(BaseModel):
    """OCR extraction results from a frame."""

    detected_text: List[str] = Field(default_factory=list)
    license_plates: List[str] = Field(default_factory=list)


class DetectionContextMetadata(BaseModel):
    class_name: str = ""
    confidence: float = 0.0
    bbox: List[float] = Field(default_factory=list)


class TrackedEntityMetadata(BaseModel):
    track_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: List[float] = Field(default_factory=list)


class FrameRichMetadata(BaseModel):
    """Complete rich metadata record for a single extracted video frame."""

    # ── Identity ──────────────────────────────────────────────────────────
    frame_id: str
    video_id: str
    timestamp_seconds: float
    timestamp_human: str
    frame_path: str

    # ── Scene ─────────────────────────────────────────────────────────────
    scene_type: str = "unknown"
    scene_description: str = ""

    # ── Detected Objects ──────────────────────────────────────────────────
    objects: List[ObjectMetadata] = Field(default_factory=list)
    location_context: List[LocationContextMetadata] = Field(default_factory=list)
    relationships: List[RelationshipMetadata] = Field(default_factory=list)

    # ── Detected Events / Incidents ───────────────────────────────────────
    events: List[EventMetadata] = Field(default_factory=list)
    """Frame-level incidents detected by the VLM (collisions, intrusions, falls, etc.)."""

    # ── Activities & People ───────────────────────────────────────────────
    people_count: int = 0
    activities: List[str] = Field(default_factory=list)

    # ── Search & Tagging ──────────────────────────────────────────────────
    keywords: List[str] = Field(default_factory=list)
    caption: str = "No description available."
    search_text: str = ""

    # ── Time Window (set by calculate_time_snippet) ───────────────────────
    timestamp_start_seconds: Optional[float] = None
    timestamp_end_seconds: Optional[float] = None

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr: Optional[Any] = None
    detected_objects: List[DetectionContextMetadata] = Field(default_factory=list)
    tracked_entities: List[TrackedEntityMetadata] = Field(default_factory=list)
    track_ids: List[int] = Field(default_factory=list)
    candidate_reasons: List[str] = Field(default_factory=list)
    object_counts: Dict[str, int] = Field(default_factory=dict)

    # ── Activity Recovery Provenance ─────────────────────────────────────
    activity_recovery_source: Optional[str] = None
    """Set by ActivityRecoveryService if activities were inferred from fallback sources."""


class FrameExtractionRequest(BaseModel):
    """Schema representing frame extraction and VLM analysis request input."""

    video_id: str = Field(
        ...,
        description="Unique UUID4 video identifier of the source file.",
        examples=["479a951c-8b89-4976-b9bd-7c98c1992015"],
    )


class FrameExtractionResponse(BaseModel):
    """Schema representing structural summary of the frame extraction and VLM analysis run."""

    video_id: str = Field(..., description="UUID4 source video identifier.")
    processed_frames: int = Field(..., description="Total count of video frames processed.")
    successful_frames: int = Field(..., description="Count of frames successfully analyzed by VLM.")
    failed_frames: int = Field(..., description="Count of frames where VLM analysis or validation failed.")
    frames: List[FrameRichMetadata] = Field(..., description="Collection of successfully analyzed frame records.")
    total_frames_extracted: int = Field(0, description="Total count of video frames extracted/read from the video file.")
    frames_retained_for_coverage: int = Field(0, description="Frames kept on disk for coverage before VLM candidate filtering.")
    frames_sent_to_qwen: int = Field(0, description="Count of frames sent to the Qwen VLM for rich analysis.")
    frames_filtered_before_vlm: int = Field(0, description="Retained frames judged empty or non-event-like and therefore not sent to VLM.")
    frames_skipped: int = Field(0, description="Count of frames skipped due to visual similarity thresholds.")
    reduction_percent: float = Field(0.0, description="Percentage of frames skipped (saved inference).")
