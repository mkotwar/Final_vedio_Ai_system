"""Configuration dataclasses for the isolated streaming tracking pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .serialization import dataclass_to_dict
from .validation import (
    normalize_optional_path,
    validate_allowed_value,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_positive_float,
    validate_positive_int,
    validate_probability,
    validate_finite_float,
)


ENV_PREFIX = "TD_CASE2_STREAM_"
SUPPORTED_TRACKING_BACKENDS = ("supervision_bytetrack", "ultralytics_bytetrack")
SUPPORTED_CLASS_VOTE_POLICIES = ("dominant",)
SUPPORTED_CROP_RETENTION_POLICIES = ("highest_preliminary_score", "uniform_temporal", "hybrid_quality_temporal")
SUPPORTED_PRIMARY_SELECTION_POLICIES = ("quality_only", "quality_with_temporal_diversity", "quality_with_visual_diversity", "hybrid")
SUPPORTED_FALLBACK_SELECTION_POLICIES = ("best_available", "earliest_valid", "largest_valid", "sharpest_valid")
SUPPORTED_DETECTION_MODES = ("combined", "dual")
SUPPORTED_VISION_BACKEND_MODES = ("auto", "florence", "gemini", "disabled")


@dataclass(frozen=True)
class SourceConfig:
    source_path: str | None = None
    source_id: str = "default_source"
    target_processing_fps: float | None = 10.0
    use_source_fps: bool = False
    rtsp_transport: str | None = None
    max_processed_frames: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", normalize_optional_path(self.source_path))
        validate_non_empty_string(self.source_id, "source.source_id")
        if self.target_processing_fps is not None:
            validate_positive_float(self.target_processing_fps, "source.target_processing_fps")
        if self.max_processed_frames is not None:
            validate_positive_int(self.max_processed_frames, "source.max_processed_frames")


@dataclass(frozen=True)
class DetectionConfig:
    model_path: str | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    allowed_class_ids: tuple[int, ...] = ()
    allowed_class_names: tuple[str, ...] = ("person", "car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle")
    device: str = "auto"
    image_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", normalize_optional_path(self.model_path))
        validate_probability(self.confidence_threshold, "detection.confidence_threshold")
        validate_probability(self.iou_threshold, "detection.iou_threshold")
        for class_id in self.allowed_class_ids:
            validate_non_negative_int(class_id, "detection.allowed_class_ids")
        for class_name in self.allowed_class_names:
            validate_non_empty_string(class_name, "detection.allowed_class_names")
        validate_non_empty_string(self.device, "detection.device")
        if self.image_size is not None:
            validate_positive_int(self.image_size, "detection.image_size")


@dataclass(frozen=True)
class ObjectTrackingConfig:
    enable_vehicle_tracking: bool = True
    enable_person_tracking: bool = False
    detection_mode: str = "combined"
    vehicle_model_path: str | None = None
    person_model_path: str | None = "object/Person_detection.pt"
    track_object_groups: tuple[str, ...] = ("vehicle",)
    vehicle_confidence_threshold: float = 0.25
    person_confidence_threshold: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "vehicle_model_path", normalize_optional_path(self.vehicle_model_path))
        object.__setattr__(self, "person_model_path", normalize_optional_path(self.person_model_path))
        object.__setattr__(
            self,
            "detection_mode",
            validate_allowed_value(self.detection_mode, SUPPORTED_DETECTION_MODES, "object_tracking.detection_mode"),
        )
        if not self.enable_vehicle_tracking and not self.enable_person_tracking:
            raise ValueError("At least one object group must be enabled for tracking.")
        groups = tuple(str(item).lower() for item in self.track_object_groups)
        for group in groups:
            validate_allowed_value(group, ("vehicle", "person"), "object_tracking.track_object_groups")
        object.__setattr__(self, "track_object_groups", groups)
        validate_probability(self.vehicle_confidence_threshold, "object_tracking.vehicle_confidence_threshold")
        validate_probability(self.person_confidence_threshold, "object_tracking.person_confidence_threshold")


@dataclass(frozen=True)
class TrackingConfig:
    """Tracker settings only; threshold names differ by backend."""

    backend: str = "ultralytics_bytetrack"
    track_activation_threshold: float = 0.30
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.60
    minimum_consecutive_frames: int = 3
    track_high_threshold: float | None = 0.30
    track_low_threshold: float | None = 0.10
    new_track_threshold: float | None = 0.30
    match_threshold: float | None = 0.80
    fuse_score: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            validate_allowed_value(self.backend, SUPPORTED_TRACKING_BACKENDS, "tracking.backend"),
        )
        validate_probability(self.track_activation_threshold, "tracking.track_activation_threshold")
        validate_positive_int(self.lost_track_buffer, "tracking.lost_track_buffer")
        validate_probability(self.minimum_matching_threshold, "tracking.minimum_matching_threshold")
        validate_positive_int(self.minimum_consecutive_frames, "tracking.minimum_consecutive_frames")
        for field_name in ("track_high_threshold", "track_low_threshold", "new_track_threshold", "match_threshold"):
            value = getattr(self, field_name)
            if value is not None:
                validate_probability(value, f"tracking.{field_name}")


@dataclass(frozen=True)
class TrackLifecycleConfig:
    """Application-level lifecycle policy around tracker output IDs."""

    minimum_confirmation_observations: int = 3
    maximum_tentative_missed_frames: int = 1
    maximum_lost_processed_frames: int = 5
    maximum_lost_seconds: float | None = None
    complete_tentative_tracks: bool = True
    emit_tentative_completion: bool = True
    allow_recovery: bool = True
    flush_on_end_of_stream: bool = True
    class_vote_policy: str = "dominant"

    def __post_init__(self) -> None:
        validate_positive_int(self.minimum_confirmation_observations, "lifecycle.minimum_confirmation_observations")
        validate_non_negative_int(self.maximum_tentative_missed_frames, "lifecycle.maximum_tentative_missed_frames")
        validate_non_negative_int(self.maximum_lost_processed_frames, "lifecycle.maximum_lost_processed_frames")
        if self.maximum_lost_seconds is not None:
            validate_positive_float(self.maximum_lost_seconds, "lifecycle.maximum_lost_seconds")
        object.__setattr__(
            self,
            "class_vote_policy",
            validate_allowed_value(self.class_vote_policy, SUPPORTED_CLASS_VOTE_POLICIES, "lifecycle.class_vote_policy"),
        )


@dataclass(frozen=True)
class CropSelectionConfig:
    max_candidates_per_track: int = 8
    primary_crop_count: int = 3
    keep_fallback_crop: bool = True
    minimum_track_observations: int = 2
    minimum_bbox_area_ratio: float = 0.01
    edge_margin_ratio: float = 0.05

    def __post_init__(self) -> None:
        validate_positive_int(self.max_candidates_per_track, "crop_selection.max_candidates_per_track")
        validate_positive_int(self.primary_crop_count, "crop_selection.primary_crop_count")
        validate_positive_int(self.minimum_track_observations, "crop_selection.minimum_track_observations")
        validate_probability(self.minimum_bbox_area_ratio, "crop_selection.minimum_bbox_area_ratio")
        validate_probability(self.edge_margin_ratio, "crop_selection.edge_margin_ratio")


@dataclass(frozen=True)
class CropCollectionConfig:
    """Step 5 runtime crop candidate collection policy."""

    enabled: bool = True
    save_crop_images: bool = True
    max_candidates_per_track: int = 8
    max_observations_per_track: int = 64
    retention_policy: str = "hybrid_quality_temporal"
    padding_ratio: float = 0.08
    minimum_crop_width: int = 8
    minimum_crop_height: int = 8
    minimum_bbox_area_ratio: float = 0.0005
    maximum_bbox_area_ratio: float | None = None
    edge_margin_ratio: float = 0.02
    preliminary_confidence_weight: float = 0.35
    preliminary_area_weight: float = 0.20
    preliminary_sharpness_weight: float = 0.15
    preliminary_brightness_weight: float = 0.10
    preliminary_contrast_weight: float = 0.10
    preliminary_completeness_weight: float = 0.20
    preliminary_edge_penalty_weight: float = 0.25
    sharpness_normalization_cap: float = 500.0
    contrast_normalization_cap: float = 80.0
    target_brightness: float = 0.50

    def __post_init__(self) -> None:
        validate_positive_int(self.max_candidates_per_track, "crop_collection.max_candidates_per_track")
        validate_positive_int(self.max_observations_per_track, "crop_collection.max_observations_per_track")
        object.__setattr__(
            self,
            "retention_policy",
            validate_allowed_value(self.retention_policy, SUPPORTED_CROP_RETENTION_POLICIES, "crop_collection.retention_policy"),
        )
        validate_probability(self.padding_ratio, "crop_collection.padding_ratio")
        validate_positive_int(self.minimum_crop_width, "crop_collection.minimum_crop_width")
        validate_positive_int(self.minimum_crop_height, "crop_collection.minimum_crop_height")
        validate_probability(self.minimum_bbox_area_ratio, "crop_collection.minimum_bbox_area_ratio")
        if self.maximum_bbox_area_ratio is not None:
            validate_probability(self.maximum_bbox_area_ratio, "crop_collection.maximum_bbox_area_ratio")
            if self.maximum_bbox_area_ratio < self.minimum_bbox_area_ratio:
                raise ValueError("crop_collection.maximum_bbox_area_ratio must be >= minimum_bbox_area_ratio.")
        validate_probability(self.edge_margin_ratio, "crop_collection.edge_margin_ratio")
        for field_info in fields(self):
            if field_info.name.startswith("preliminary_") and field_info.name.endswith("_weight"):
                value = validate_finite_float(getattr(self, field_info.name), f"crop_collection.{field_info.name}")
                if value < 0.0:
                    raise ValueError(f"crop_collection.{field_info.name} must be non-negative.")
        validate_positive_float(self.sharpness_normalization_cap, "crop_collection.sharpness_normalization_cap")
        validate_positive_float(self.contrast_normalization_cap, "crop_collection.contrast_normalization_cap")
        validate_probability(self.target_brightness, "crop_collection.target_brightness")


@dataclass(frozen=True)
class BestCropScoreConfig:
    """Step 6 final crop-selection scoring policy and normalization ranges."""

    confidence_weight: float = 0.28
    bbox_area_weight: float = 0.14
    sharpness_weight: float = 0.16
    brightness_weight: float = 0.10
    contrast_weight: float = 0.10
    completeness_weight: float = 0.14
    plate_visibility_weight: float = 0.00
    temporal_position_weight: float = 0.08
    edge_penalty_weight: float = 0.18
    clipping_penalty_weight: float = 0.18
    low_resolution_penalty_weight: float = 0.12
    bbox_area_normalization_cap: float = 0.08
    sharpness_normalization_cap: float = 5000.0
    contrast_normalization_cap: float = 80.0
    target_brightness: float = 0.50
    low_resolution_width: int = 24
    low_resolution_height: int = 24

    def __post_init__(self) -> None:
        for field_info in fields(self):
            if field_info.name.endswith("_weight"):
                value = validate_finite_float(getattr(self, field_info.name), f"best_crop_score.{field_info.name}")
                if value < 0.0:
                    raise ValueError(f"best_crop_score.{field_info.name} must be non-negative.")
        validate_positive_float(self.bbox_area_normalization_cap, "best_crop_score.bbox_area_normalization_cap")
        validate_positive_float(self.sharpness_normalization_cap, "best_crop_score.sharpness_normalization_cap")
        validate_positive_float(self.contrast_normalization_cap, "best_crop_score.contrast_normalization_cap")
        validate_probability(self.target_brightness, "best_crop_score.target_brightness")
        validate_positive_int(self.low_resolution_width, "best_crop_score.low_resolution_width")
        validate_positive_int(self.low_resolution_height, "best_crop_score.low_resolution_height")


@dataclass(frozen=True)
class BestCropSelectionConfig:
    """Step 6 final primary/fallback crop selection policy."""

    enabled: bool = True
    primary_crop_count: int = 3
    keep_fallback_crop: bool = True
    minimum_primary_score: float = 0.45
    minimum_fallback_score: float | None = 0.20
    minimum_track_observations_for_primary: int = 3
    minimum_candidates_for_primary: int = 2
    minimum_temporal_separation_sec: float = 0.50
    minimum_frame_separation: int = 2
    maximum_bbox_overlap_similarity: float | None = 0.92
    prefer_plate_visible_candidates: bool = False
    require_non_edge_touching_for_primary: bool = True
    allow_edge_touching_fallback: bool = True
    require_complete_crop_for_primary: bool = True
    allow_incomplete_fallback: bool = True
    minimum_sharpness_for_primary: float | None = None
    minimum_brightness_for_primary: float | None = 0.12
    maximum_brightness_for_primary: float | None = 0.92
    minimum_contrast_for_primary: float | None = 8.0
    primary_selection_policy: str = "hybrid"
    fallback_selection_policy: str = "best_available"
    require_crop_path: bool = True
    keep_distinct_fallback_when_primary_short: bool = True
    allow_relaxed_diversity_backfill: bool = False
    minimum_crop_width: int = 8
    minimum_crop_height: int = 8
    create_previews: bool = True

    def __post_init__(self) -> None:
        validate_positive_int(self.primary_crop_count, "best_crop_selection.primary_crop_count")
        validate_probability(self.minimum_primary_score, "best_crop_selection.minimum_primary_score")
        if self.minimum_fallback_score is not None:
            validate_probability(self.minimum_fallback_score, "best_crop_selection.minimum_fallback_score")
        validate_positive_int(self.minimum_track_observations_for_primary, "best_crop_selection.minimum_track_observations_for_primary")
        validate_positive_int(self.minimum_candidates_for_primary, "best_crop_selection.minimum_candidates_for_primary")
        value = validate_finite_float(self.minimum_temporal_separation_sec, "best_crop_selection.minimum_temporal_separation_sec")
        if value < 0.0:
            raise ValueError("best_crop_selection.minimum_temporal_separation_sec must be non-negative.")
        validate_non_negative_int(self.minimum_frame_separation, "best_crop_selection.minimum_frame_separation")
        if self.maximum_bbox_overlap_similarity is not None:
            validate_probability(self.maximum_bbox_overlap_similarity, "best_crop_selection.maximum_bbox_overlap_similarity")
        for field_name in (
            "minimum_sharpness_for_primary",
            "minimum_brightness_for_primary",
            "maximum_brightness_for_primary",
            "minimum_contrast_for_primary",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                finite = validate_finite_float(field_value, f"best_crop_selection.{field_name}")
                if finite < 0.0:
                    raise ValueError(f"best_crop_selection.{field_name} must be non-negative.")
        if (
            self.minimum_brightness_for_primary is not None
            and self.maximum_brightness_for_primary is not None
            and self.minimum_brightness_for_primary > self.maximum_brightness_for_primary
        ):
            raise ValueError("best_crop_selection minimum brightness must be <= maximum brightness.")
        object.__setattr__(
            self,
            "primary_selection_policy",
            validate_allowed_value(self.primary_selection_policy, SUPPORTED_PRIMARY_SELECTION_POLICIES, "best_crop_selection.primary_selection_policy"),
        )
        object.__setattr__(
            self,
            "fallback_selection_policy",
            validate_allowed_value(self.fallback_selection_policy, SUPPORTED_FALLBACK_SELECTION_POLICIES, "best_crop_selection.fallback_selection_policy"),
        )
        validate_positive_int(self.minimum_crop_width, "best_crop_selection.minimum_crop_width")
        validate_positive_int(self.minimum_crop_height, "best_crop_selection.minimum_crop_height")


@dataclass(frozen=True)
class PlateDetectionConfig:
    enabled: bool = True
    model_path: str | None = None
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.45
    device: str = "auto"
    image_size: int = 640
    max_plate_detections_per_vehicle_crop: int = 3
    minimum_plate_width: int = 6
    minimum_plate_height: int = 4
    crop_padding_ratio: float = 0.08
    save_plate_crops: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", normalize_optional_path(self.model_path))
        validate_probability(self.confidence_threshold, "plate_detection.confidence_threshold")
        validate_probability(self.iou_threshold, "plate_detection.iou_threshold")
        validate_non_empty_string(self.device, "plate_detection.device")
        validate_positive_int(self.image_size, "plate_detection.image_size")
        validate_positive_int(self.max_plate_detections_per_vehicle_crop, "plate_detection.max_plate_detections_per_vehicle_crop")
        validate_positive_int(self.minimum_plate_width, "plate_detection.minimum_plate_width")
        validate_positive_int(self.minimum_plate_height, "plate_detection.minimum_plate_height")
        validate_probability(self.crop_padding_ratio, "plate_detection.crop_padding_ratio")


@dataclass(frozen=True)
class FlorenceConfig:
    enabled: bool = True
    base_model_path: str | None = None
    adapter_path: str | None = None
    device: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = True
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_new_tokens: int = 128
    num_beams: int = 3
    do_sample: bool = False
    ocr_task_prompt: str = "<OCR>"
    colour_task_prompt: str = "<VQA>What is the primary colour of the vehicle?"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_model_path", normalize_optional_path(self.base_model_path))
        object.__setattr__(self, "adapter_path", normalize_optional_path(self.adapter_path))
        validate_non_empty_string(self.device, "florence.device")
        validate_non_empty_string(self.dtype, "florence.dtype")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("florence.load_in_4bit and florence.load_in_8bit cannot both be enabled.")
        validate_positive_int(self.max_new_tokens, "florence.max_new_tokens")
        validate_positive_int(self.num_beams, "florence.num_beams")
        validate_non_empty_string(self.ocr_task_prompt, "florence.ocr_task_prompt")
        validate_non_empty_string(self.colour_task_prompt, "florence.colour_task_prompt")


@dataclass(frozen=True)
class VisionBackendConfig:
    backend_mode: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_mode",
            validate_allowed_value(self.backend_mode, SUPPORTED_VISION_BACKEND_MODES, "vision.backend_mode"),
        )


@dataclass(frozen=True)
class GeminiConfig:
    enabled: bool = True
    api_key: str | None = None
    model_name: str = "gemini-2.5-flash"
    timeout_seconds: int = 90
    max_retries: int = 1
    retry_backoff_seconds: float = 2.0
    min_confidence: float = 0.75

    def __post_init__(self) -> None:
        validate_non_empty_string(self.model_name, "gemini.model_name")
        validate_positive_int(self.timeout_seconds, "gemini.timeout_seconds")
        validate_non_negative_int(self.max_retries, "gemini.max_retries")
        if float(self.retry_backoff_seconds) < 0.0:
            raise ValueError("gemini.retry_backoff_seconds must be non-negative.")
        validate_probability(self.min_confidence, "gemini.min_confidence")


@dataclass(frozen=True)
class Step7InferenceConfig:
    process_primary_crops: bool = True
    process_fallback_crops: bool = True
    maximum_vehicle_crops_per_track: int = 4
    maximum_plate_candidates_per_vehicle_crop: int = 3
    stop_after_first_raw_plate_text: bool = True
    run_colour_once_per_track: bool = True
    prefer_primary_for_colour: bool = True
    save_inference_inputs: bool = False
    save_inference_outputs: bool = True

    def __post_init__(self) -> None:
        validate_positive_int(self.maximum_vehicle_crops_per_track, "step7.maximum_vehicle_crops_per_track")
        validate_positive_int(self.maximum_plate_candidates_per_vehicle_crop, "step7.maximum_plate_candidates_per_vehicle_crop")


@dataclass(frozen=True)
class PlateDiagnosticConfig:
    enabled: bool = True
    retry_order: tuple[str, ...] = ("primary_rank_1", "primary_rank_2", "primary_rank_3", "fallback")
    maximum_vehicle_crop_attempts_per_track: int = 4
    run_multiple_confidence_thresholds: bool = True
    diagnostic_confidence_thresholds: tuple[float, ...] = (0.25, 0.15, 0.10, 0.05)
    minimum_box_confidence_to_record: float = 0.0
    keep_below_threshold_detections: bool = True
    save_annotated_vehicle_crops: bool = True
    save_rejected_plate_crops: bool = True
    save_valid_plate_crops: bool = True
    run_ocr_on_valid_plate_candidates: bool = False
    stop_after_first_valid_plate_candidate: bool = False
    stop_after_first_non_empty_ocr_text: bool = True
    record_raw_detector_output: bool = True
    maximum_raw_boxes_per_attempt: int = 50

    def __post_init__(self) -> None:
        if not self.retry_order:
            raise ValueError("plate_diagnostics.retry_order must not be empty.")
        supported = {"primary_rank_1", "primary_rank_2", "primary_rank_3", "fallback"}
        normalized_order = tuple(validate_allowed_value(item, supported, "plate_diagnostics.retry_order") for item in self.retry_order)
        if not self.diagnostic_confidence_thresholds:
            raise ValueError("plate_diagnostics.diagnostic_confidence_thresholds must not be empty.")
        thresholds = tuple(float(item) for item in self.diagnostic_confidence_thresholds)
        for threshold in thresholds:
            validate_probability(threshold, "plate_diagnostics.diagnostic_confidence_thresholds")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("plate_diagnostics.diagnostic_confidence_thresholds must be unique.")
        object.__setattr__(self, "retry_order", normalized_order)
        object.__setattr__(self, "diagnostic_confidence_thresholds", thresholds)
        validate_positive_int(self.maximum_vehicle_crop_attempts_per_track, "plate_diagnostics.maximum_vehicle_crop_attempts_per_track")
        validate_probability(self.minimum_box_confidence_to_record, "plate_diagnostics.minimum_box_confidence_to_record")
        validate_positive_int(self.maximum_raw_boxes_per_attempt, "plate_diagnostics.maximum_raw_boxes_per_attempt")


@dataclass(frozen=True)
class RetryConfig:
    enabled: bool = True
    max_ocr_attempts: int = 2
    use_alternate_crops: bool = True
    use_image_enhancement: bool = True
    allow_fallback_ocr: bool = True

    def __post_init__(self) -> None:
        validate_non_negative_int(self.max_ocr_attempts, "retry.max_ocr_attempts")


@dataclass(frozen=True)
class QueueConfig:
    frame_queue_size: int = 2
    detection_queue_size: int = 2
    tracked_frame_queue_size: int = 2
    completed_track_queue_size: int = 16
    plate_queue_size: int = 8
    florence_queue_size: int = 4
    validation_queue_size: int = 8
    result_queue_size: int = 16

    def __post_init__(self) -> None:
        for field_info in fields(self):
            validate_positive_int(getattr(self, field_info.name), f"queue.{field_info.name}")


@dataclass(frozen=True)
class OutputConfig:
    output_root: str = "debug_runs/streaming_tracking_pipeline"
    save_full_frames: bool = True
    save_vehicle_crops: bool = True
    save_plate_crops: bool = True
    write_frame_jsonl: bool = True
    write_track_json: bool = True
    write_object_json: bool = True
    save_annotated_video: bool = False
    annotated_video_fps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", str(Path(self.output_root).expanduser()))
        validate_non_empty_string(self.output_root, "output.output_root")
        if self.annotated_video_fps is not None:
            validate_positive_float(self.annotated_video_fps, "output.annotated_video_fps")


@dataclass(frozen=True)
class PipelineConfig:
    source: SourceConfig = SourceConfig()
    detection: DetectionConfig = DetectionConfig()
    object_tracking: ObjectTrackingConfig = ObjectTrackingConfig()
    tracking: TrackingConfig = TrackingConfig()
    lifecycle: TrackLifecycleConfig = TrackLifecycleConfig()
    crop_selection: CropSelectionConfig = CropSelectionConfig()
    crop_collection: CropCollectionConfig = CropCollectionConfig()
    best_crop_score: BestCropScoreConfig = BestCropScoreConfig()
    best_crop_selection: BestCropSelectionConfig = BestCropSelectionConfig()
    plate_detection: PlateDetectionConfig = PlateDetectionConfig()
    vision: VisionBackendConfig = VisionBackendConfig()
    florence: FlorenceConfig = FlorenceConfig()
    gemini: GeminiConfig = GeminiConfig()
    step7_inference: Step7InferenceConfig = Step7InferenceConfig()
    plate_diagnostics: PlateDiagnosticConfig = PlateDiagnosticConfig()
    retry: RetryConfig = RetryConfig()
    queue: QueueConfig = QueueConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> "PipelineConfig":
        """Create config from defaults, optional JSON, then environment variables."""

        config = cls()
        if config_path is not None:
            config = config.with_overrides(_load_json_config(config_path))
        config = config.with_overrides(_env_overrides(os.environ))
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Create config from defaults plus a JSON file."""

        return cls().with_overrides(_load_json_config(path))

    def with_overrides(self, overrides: dict[str, Any]) -> "PipelineConfig":
        """Return a new config with nested dictionary overrides applied."""

        config: PipelineConfig = self
        for group_name, group_overrides in overrides.items():
            if not hasattr(config, group_name):
                raise ValueError(f"Unknown configuration group: {group_name}")
            group_value = getattr(config, group_name)
            if not is_dataclass(group_value) or isinstance(group_value, type):
                raise ValueError(f"Unsupported configuration group: {group_name}")
            if not isinstance(group_overrides, dict):
                raise ValueError(f"Configuration group {group_name} must be an object.")
            group_field_names = {field_info.name for field_info in fields(group_value)}
            unknown_fields = set(group_overrides) - group_field_names
            if unknown_fields:
                raise ValueError(f"Unknown fields for {group_name}: {sorted(unknown_fields)}")
            converted = _convert_group_values(group_value, group_overrides)
            config = replace(config, **{group_name: replace(group_value, **converted)})
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration dictionary."""

        return dataclass_to_dict(self)


def _load_json_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline JSON configuration must be an object.")
    return payload


def _parse_bool(value: str, env_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{env_name} must be a boolean value.")


def _parse_int(value: str, env_name: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer.") from exc


def _parse_float(value: str, env_name: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a float.") from exc


def _convert_tuple(value: Any, item_type: type, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_items = list(value)
    try:
        return tuple(item_type(item) for item in raw_items)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain {item_type.__name__} values.") from exc


def _convert_group_values(group_value: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    field_lookup = {field_info.name: field_info for field_info in fields(group_value)}
    for name, value in overrides.items():
        current_value = getattr(group_value, name)
        if isinstance(current_value, bool):
            converted[name] = _parse_bool(value, name) if isinstance(value, str) else bool(value)
        elif isinstance(current_value, int) and not isinstance(current_value, bool):
            converted[name] = _parse_int(value, name) if isinstance(value, str) else int(value)
        elif isinstance(current_value, float):
            converted[name] = _parse_float(value, name) if isinstance(value, str) else float(value)
        elif isinstance(current_value, tuple):
            if field_lookup[name].name == "diagnostic_confidence_thresholds":
                item_type = float
            else:
                item_type = int if current_value and isinstance(current_value[0], int) else str
            if not current_value and field_lookup[name].name == "allowed_class_ids":
                item_type = int
            converted[name] = _convert_tuple(value, item_type, name)
        elif current_value is None and name in {
            "maximum_lost_seconds",
            "maximum_bbox_area_ratio",
            "minimum_fallback_score",
            "maximum_bbox_overlap_similarity",
            "minimum_sharpness_for_primary",
            "minimum_brightness_for_primary",
            "maximum_brightness_for_primary",
            "minimum_contrast_for_primary",
        } and value is not None:
            converted[name] = _parse_float(value, name) if isinstance(value, str) else float(value)
        elif current_value is None and name in {"source_path", "model_path", "base_model_path", "adapter_path", "api_key"}:
            converted[name] = None if value is None else str(value)
        else:
            converted[name] = value
    return converted


def _set_nested(overrides: dict[str, Any], group_name: str, field_name: str, value: Any) -> None:
    overrides.setdefault(group_name, {})[field_name] = value


def _env_overrides(environ: os._Environ[str]) -> dict[str, Any]:
    mappings: dict[str, tuple[str, str, str]] = {
        "TD_CASE2_STREAM_SOURCE_PATH": ("source", "source_path", "str"),
        "TD_CASE2_STREAM_SOURCE_ID": ("source", "source_id", "str"),
        "TD_CASE2_STREAM_TARGET_PROCESSING_FPS": ("source", "target_processing_fps", "float"),
        "TD_CASE2_STREAM_USE_SOURCE_FPS": ("source", "use_source_fps", "bool"),
        "TD_CASE2_STREAM_RTSP_TRANSPORT": ("source", "rtsp_transport", "str"),
        "TD_CASE2_STREAM_MAX_PROCESSED_FRAMES": ("source", "max_processed_frames", "int"),
        "TD_CASE2_STREAM_DETECTOR_MODEL_PATH": ("detection", "model_path", "str"),
        "TD_CASE2_STREAM_DETECTION_CONFIDENCE": ("detection", "confidence_threshold", "float"),
        "TD_CASE2_STREAM_DETECTION_IOU": ("detection", "iou_threshold", "float"),
        "TD_CASE2_STREAM_DETECTION_DEVICE": ("detection", "device", "str"),
        "TD_CASE2_STREAM_DETECTION_IMAGE_SIZE": ("detection", "image_size", "int"),
        "TD_CASE2_STREAM_TRACKING_BACKEND": ("tracking", "backend", "str"),
        "TD_CASE2_STREAM_TRACKING_BUFFER": ("tracking", "lost_track_buffer", "int"),
        "TD_CASE2_STREAM_TRACKING_ACTIVATION_THRESHOLD": ("tracking", "track_activation_threshold", "float"),
        "TD_CASE2_STREAM_TRACKING_MATCH_THRESHOLD": ("tracking", "match_threshold", "float"),
        "TD_CASE2_STREAM_LIFECYCLE_CONFIRMATION_OBSERVATIONS": ("lifecycle", "minimum_confirmation_observations", "int"),
        "TD_CASE2_STREAM_LIFECYCLE_TENTATIVE_MISSED_FRAMES": ("lifecycle", "maximum_tentative_missed_frames", "int"),
        "TD_CASE2_STREAM_LIFECYCLE_LOST_PROCESSED_FRAMES": ("lifecycle", "maximum_lost_processed_frames", "int"),
        "TD_CASE2_STREAM_LIFECYCLE_LOST_SECONDS": ("lifecycle", "maximum_lost_seconds", "float"),
        "TD_CASE2_STREAM_LIFECYCLE_ALLOW_RECOVERY": ("lifecycle", "allow_recovery", "bool"),
        "TD_CASE2_STREAM_LIFECYCLE_FLUSH_ON_EOS": ("lifecycle", "flush_on_end_of_stream", "bool"),
        "TD_CASE2_STREAM_CROP_COLLECTION_ENABLED": ("crop_collection", "enabled", "bool"),
        "TD_CASE2_STREAM_CROP_SAVE_IMAGES": ("crop_collection", "save_crop_images", "bool"),
        "TD_CASE2_STREAM_CROP_MAX_CANDIDATES": ("crop_collection", "max_candidates_per_track", "int"),
        "TD_CASE2_STREAM_CROP_MAX_OBSERVATIONS": ("crop_collection", "max_observations_per_track", "int"),
        "TD_CASE2_STREAM_CROP_RETENTION_POLICY": ("crop_collection", "retention_policy", "str"),
        "TD_CASE2_STREAM_CROP_PADDING_RATIO": ("crop_collection", "padding_ratio", "float"),
        "TD_CASE2_STREAM_CROP_MIN_WIDTH": ("crop_collection", "minimum_crop_width", "int"),
        "TD_CASE2_STREAM_CROP_MIN_HEIGHT": ("crop_collection", "minimum_crop_height", "int"),
        "TD_CASE2_STREAM_CROP_MIN_AREA_RATIO": ("crop_collection", "minimum_bbox_area_ratio", "float"),
        "TD_CASE2_STREAM_CROP_MAX_AREA_RATIO": ("crop_collection", "maximum_bbox_area_ratio", "float"),
        "TD_CASE2_STREAM_SELECTION_PRIMARY_COUNT": ("best_crop_selection", "primary_crop_count", "int"),
        "TD_CASE2_STREAM_SELECTION_MIN_PRIMARY_SCORE": ("best_crop_selection", "minimum_primary_score", "float"),
        "TD_CASE2_STREAM_SELECTION_MIN_FALLBACK_SCORE": ("best_crop_selection", "minimum_fallback_score", "float"),
        "TD_CASE2_STREAM_SELECTION_MIN_FRAME_SEPARATION": ("best_crop_selection", "minimum_frame_separation", "int"),
        "TD_CASE2_STREAM_SELECTION_MIN_TIME_SEPARATION": ("best_crop_selection", "minimum_temporal_separation_sec", "float"),
        "TD_CASE2_STREAM_SELECTION_KEEP_FALLBACK": ("best_crop_selection", "keep_fallback_crop", "bool"),
        "TD_CASE2_STREAM_SELECTION_PRIMARY_POLICY": ("best_crop_selection", "primary_selection_policy", "str"),
        "TD_CASE2_STREAM_SELECTION_FALLBACK_POLICY": ("best_crop_selection", "fallback_selection_policy", "str"),
        "TD_CASE2_STREAM_PLATE_MODEL_PATH": ("plate_detection", "model_path", "str"),
        "TD_CASE2_STREAM_PLATE_ENABLED": ("plate_detection", "enabled", "bool"),
        "TD_CASE2_STREAM_PLATE_CONFIDENCE_THRESHOLD": ("plate_detection", "confidence_threshold", "float"),
        "TD_CASE2_STREAM_PLATE_IOU_THRESHOLD": ("plate_detection", "iou_threshold", "float"),
        "TD_CASE2_STREAM_PLATE_DEVICE": ("plate_detection", "device", "str"),
        "TD_CASE2_STREAM_PLATE_IMAGE_SIZE": ("plate_detection", "image_size", "int"),
        "TD_CASE2_STREAM_PLATE_MAX_DETECTIONS_PER_CROP": ("plate_detection", "max_plate_detections_per_vehicle_crop", "int"),
        "TD_CASE2_STREAM_PLATE_MINIMUM_WIDTH": ("plate_detection", "minimum_plate_width", "int"),
        "TD_CASE2_STREAM_PLATE_MINIMUM_HEIGHT": ("plate_detection", "minimum_plate_height", "int"),
        "TD_CASE2_STREAM_PLATE_CROP_PADDING_RATIO": ("plate_detection", "crop_padding_ratio", "float"),
        "TD_CASE2_STREAM_PLATE_SAVE_CROPS": ("plate_detection", "save_plate_crops", "bool"),
        "TD_CASE2_VISION_BACKEND": ("vision", "backend_mode", "str"),
        "TD_CASE2_STREAM_FLORENCE_MODEL_PATH": ("florence", "base_model_path", "str"),
        "TD_CASE2_FLORENCE_MODEL_PATH": ("florence", "base_model_path", "str"),
        "TD_CASE2_STREAM_FLORENCE_ADAPTER_PATH": ("florence", "adapter_path", "str"),
        "TD_CASE2_STREAM_FLORENCE_ENABLED": ("florence", "enabled", "bool"),
        "TD_CASE2_STREAM_FLORENCE_DEVICE": ("florence", "device", "str"),
        "TD_CASE2_STREAM_FLORENCE_DTYPE": ("florence", "dtype", "str"),
        "TD_CASE2_STREAM_FLORENCE_LOCAL_FILES_ONLY": ("florence", "local_files_only", "bool"),
        "TD_CASE2_STREAM_FLORENCE_TRUST_REMOTE_CODE": ("florence", "trust_remote_code", "bool"),
        "TD_CASE2_STREAM_FLORENCE_LOAD_IN_4BIT": ("florence", "load_in_4bit", "bool"),
        "TD_CASE2_STREAM_FLORENCE_LOAD_IN_8BIT": ("florence", "load_in_8bit", "bool"),
        "TD_CASE2_STREAM_FLORENCE_MAX_NEW_TOKENS": ("florence", "max_new_tokens", "int"),
        "TD_CASE2_STREAM_FLORENCE_NUM_BEAMS": ("florence", "num_beams", "int"),
        "TD_CASE2_STREAM_FLORENCE_DO_SAMPLE": ("florence", "do_sample", "bool"),
        "TD_CASE2_STREAM_FLORENCE_OCR_TASK_PROMPT": ("florence", "ocr_task_prompt", "str"),
        "TD_CASE2_STREAM_FLORENCE_COLOUR_TASK_PROMPT": ("florence", "colour_task_prompt", "str"),
        "TD_CASE2_STREAM_GEMINI_ENABLED": ("gemini", "enabled", "bool"),
        "TD_CASE2_GEMINI_MODEL": ("gemini", "model_name", "str"),
        "TD_CASE2_GEMINI_TIMEOUT_SEC": ("gemini", "timeout_seconds", "int"),
        "TD_CASE2_GEMINI_TIMEOUT_SECONDS": ("gemini", "timeout_seconds", "int"),
        "TD_CASE2_GEMINI_MAX_RETRIES": ("gemini", "max_retries", "int"),
        "TD_CASE2_GEMINI_RETRY_BACKOFF_SEC": ("gemini", "retry_backoff_seconds", "float"),
        "TD_CASE2_GEMINI_MIN_CONFIDENCE": ("gemini", "min_confidence", "float"),
        "TD_CASE2_STREAM_STEP7_PROCESS_PRIMARY_CROPS": ("step7_inference", "process_primary_crops", "bool"),
        "TD_CASE2_STREAM_STEP7_PROCESS_FALLBACK_CROPS": ("step7_inference", "process_fallback_crops", "bool"),
        "TD_CASE2_STREAM_STEP7_MAXIMUM_VEHICLE_CROPS_PER_TRACK": ("step7_inference", "maximum_vehicle_crops_per_track", "int"),
        "TD_CASE2_STREAM_STEP7_MAXIMUM_PLATE_CANDIDATES_PER_VEHICLE_CROP": ("step7_inference", "maximum_plate_candidates_per_vehicle_crop", "int"),
        "TD_CASE2_STREAM_STEP7_STOP_AFTER_FIRST_RAW_PLATE_TEXT": ("step7_inference", "stop_after_first_raw_plate_text", "bool"),
        "TD_CASE2_STREAM_STEP7_RUN_COLOUR_ONCE_PER_TRACK": ("step7_inference", "run_colour_once_per_track", "bool"),
        "TD_CASE2_STREAM_STEP7_PREFER_PRIMARY_FOR_COLOUR": ("step7_inference", "prefer_primary_for_colour", "bool"),
        "TD_CASE2_STREAM_STEP7_SAVE_INFERENCE_INPUTS": ("step7_inference", "save_inference_inputs", "bool"),
        "TD_CASE2_STREAM_STEP7_SAVE_INFERENCE_OUTPUTS": ("step7_inference", "save_inference_outputs", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_ENABLED": ("plate_diagnostics", "enabled", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_RETRY_ORDER": ("plate_diagnostics", "retry_order", "str"),
        "TD_CASE2_STREAM_PLATE_DIAG_MAX_ATTEMPTS": ("plate_diagnostics", "maximum_vehicle_crop_attempts_per_track", "int"),
        "TD_CASE2_STREAM_PLATE_DIAG_MULTI_THRESHOLDS": ("plate_diagnostics", "run_multiple_confidence_thresholds", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_THRESHOLDS": ("plate_diagnostics", "diagnostic_confidence_thresholds", "str"),
        "TD_CASE2_STREAM_PLATE_DIAG_MIN_BOX_CONFIDENCE_TO_RECORD": ("plate_diagnostics", "minimum_box_confidence_to_record", "float"),
        "TD_CASE2_STREAM_PLATE_DIAG_KEEP_BELOW_THRESHOLD": ("plate_diagnostics", "keep_below_threshold_detections", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_SAVE_ANNOTATIONS": ("plate_diagnostics", "save_annotated_vehicle_crops", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_SAVE_REJECTED": ("plate_diagnostics", "save_rejected_plate_crops", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_SAVE_VALID": ("plate_diagnostics", "save_valid_plate_crops", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_RUN_OCR": ("plate_diagnostics", "run_ocr_on_valid_plate_candidates", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_STOP_AFTER_PLATE": ("plate_diagnostics", "stop_after_first_valid_plate_candidate", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_STOP_AFTER_OCR": ("plate_diagnostics", "stop_after_first_non_empty_ocr_text", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_RECORD_RAW": ("plate_diagnostics", "record_raw_detector_output", "bool"),
        "TD_CASE2_STREAM_PLATE_DIAG_MAX_RAW_BOXES": ("plate_diagnostics", "maximum_raw_boxes_per_attempt", "int"),
        "TD_CASE2_STREAM_RETRY_ENABLED": ("retry", "enabled", "bool"),
        "TD_CASE2_STREAM_OUTPUT_ROOT": ("output", "output_root", "str"),
        "TD_CASE2_STREAM_SAVE_ANNOTATED_VIDEO": ("output", "save_annotated_video", "bool"),
        "TD_CASE2_STREAM_ANNOTATED_VIDEO_FPS": ("output", "annotated_video_fps", "float"),
    }
    overrides: dict[str, Any] = {}
    for env_name, (group_name, field_name, value_type) in mappings.items():
        raw_value = environ.get(env_name)
        if raw_value is None:
            continue
        if value_type == "bool":
            parsed: Any = _parse_bool(raw_value, env_name)
        elif value_type == "int":
            parsed = _parse_int(raw_value, env_name)
        elif value_type == "float":
            parsed = _parse_float(raw_value, env_name)
        else:
            parsed = raw_value
        _set_nested(overrides, group_name, field_name, parsed)
    if "GEMINI_API_KEY" in environ:
        _set_nested(overrides, "gemini", "api_key", environ.get("GEMINI_API_KEY"))
    elif "GOOGLE_API_KEY" in environ:
        _set_nested(overrides, "gemini", "api_key", environ.get("GOOGLE_API_KEY"))
    return overrides
