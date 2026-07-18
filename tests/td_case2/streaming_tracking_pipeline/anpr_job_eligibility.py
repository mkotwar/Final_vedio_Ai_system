from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .crop_selection import SelectedCropJob
from .plate_detection import _cv2, resolve_image_path
from .serialization import dataclass_to_dict
from .validation import validate_non_empty_string, validate_positive_int


DEFAULT_ALLOWED_ANPR_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle")


@dataclass(frozen=True)
class AnprJobEligibilityConfig:
    allowed_anpr_classes: tuple[str, ...] = DEFAULT_ALLOWED_ANPR_CLASSES
    minimum_anpr_vehicle_crop_width: int = 64
    minimum_anpr_vehicle_crop_height: int = 32
    minimum_anpr_vehicle_crop_area: int = 2048

    def __post_init__(self) -> None:
        if not self.allowed_anpr_classes:
            raise ValueError("allowed_anpr_classes must not be empty.")
        for class_name in self.allowed_anpr_classes:
            validate_non_empty_string(class_name, "allowed_anpr_classes")
        validate_positive_int(self.minimum_anpr_vehicle_crop_width, "minimum_anpr_vehicle_crop_width")
        validate_positive_int(self.minimum_anpr_vehicle_crop_height, "minimum_anpr_vehicle_crop_height")
        validate_positive_int(self.minimum_anpr_vehicle_crop_area, "minimum_anpr_vehicle_crop_area")


@dataclass(frozen=True)
class AnprJobEligibilityRecord:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    object_class: str | None
    crop_role: str
    crop_rank: int
    frame_index: int
    timestamp_sec: float
    vehicle_crop_path: str
    eligible: bool
    exclusion_reason: str | None
    crop_width: int | None
    crop_height: int | None
    crop_area: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def filter_anpr_eligible_jobs(
    jobs: Iterable[SelectedCropJob],
    config: AnprJobEligibilityConfig,
    *,
    run_dir: str | Path | None = None,
) -> tuple[list[SelectedCropJob], list[AnprJobEligibilityRecord]]:
    eligible_jobs: list[SelectedCropJob] = []
    records: list[AnprJobEligibilityRecord] = []
    allowed = {item.lower() for item in config.allowed_anpr_classes}
    for job in jobs:
        eligible, reason, width, height, area = _evaluate_job(job, config, allowed, run_dir=run_dir)
        records.append(
            AnprJobEligibilityRecord(
                source_id=job.source_id,
                track_id=job.track_id,
                track_generation=job.track_generation,
                source_track_id=job.source_track_id,
                object_class=job.object_class,
                crop_role=job.crop_role,
                crop_rank=job.crop_rank,
                frame_index=job.frame_index,
                timestamp_sec=job.timestamp_sec,
                vehicle_crop_path=job.vehicle_crop_path,
                eligible=eligible,
                exclusion_reason=reason,
                crop_width=width,
                crop_height=height,
                crop_area=area,
                metadata={
                    "selection_score": job.selection_score,
                    "allowed_anpr_classes": list(config.allowed_anpr_classes),
                    "minimum_width": config.minimum_anpr_vehicle_crop_width,
                    "minimum_height": config.minimum_anpr_vehicle_crop_height,
                    "minimum_area": config.minimum_anpr_vehicle_crop_area,
                },
            )
        )
        if eligible:
            eligible_jobs.append(job)
    return eligible_jobs, records


def _evaluate_job(
    job: SelectedCropJob,
    config: AnprJobEligibilityConfig,
    allowed: set[str],
    *,
    run_dir: str | Path | None,
) -> tuple[bool, str | None, int | None, int | None, int | None]:
    object_class = (job.object_class or "").lower()
    if object_class not in allowed:
        return False, "non_vehicle_class", None, None, None
    crop_path = resolve_image_path(job.vehicle_crop_path, run_dir=run_dir)
    if not crop_path.exists():
        return False, "vehicle_crop_missing", None, None, None
    image = _cv2().imread(str(crop_path))
    if image is None:
        return False, "vehicle_crop_unreadable", None, None, None
    height, width = image.shape[:2]
    area = int(width * height)
    if (
        width < config.minimum_anpr_vehicle_crop_width
        or height < config.minimum_anpr_vehicle_crop_height
        or area < config.minimum_anpr_vehicle_crop_area
    ):
        return False, "vehicle_crop_too_small", int(width), int(height), area
    return True, None, int(width), int(height), area
