from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dominant_colour_analysis import COLOUR_PALETTE_BGR
from .serialization import dataclass_to_dict, write_json, write_jsonl


CLOTHING_COLOUR_STATUSES = {"detected", "uncertain", "not_visible", "failed"}
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PersonClothingColourResult:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    object_class: str | None
    upper_clothing_color: str
    lower_clothing_color: str
    dominant_clothing_color: str
    clothing_color_confidence: float
    clothing_color_status: str
    crop_attempts: list[dict[str, Any]] = field(default_factory=list)
    selected_crop_path: str | None = None
    full_frame_path: str | None = None
    method: str = "deterministic_visible_clothing_colour"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def analyse_person_clothing_for_selected_sets(selected_sets: list[Any], *, output_dir: str | Path | None = None) -> list[PersonClothingColourResult]:
    results: list[PersonClothingColourResult] = []
    for selected in selected_sets:
        object_class = str(getattr(selected.lifecycle_record, "dominant_class", "") or "").lower()
        if object_class != "person":
            continue
        jobs = sorted(
            selected.to_crop_jobs(),
            key=lambda job: (
                0 if job.crop_role == "primary" and job.crop_rank == 1 else 1 if job.crop_role == "primary" else 2,
                job.crop_rank,
                job.frame_index,
            ),
        )[:3]
        results.append(_analyse_person_jobs(selected, jobs))
    if output_dir is not None:
        sink = PersonClothingColourArtifactSink(output_dir)
        sink.write(results)
    return results


class PersonClothingColourArtifactSink:
    def __init__(self, run_dir: str | Path) -> None:
        self.output_dir = Path(run_dir) / "07_person_clothing_colour"
        self.report_dir = self.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(self, results: list[PersonClothingColourResult]) -> dict[str, str]:
        records = [item.to_dict() for item in results]
        summary = build_person_clothing_colour_summary(results)
        result_path = self.output_dir / "person_clothing_colour_results.jsonl"
        summary_path = self.report_dir / "person_clothing_colour_summary.json"
        report_path = self.report_dir / "person_clothing_colour_report.json"
        write_jsonl(result_path, records)
        write_json(summary_path, summary)
        write_json(report_path, {"summary": summary, "results": records})
        return {"results": str(result_path), "summary": str(summary_path), "report": str(report_path)}


def build_person_clothing_colour_summary(results: list[PersonClothingColourResult]) -> dict[str, Any]:
    return {
        "person_records_processed": len(results),
        "status_counts": dict(sorted(Counter(item.clothing_color_status for item in results).items())),
        "dominant_clothing_counts": dict(sorted(Counter(item.dominant_clothing_color for item in results).items())),
        "upper_clothing_counts": dict(sorted(Counter(item.upper_clothing_color for item in results).items())),
        "lower_clothing_counts": dict(sorted(Counter(item.lower_clothing_color for item in results).items())),
        "records_with_detected_colour": sum(1 for item in results if item.clothing_color_status == "detected"),
        "records_with_uncertain_colour": sum(1 for item in results if item.clothing_color_status == "uncertain"),
    }


def _analyse_person_jobs(selected: Any, jobs: list[Any]) -> PersonClothingColourResult:
    attempts = [_analyse_crop(job) for job in jobs]
    valid = [item for item in attempts if item.get("status") == "detected"]
    warnings: list[str] = []
    if not attempts:
        warnings.append("no_person_crop_jobs")
    upper = _aggregate_region(valid, "upper")
    lower = _aggregate_region(valid, "lower")
    dominant = _aggregate_region(valid, "dominant")
    confidence_values = [float(item.get("confidence") or 0.0) for item in valid]
    confidence = round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else 0.0
    if valid and dominant != UNKNOWN:
        status = "detected"
    elif attempts:
        status = "uncertain"
    else:
        status = "not_visible"
    selected_job = jobs[0] if jobs else None
    return PersonClothingColourResult(
        source_id=selected.source_id,
        track_id=selected.track_id,
        track_generation=selected.track_generation,
        source_track_id=selected.source_track_id,
        object_class="person",
        upper_clothing_color=upper,
        lower_clothing_color=lower,
        dominant_clothing_color=dominant,
        clothing_color_confidence=confidence,
        clothing_color_status=status,
        crop_attempts=attempts,
        selected_crop_path=selected_job.vehicle_crop_path if selected_job else None,
        full_frame_path=selected_job.full_frame_path if selected_job else None,
        warnings=warnings,
    )


def _analyse_crop(job: Any) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "crop_path": job.vehicle_crop_path, "warning": str(exc)}
    image = cv2.imread(str(job.vehicle_crop_path))
    if image is None:
        return {"status": "failed", "crop_path": job.vehicle_crop_path, "warning": "crop_unreadable"}
    height, width = image.shape[:2]
    if height < 16 or width < 8:
        return {"status": "not_visible", "crop_path": job.vehicle_crop_path, "warning": "tiny_person_crop"}
    upper_region = image[int(height * 0.18) : int(height * 0.55), int(width * 0.18) : int(width * 0.82)]
    lower_region = image[int(height * 0.55) : int(height * 0.90), int(width * 0.20) : int(width * 0.80)]
    upper, upper_cov = _dominant_colour_for_region(upper_region, np=np, cv2=cv2)
    lower, lower_cov = _dominant_colour_for_region(lower_region, np=np, cv2=cv2)
    dominant = _dominant_from_counts([upper, lower])
    confidence = max(upper_cov, lower_cov)
    status = "detected" if dominant != UNKNOWN and confidence >= 0.18 else "uncertain"
    return {
        "status": status,
        "crop_path": job.vehicle_crop_path,
        "full_frame_path": job.full_frame_path,
        "frame_index": job.frame_index,
        "timestamp_sec": job.timestamp_sec,
        "upper_clothing_color": upper,
        "lower_clothing_color": lower,
        "dominant_clothing_color": dominant,
        "confidence": round(float(confidence), 6),
    }


def _dominant_colour_for_region(region: Any, *, np: Any, cv2: Any) -> tuple[str, float]:
    if region.size == 0:
        return UNKNOWN, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((value > 25) & (value < 248)).astype("uint8")
    skin_like = ((hsv[:, :, 0] < 25) & (saturation > 35) & (saturation < 170) & (value > 70))
    mask[skin_like] = 0
    pixels = region[mask > 0]
    if len(pixels) < 20:
        return UNKNOWN, 0.0
    palette = {name: np.asarray(value, dtype=float) for name, value in _person_palette().items()}
    values = pixels.astype(float)
    names = list(palette)
    distances = [np.linalg.norm(values - palette[name], axis=1) for name in names]
    nearest = np.argmin(np.stack(distances, axis=1), axis=1)
    counts = {name: int((nearest == index).sum()) for index, name in enumerate(names)}
    colour, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    coverage = count / max(sum(counts.values()), 1)
    return (colour if coverage >= 0.12 else UNKNOWN, float(coverage))


def _aggregate_region(attempts: list[dict[str, Any]], field: str) -> str:
    values = [
        str(item.get(f"{field}_clothing_color") or item.get("dominant_clothing_color") or UNKNOWN)
        for item in attempts
    ]
    values = [value for value in values if value and value != UNKNOWN]
    if not values:
        return UNKNOWN
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    if len(counts) > 1 and count < 2:
        return UNKNOWN
    return value


def _dominant_from_counts(values: list[str]) -> str:
    filtered = [value for value in values if value and value != UNKNOWN]
    if not filtered:
        return UNKNOWN
    return Counter(filtered).most_common(1)[0][0]


def _person_palette() -> dict[str, tuple[int, int, int]]:
    palette = dict(COLOUR_PALETTE_BGR)
    palette.update(
        {
            "orange": (35, 130, 230),
            "brown": (45, 75, 120),
            "beige": (180, 200, 215),
            "purple": (150, 60, 130),
        }
    )
    palette["grey"] = palette["gray"]
    return palette
