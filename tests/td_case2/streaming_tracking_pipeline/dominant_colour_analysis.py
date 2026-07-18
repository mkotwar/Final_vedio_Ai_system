from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .serialization import dataclass_to_dict


COLOUR_PALETTE_BGR = {
    "black": (25, 25, 25),
    "white": (230, 230, 230),
    "gray": (128, 128, 128),
    "red": (40, 40, 200),
    "green": (50, 150, 60),
    "blue": (180, 80, 40),
    "yellow": (40, 210, 210),
    "silver": (185, 185, 185),
    "maroon": (45, 40, 110),
    "pink": (180, 120, 210),
}


@dataclass(frozen=True)
class DominantColourResult:
    raw_colour: str | None
    dominant_colour: str | None
    colour_confidence: float | None
    colour_coverage: float | None
    colour_method: str
    colour_region: str
    colour_warnings: list[str] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def estimate_dominant_colour(
    crop_path: str | Path,
    *,
    object_class: str,
    raw_colour: str | None = None,
    minimum_coverage: float = 0.18,
    debug_dir: str | Path | None = None,
    record_id: str | None = None,
) -> DominantColourResult:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return DominantColourResult(raw_colour, None, None, None, "opencv_unavailable", "none", [str(exc)])

    image = cv2.imread(str(crop_path))
    warnings: list[str] = []
    if image is None:
        return DominantColourResult(raw_colour, None, None, None, "masked_dominant_colour", "none", ["crop_unreadable"])
    height, width = image.shape[:2]
    if width < 12 or height < 12:
        return DominantColourResult(raw_colour, None, None, None, "masked_dominant_colour", "none", ["tiny_object_crop"])

    if object_class == "person":
        region, mask = _person_region(image)
        method = "person_clothing_colour"
        region_name = "central_torso"
    else:
        region, mask = _vehicle_region(image)
        method = "central_body_colour"
        region_name = "central_body"
    if region.size == 0 or mask.size == 0:
        return DominantColourResult(raw_colour, None, None, None, method, region_name, ["empty_analysis_region"])

    valid = region[mask > 0]
    if len(valid) < 25:
        return DominantColourResult(raw_colour, None, None, None, method, region_name, ["too_few_valid_pixels"])

    colours = _classify_pixels(valid)
    total = sum(colours.values())
    if total <= 0:
        return DominantColourResult(raw_colour, None, None, None, method, region_name, ["no_valid_colour_pixels"])
    colour, count = sorted(colours.items(), key=lambda item: (-item[1], item[0]))[0]
    coverage = count / total
    confidence = coverage
    if coverage < minimum_coverage:
        warnings.append("dominant_colour_low_coverage")
        colour = None

    debug_paths: dict[str, str] = {}
    if debug_dir is not None:
        debug_paths = _write_debug_artifacts(debug_dir, record_id or Path(crop_path).stem, image, region, mask, colours, raw_colour, colour, coverage, confidence)

    return DominantColourResult(
        raw_colour=raw_colour,
        dominant_colour=colour,
        colour_confidence=round(float(confidence), 6),
        colour_coverage=round(float(coverage), 6),
        colour_method=method,
        colour_region=region_name,
        colour_warnings=warnings,
        debug_paths=debug_paths,
    )


def _vehicle_region(image: Any) -> tuple[Any, Any]:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    y1, y2 = int(height * 0.18), int(height * 0.78)
    x1, x2 = int(width * 0.12), int(width * 0.88)
    region = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((value > 35) & (value < 245)).astype("uint8") * 255
    mask[:, : max(1, mask.shape[1] // 12)] = 0
    mask[:, -max(1, mask.shape[1] // 12) :] = 0
    # Lower strip often contains plates/lights/road reflections.
    mask[int(mask.shape[0] * 0.72) :, :] = 0
    glare = ((value > 248) & (saturation < 35)) | ((value > 180) & (saturation > 150))
    mask[glare] = 0
    return region, mask


def _person_region(image: Any) -> tuple[Any, Any]:
    import cv2

    height, width = image.shape[:2]
    y1, y2 = int(height * 0.18), int(height * 0.72)
    x1, x2 = int(width * 0.20), int(width * 0.80)
    region = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((value > 30) & (value < 245)).astype("uint8") * 255
    skin_like = ((hsv[:, :, 0] < 25) & (saturation > 35) & (saturation < 170) & (value > 70))
    mask[skin_like] = 0
    return region, mask


def _classify_pixels(pixels: Any) -> dict[str, int]:
    import numpy as np

    palette = {name: np.asarray(value, dtype=float) for name, value in COLOUR_PALETTE_BGR.items()}
    values = pixels.astype(float)
    distances = {name: np.linalg.norm(values - colour, axis=1) for name, colour in palette.items()}
    names = list(distances)
    stacked = np.stack([distances[name] for name in names], axis=1)
    nearest = np.argmin(stacked, axis=1)
    counts = {name: int((nearest == index).sum()) for index, name in enumerate(names)}
    # Fold silver into gray unless silver is very dominant and bright.
    if counts.get("gray", 0) >= counts.get("silver", 0):
        counts["gray"] = counts.get("gray", 0) + counts.pop("silver", 0)
    return {key: value for key, value in counts.items() if value > 0}


def _write_debug_artifacts(
    debug_dir: str | Path,
    record_id: str,
    image: Any,
    region: Any,
    mask: Any,
    counts: dict[str, int],
    raw_colour: str | None,
    final_colour: str | None,
    coverage: float,
    confidence: float,
) -> dict[str, str]:
    import cv2
    import re

    root = Path(debug_dir)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)
    crop_dir = root / safe_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    original_path = crop_dir / "original_crop.jpg"
    region_path = crop_dir / "analysed_region.jpg"
    mask_path = crop_dir / "excluded_region_mask.jpg"
    report_path = crop_dir / "dominant_colour_report.json"
    cv2.imwrite(str(original_path), image)
    cv2.imwrite(str(region_path), region)
    cv2.imwrite(str(mask_path), mask)
    from .serialization import write_json

    write_json(
        report_path,
        {
            "raw_florence_colour": raw_colour,
            "dominant_masked_colour": final_colour,
            "coverage": round(float(coverage), 6),
            "confidence": round(float(confidence), 6),
            "counts": counts,
        },
    )
    return {
        "original_crop": str(original_path),
        "analysed_region": str(region_path),
        "excluded_region_mask": str(mask_path),
        "dominant_colour_report": str(report_path),
    }
