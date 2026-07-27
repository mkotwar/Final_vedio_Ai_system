from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreprocessedPlateVariant:
    variant_name: str
    output_path: Path
    metadata: dict[str, object]


def generate_plate_variants(
    image_path: Path,
    *,
    output_directory: Path,
    max_variants: int,
) -> list[PreprocessedPlateVariant]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for plate preprocessing.") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load plate image for preprocessing: {image_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, object, dict[str, object]]] = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("original", image, {"colour_mode": "bgr"}))
    variants.append(("upscale_x3", cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC), {"scale": 3.0}))
    variants.append(("upscale_x4", cv2.resize(image, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_LANCZOS4), {"scale": 4.0}))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("grayscale", gray, {"colour_mode": "gray"}))
    variants.append(("clahe", clahe, {"colour_mode": "gray", "enhancement": "clahe"}))
    sharpened = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 2.0), -0.6, 0)
    variants.append(("sharpened", sharpened, {"colour_mode": "gray", "enhancement": "unsharp_mask"}))
    adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    variants.append(("adaptive_threshold", adaptive, {"colour_mode": "gray", "threshold": "adaptive_gaussian"}))
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_threshold", otsu, {"colour_mode": "gray", "threshold": "otsu"}))

    saved: list[PreprocessedPlateVariant] = []
    for variant_name, variant_image, metadata in variants[: max(1, int(max_variants))]:
        output_path = output_directory / f"{image_path.stem}_{variant_name}.jpg"
        if not cv2.imwrite(str(output_path), variant_image):
            raise RuntimeError(f"Failed to write preprocessed plate image: {output_path}")
        saved.append(PreprocessedPlateVariant(variant_name=variant_name, output_path=output_path, metadata=metadata))
    return saved


def preprocess_plate_image(image_path: Path, *, output_path: Path) -> Path:
    variants = generate_plate_variants(image_path, output_directory=output_path.parent, max_variants=2)
    for variant in variants:
        if variant.variant_name != "original":
            if variant.output_path != output_path:
                output_path.write_bytes(variant.output_path.read_bytes())
            return output_path
    output_path.write_bytes(image_path.read_bytes())
    return output_path
