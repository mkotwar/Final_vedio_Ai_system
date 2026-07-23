from __future__ import annotations

from pathlib import Path


def preprocess_plate_image(image_path: Path, *, output_path: Path) -> Path:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for plate preprocessing.") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load plate image for preprocessing: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    sharpened = cv2.GaussianBlur(equalized, (3, 3), 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sharpened):
        raise RuntimeError(f"Failed to write preprocessed plate image: {output_path}")
    return output_path
