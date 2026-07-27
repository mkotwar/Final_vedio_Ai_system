from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_image_preprocessor import generate_plate_variants


class PlateImagePreprocessorTests(unittest.TestCase):
    def test_generates_bounded_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "plate.jpg"
            _write_test_image(image_path)
            variants = generate_plate_variants(image_path, output_directory=Path(tmpdir) / "variants", max_variants=6)
            self.assertEqual(len(variants), 6)
            self.assertIn("clahe", {variant.variant_name for variant in variants})
            self.assertIn("sharpened", {variant.variant_name for variant in variants})


def _write_test_image(path: Path) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and numpy are required for preprocessor tests.") from exc
    image = np.full((24, 64, 3), 200, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()
