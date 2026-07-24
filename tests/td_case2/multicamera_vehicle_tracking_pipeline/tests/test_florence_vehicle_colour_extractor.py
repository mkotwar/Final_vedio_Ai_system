from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_vehicle_colour_extractor import FlorenceVehicleColourExtractor
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_mapping import SUPPORTED_VEHICLE_COLOURS
from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.florence_runtime import FlorenceRuntimeError


class _FakeRuntime:
    def __init__(self, output: str = "</s><s>grey</s>", *, raise_error: Exception | None = None) -> None:
        self.output = output
        self.raise_error = raise_error
        self.calls: list[tuple[Path, str, bool]] = []
        self.model_path = Path("florence")
        self.adapter_path = Path("adapter")

    def run_image_task(self, *, image_path: Path, prompt: str, disable_adapter: bool = False) -> str:
        self.calls.append((image_path, prompt, disable_adapter))
        if self.raise_error is not None:
            raise self.raise_error
        return self.output


class FlorenceVehicleColourExtractorTests(unittest.TestCase):
    def test_valid_token_wrapped_colour_returns_success_and_preserves_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vehicle.jpg"
            image_path.write_bytes(b"img")
            runtime = _FakeRuntime("</s><s>grey</s>")
            extractor = FlorenceVehicleColourExtractor(
                runtime=runtime,
                prompt="prompt",
                allowed_colours=SUPPORTED_VEHICLE_COLOURS,
                minimum_confidence=0.5,
            )
            result = extractor.extract(image_path, track_uuid="track", camera_code="IMAGE", source_storage_uri="vehicle_1/vehicle.jpg")
            self.assertEqual(result.canonical_colour, "GREY")
            self.assertEqual(result.status, "SUCCESS")
            self.assertEqual(result.raw_output, "</s><s>grey</s>")
            self.assertEqual(result.metadata["cleaned_output"], "grey")

    def test_minimum_confidence_boundary_is_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vehicle.jpg"
            image_path.write_bytes(b"img")
            runtime = _FakeRuntime("</s><s>blue</s>")
            extractor = FlorenceVehicleColourExtractor(
                runtime=runtime,
                prompt="prompt",
                allowed_colours=SUPPORTED_VEHICLE_COLOURS,
                minimum_confidence=0.5,
            )
            result = extractor.extract(image_path, track_uuid="track", camera_code="IMAGE", source_storage_uri="vehicle_1/vehicle.jpg")
            self.assertEqual(result.canonical_colour, "BLUE")
            self.assertEqual(result.status, "SUCCESS")
            self.assertEqual(result.confidence, 0.5)

    def test_model_error_still_returns_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vehicle.jpg"
            image_path.write_bytes(b"img")
            runtime = _FakeRuntime(raise_error=FlorenceRuntimeError("boom"))
            extractor = FlorenceVehicleColourExtractor(
                runtime=runtime,
                prompt="prompt",
                allowed_colours=SUPPORTED_VEHICLE_COLOURS,
                minimum_confidence=0.5,
            )
            result = extractor.extract(image_path, track_uuid="track", camera_code="IMAGE", source_storage_uri="vehicle_1/vehicle.jpg")
            self.assertEqual(result.status, "MODEL_ERROR")
            self.assertEqual(result.canonical_colour, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
