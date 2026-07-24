from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.florence_vehicle_body_type_extractor import FlorenceVehicleBodyTypeExtractor
from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.florence_runtime import FlorenceRuntimeError


class _FakeRuntime:
    def __init__(self, output: str = "SUV", *, raise_error: Exception | None = None) -> None:
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


class FlorenceVehicleBodyTypeExtractorTests(unittest.TestCase):
    def test_valid_output_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vehicle.jpg"
            image_path.write_bytes(b"img")
            runtime = _FakeRuntime("SUV")
            extractor = FlorenceVehicleBodyTypeExtractor(
                runtime=runtime,
                prompt="prompt",
                allowed_body_types=("SUV", "UNKNOWN"),
                minimum_confidence=0.5,
                default_confidence_when_missing=0.6,
            )
            result = extractor.extract(image_path, source_storage_uri="vehicle_1/vehicle.jpg")
            self.assertEqual(result.canonical_body_type, "SUV")
            self.assertEqual(result.status, "SUCCESS")
            self.assertEqual(runtime.calls[0][0], image_path)

    def test_low_confidence_becomes_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vehicle.jpg"
            image_path.write_bytes(b"img")
            runtime = _FakeRuntime('{"body_type":"SUV","confidence":0.2}')
            extractor = FlorenceVehicleBodyTypeExtractor(
                runtime=runtime,
                prompt="prompt",
                allowed_body_types=("SUV", "UNKNOWN"),
                minimum_confidence=0.5,
                default_confidence_when_missing=0.5,
            )
            result = extractor.extract(image_path, source_storage_uri="vehicle_1/vehicle.jpg")
            self.assertEqual(result.canonical_body_type, "UNKNOWN")
            self.assertEqual(result.status, "LOW_CONFIDENCE")

    def test_missing_image_and_model_error_return_structured_status(self) -> None:
        runtime = _FakeRuntime(raise_error=FlorenceRuntimeError("boom"))
        extractor = FlorenceVehicleBodyTypeExtractor(
            runtime=runtime,
            prompt="prompt",
            allowed_body_types=("SUV", "UNKNOWN"),
            minimum_confidence=0.5,
            default_confidence_when_missing=0.5,
        )
        missing = extractor.extract(Path("missing.jpg"), source_storage_uri="vehicle_1/missing.jpg")
        self.assertEqual(missing.status, "MODEL_ERROR")


if __name__ == "__main__":
    unittest.main()
