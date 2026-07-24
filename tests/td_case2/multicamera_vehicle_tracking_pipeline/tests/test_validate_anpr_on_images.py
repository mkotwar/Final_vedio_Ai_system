from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.anpr_config import AnprConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_models import PlateCandidate
from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import validate_anpr_on_images


class _FakeCollector:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.calls = []

    def collect(self, vehicle_evidence):
        self.calls.append(vehicle_evidence)
        plate_path = self.artifact_root / "vehicle_1" / "plate_evidence" / "candidate_001.jpg"
        plate_path.parent.mkdir(parents=True, exist_ok=True)
        plate_path.write_bytes(b"jpg")
        return [
            PlateCandidate(
                track_uuid="IMAGE:vehicle_1",
                camera_code="IMAGE",
                source_vehicle_role="BEST_OVERALL",
                source_vehicle_storage_uri="vehicle_1/vehicle.jpg",
                plate_bbox_xyxy=(1.0, 2.0, 30.0, 15.0),
                detector_confidence=0.9,
                crop_width=120,
                crop_height=40,
                area=4800,
                aspect_ratio=3.0,
                sharpness_score=0.8,
                edge_penalty=0.0,
                overall_score=0.9,
                local_file_path=plate_path,
                relative_storage_uri="vehicle_1/plate_evidence/candidate_001.jpg",
                frame_number=0,
                video_time_seconds=0.0,
            )
        ]


class _FakeOcrExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract(self, candidate):
        self.calls.append(candidate.local_file_path)
        from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.plate_models import PlateOcrResult

        return PlateOcrResult(
            raw_text="DL8CBF6268",
            normalized_text="DL8CBF6268",
            confidence=0.9,
            status="VERIFIED",
            verification_status="VERIFIED",
            country_profile="INDIA",
            backend="florence",
            model_name="florence",
            adapter_name="adapter",
            source_vehicle_track_id=candidate.track_uuid,
            source_plate_storage_uri=candidate.relative_storage_uri,
            source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
            metadata={"matched_pattern": "STANDARD"},
        )


class _FakeColourExtractor:
    def __init__(self, raw_output: str = "white", canonical_colour: str = "WHITE", status: str = "SUCCESS") -> None:
        self.raw_output = raw_output
        self.canonical_colour = canonical_colour
        self.status = status
        self.calls = []

    def extract(self, image_path, *, track_uuid: str, camera_code: str, source_storage_uri: str):
        self.calls.append(image_path)
        from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_colour_models import VehicleColourResult

        return VehicleColourResult(
            canonical_colour=self.canonical_colour,
            raw_output=self.raw_output,
            confidence=0.82 if self.status == "SUCCESS" else 0.5,
            status=self.status,
            source_storage_uri=source_storage_uri,
            metadata={"track_uuid": track_uuid, "camera_code": camera_code, "cleaned_output": self.raw_output.strip("</s><s>").strip("</s>")},
        )


class _WrappedColourExtractor(_FakeColourExtractor):
    def __init__(self) -> None:
        super().__init__(raw_output="</s><s>grey</s>", canonical_colour="GREY", status="SUCCESS")


class _FakeBodyTypeExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract(self, image_path, *, source_storage_uri: str | None = None):
        self.calls.append(image_path)
        from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.vehicle_body_type_models import VehicleBodyTypeResult

        return VehicleBodyTypeResult(
            canonical_body_type="HATCHBACK",
            raw_output="hatchback",
            confidence=0.76,
            status="SUCCESS",
            source_storage_uri=source_storage_uri,
        )


class ValidateAnprOnImagesTests(unittest.TestCase):
    def test_combined_image_result_contains_colour_body_type_and_anpr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            visual_output_dir = Path(tmpdir)
            image_path = visual_output_dir / "vehicle.jpg"
            _write_test_image(image_path)
            collector = _FakeCollector(visual_output_dir)
            colour_extractor = _FakeColourExtractor()
            body_type_extractor = _FakeBodyTypeExtractor()
            ocr_extractor = _FakeOcrExtractor()
            result = validate_anpr_on_images._process_image(
                image_path=image_path,
                visual_output_dir=visual_output_dir,
                collector=collector,
                ocr_extractor=ocr_extractor,
                colour_extractor=colour_extractor,
                body_type_extractor=body_type_extractor,
                anpr_config=AnprConfig(enabled=True),
                save_all_plate_candidates=True,
            )
            self.assertEqual(result["vehicle_colour"]["canonical_colour"], "WHITE")
            self.assertEqual(result["vehicle_body_type"]["canonical_body_type"], "HATCHBACK")
            self.assertEqual(result["anpr"]["normalized_ocr"], "DL8CBF6268")
            self.assertEqual(colour_extractor.calls[0].name, "vehicle.jpg")
            self.assertEqual(body_type_extractor.calls[0].name, "vehicle.jpg")
            self.assertNotEqual(ocr_extractor.calls[0].name, "vehicle.jpg")
            self.assertEqual(result["errors"], [])

    def test_wrapped_colour_outputs_are_preserved_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            visual_output_dir = Path(tmpdir)
            image_path = visual_output_dir / "vehicle.jpg"
            _write_test_image(image_path)
            result = validate_anpr_on_images._process_image(
                image_path=image_path,
                visual_output_dir=visual_output_dir,
                collector=_FakeCollector(visual_output_dir),
                ocr_extractor=_FakeOcrExtractor(),
                colour_extractor=_WrappedColourExtractor(),
                body_type_extractor=_FakeBodyTypeExtractor(),
                anpr_config=AnprConfig(enabled=True),
                save_all_plate_candidates=True,
            )
            self.assertEqual(result["vehicle_colour"]["canonical_colour"], "GREY")
            self.assertEqual(result["vehicle_colour"]["status"], "SUCCESS")
            self.assertEqual(result["vehicle_colour"]["raw_output"], "</s><s>grey</s>")

    def test_safe_process_image_returns_error_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_anpr_on_images._safe_process_image(
                image_path=Path(tmpdir) / "missing.jpg",
                visual_output_dir=Path(tmpdir),
                collector=_FakeCollector(Path(tmpdir)),
                ocr_extractor=_FakeOcrExtractor(),
                colour_extractor=None,
                body_type_extractor=None,
                anpr_config=AnprConfig(enabled=True),
                save_all_plate_candidates=False,
            )
            self.assertEqual(result["anpr"]["plate_detected"], False)
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()


def _write_test_image(path: Path) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and numpy are required for validator tests.") from exc
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test image: {path}")
