from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..enrichment.anpr_config import load_anpr_config
from ..enrichment.best_plate_selector import select_best_plate_candidates
from ..enrichment.florence_config import load_florence_config
from ..enrichment.florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from ..enrichment.florence_vehicle_body_type_extractor import FlorenceVehicleBodyTypeExtractor
from ..enrichment.florence_vehicle_colour_extractor import FlorenceVehicleColourExtractor
from ..enrichment.plate_candidate_collector import PlateCandidateCollector
from ..enrichment.plate_models import PlateCandidate, VehicleEvidenceInput
from ..enrichment.vehicle_body_type_config import load_vehicle_body_type_config
from ..enrichment.vehicle_colour_config import load_vehicle_colour_config
from ..models.florence_runtime_factory import FlorenceRuntimeFactory
from ..models.plate_detector_runtime_factory import PlateDetectorRuntimeFactory


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANPR directly on one or more vehicle image files.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--plate-detector-model-path", default=None)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--anpr-config", required=True)
    parser.add_argument("--florence-config", required=True)
    parser.add_argument("--vehicle-colour-config", default=None)
    parser.add_argument("--vehicle-body-type-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--visual-output-dir", required=True)
    parser.add_argument("--save-all-plate-candidates", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")
    image_paths = _resolve_input_images(images=args.image, image_dir=args.image_dir)
    if not image_paths:
        raise ValueError("At least one input image is required via --image or --image-dir.")
    output_report = Path(args.output_report).expanduser().resolve()
    visual_output_dir = Path(args.visual_output_dir).expanduser().resolve()
    visual_output_dir.mkdir(parents=True, exist_ok=True)

    anpr_config = load_anpr_config(args.anpr_config, overrides={"enabled": True})
    florence_config = load_florence_config(args.florence_config, overrides={"enabled": True})
    vehicle_colour_config = load_vehicle_colour_config(args.vehicle_colour_config, overrides={"enabled": True}) if args.vehicle_colour_config else None
    vehicle_body_type_config = load_vehicle_body_type_config(args.vehicle_body_type_config, overrides={"enabled": True}) if args.vehicle_body_type_config else None
    project_root = Path.cwd()
    plate_runtime_factory = PlateDetectorRuntimeFactory(project_root=project_root)
    florence_runtime_factory = FlorenceRuntimeFactory(project_root=project_root)
    plate_runtime = plate_runtime_factory.get_runtime(
        config=anpr_config,
        model_path_cli=args.plate_detector_model_path,
        device_override=args.device,
    )
    florence_runtime = florence_runtime_factory.get_runtime(
        config=florence_config,
        model_path_cli=args.florence_model_path,
        adapter_path_cli=args.florence_adapter_path,
        processor_path_cli=args.florence_processor_path,
        device_override=args.device,
    )
    if plate_runtime is None or florence_runtime is None:
        raise RuntimeError("Plate detector and Florence runtime must both be enabled.")

    collector = PlateCandidateCollector(
        detector_runtime=plate_runtime,
        config=anpr_config,
        artifact_root=visual_output_dir,
    )
    ocr_extractor = FlorencePlateOcrExtractor(
        runtime=florence_runtime,
        ocr_config=anpr_config.ocr,
        validation_config=anpr_config.validation,
    )
    colour_extractor = (
        FlorenceVehicleColourExtractor(
            runtime=florence_runtime,
            prompt=florence_config.colour_prompt,
            allowed_colours=vehicle_colour_config.allowed_colours,
            minimum_confidence=vehicle_colour_config.minimum_confidence,
        )
        if vehicle_colour_config is not None and vehicle_colour_config.enabled
        else None
    )
    body_type_extractor = (
        FlorenceVehicleBodyTypeExtractor(
            runtime=florence_runtime,
            prompt=vehicle_body_type_config.prompt,
            allowed_body_types=vehicle_body_type_config.allowed_body_types,
            minimum_confidence=vehicle_body_type_config.minimum_confidence,
            default_confidence_when_missing=vehicle_body_type_config.default_confidence_when_missing,
        )
        if vehicle_body_type_config is not None and vehicle_body_type_config.enabled
        else None
    )

    results: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        LOGGER.info("Processing image %s/%s: %s", index, len(image_paths), image_path)
        result = _safe_process_image(
            image_path=image_path,
            visual_output_dir=visual_output_dir,
            collector=collector,
            ocr_extractor=ocr_extractor,
            colour_extractor=colour_extractor,
            body_type_extractor=body_type_extractor,
            anpr_config=anpr_config,
            save_all_plate_candidates=args.save_all_plate_candidates,
        )
        results.append(result)

    report = {
        "input_count": len(image_paths),
        "plate_detector_loaded_once": bool(getattr(plate_runtime, "loaded", False)),
        "florence_loaded_once": bool(getattr(florence_runtime, "loaded", False)),
        "colour_calls": sum(1 for item in results if item.get("vehicle_colour") is not None),
        "body_type_calls": sum(1 for item in results if item.get("vehicle_body_type") is not None),
        "ocr_calls": sum(1 for item in results if item.get("anpr", {}).get("raw_ocr") is not None),
        "results": results,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (visual_output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _resolve_input_images(*, images: list[str], image_dir: str | None) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in images:
        path = Path(raw).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    if image_dir:
        directory = Path(image_dir).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Input image directory does not exist: {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def _safe_process_image(
    *,
    image_path: Path,
    visual_output_dir: Path,
    collector: PlateCandidateCollector,
    ocr_extractor: FlorencePlateOcrExtractor,
    colour_extractor: FlorenceVehicleColourExtractor | None,
    body_type_extractor: FlorenceVehicleBodyTypeExtractor | None,
    anpr_config,
    save_all_plate_candidates: bool,
) -> dict[str, object]:
    try:
        return _process_image(
            image_path=image_path,
            visual_output_dir=visual_output_dir,
            collector=collector,
            ocr_extractor=ocr_extractor,
            colour_extractor=colour_extractor,
            body_type_extractor=body_type_extractor,
            anpr_config=anpr_config,
            save_all_plate_candidates=save_all_plate_candidates,
        )
    except Exception as exc:
        LOGGER.exception("Failed to process image: %s", image_path)
        return {
            "input_image": str(image_path),
            "vehicle_colour": None,
            "vehicle_body_type": None,
            "anpr": {
                "plate_detected": False,
                "raw_ocr": None,
                "normalized_ocr": None,
                "verification_status": "NOT_VERIFIED",
            },
            "errors": [str(exc)],
        }


def _process_image(
    *,
    image_path: Path,
    visual_output_dir: Path,
    collector: PlateCandidateCollector,
    ocr_extractor: FlorencePlateOcrExtractor,
    colour_extractor: FlorenceVehicleColourExtractor | None,
    body_type_extractor: FlorenceVehicleBodyTypeExtractor | None,
    anpr_config,
    save_all_plate_candidates: bool,
) -> dict[str, object]:
    import cv2  # type: ignore

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    image_key = image_path.stem
    image_output_dir = visual_output_dir / image_key
    image_output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_copy = image_output_dir / image_path.name
    if not vehicle_copy.exists():
        vehicle_copy.write_bytes(image_path.read_bytes())
    source_storage_uri = f"{image_key}/{image_path.name}"

    colour_result = None
    if colour_extractor is not None:
        colour_result = colour_extractor.extract(
            vehicle_copy,
            track_uuid=f"IMAGE:{image_key}",
            camera_code="IMAGE",
            source_storage_uri=source_storage_uri,
        )
    body_type_result = None
    if body_type_extractor is not None:
        body_type_result = body_type_extractor.extract(
            vehicle_copy,
            source_storage_uri=source_storage_uri,
        )

    evidence = VehicleEvidenceInput(
        track_uuid=f"IMAGE:{image_key}",
        camera_code="IMAGE",
        source_vehicle_role="BEST_OVERALL",
        source_vehicle_storage_uri=source_storage_uri,
        local_file_path=vehicle_copy,
        frame_number=0,
        video_time_seconds=0.0,
        confidence=1.0,
        bbox_xyxy=(0.0, 0.0, float(width), float(height)),
        crop_width=width,
        crop_height=height,
        sharpness_score=1.0,
        edge_penalty=0.0,
        overall_score=1.0,
    )
    candidates = collector.collect([evidence])
    selected = select_best_plate_candidates(
        candidates,
        maximum_for_ocr=anpr_config.ocr.maximum_plate_candidates_for_ocr,
        config=anpr_config.plate_selection,
    )

    selected_candidate = selected[0] if selected else None
    ocr_result = ocr_extractor.extract(selected_candidate) if selected_candidate is not None else None
    _save_visual_outputs(
        image=image,
        image_output_dir=image_output_dir,
        selected_candidate=selected_candidate,
        ocr_result=ocr_result,
        colour_result=colour_result,
        body_type_result=body_type_result,
        save_all_plate_candidates=save_all_plate_candidates,
        all_candidates=candidates,
    )

    status = "NO_PLATE_DETECTED"
    verification_status = "NOT_VERIFIED"
    raw_ocr = None
    normalized_ocr = None
    matched_pattern = None
    detector_confidence = None
    selected_plate_crop = None
    errors: list[str] = []
    if selected_candidate is not None:
        detector_confidence = selected_candidate.detector_confidence
        selected_plate_crop = f"{image_key}/best_plate.jpg"
    if ocr_result is not None:
        status = ocr_result.status
        verification_status = ocr_result.verification_status
        raw_ocr = ocr_result.raw_text
        normalized_ocr = ocr_result.normalized_text
        matched_pattern = ocr_result.metadata.get("matched_pattern")
    anpr_payload = {
        "status": status,
        "plate_detected": selected_candidate is not None,
        "plate_candidate_count": len(candidates),
        "selected_plate_crop": selected_plate_crop,
        "detector_confidence": detector_confidence,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized_ocr,
        "verification_status": verification_status,
        "matched_pattern": matched_pattern,
    }
    result = {
        "input_image": str(image_path),
        "vehicle_colour": colour_result.to_report_payload() if colour_result is not None else None,
        "vehicle_body_type": body_type_result.to_report_payload() if body_type_result is not None else None,
        "anpr": anpr_payload,
        "status": status,
        "plate_detected": selected_candidate is not None,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized_ocr,
        "verification_status": verification_status,
        "errors": errors,
    }
    (image_output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_visual_summary(
        image_output_dir=image_output_dir,
        colour_result=colour_result,
        body_type_result=body_type_result,
        anpr_payload=anpr_payload,
    )
    return result


def _save_visual_outputs(
    *,
    image,
    image_output_dir: Path,
    selected_candidate: PlateCandidate | None,
    ocr_result,
    colour_result,
    body_type_result,
    save_all_plate_candidates: bool,
    all_candidates: list[PlateCandidate],
) -> None:
    import cv2  # type: ignore

    annotated = image.copy()
    for index, candidate in enumerate(all_candidates, start=1):
        candidate_copy = image_output_dir / f"candidate_{index:03d}.jpg"
        if save_all_plate_candidates or (selected_candidate is not None and candidate.local_file_path == selected_candidate.local_file_path):
            candidate_copy.write_bytes(candidate.local_file_path.read_bytes())
    if selected_candidate is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in selected_candidate.plate_bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_lines = [
            f"colour: {colour_result.canonical_colour if colour_result is not None else 'N/A'}",
            f"body: {body_type_result.canonical_body_type if body_type_result is not None else 'N/A'}",
            f"raw: {ocr_result.raw_text if ocr_result is not None else 'N/A'}",
            f"normalized: {ocr_result.normalized_text if ocr_result is not None else 'N/A'}",
            f"status: {ocr_result.verification_status if ocr_result is not None else 'N/A'}",
        ]
        for idx, line in enumerate(text_lines):
            cv2.putText(annotated, line, (10, 30 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        best_plate_path = image_output_dir / "best_plate.jpg"
        best_plate_path.write_bytes(selected_candidate.local_file_path.read_bytes())
    cv2.imwrite(str(image_output_dir / "original_with_plate_box.jpg"), annotated)

    if not save_all_plate_candidates and selected_candidate is not None:
        for candidate in all_candidates:
            if candidate.local_file_path == selected_candidate.local_file_path:
                continue
            if candidate.local_file_path.exists():
                candidate.local_file_path.unlink()


def _save_visual_summary(
    *,
    image_output_dir: Path,
    colour_result,
    body_type_result,
    anpr_payload: dict[str, object],
) -> None:
    summary = {
        "vehicle_colour": colour_result.to_report_payload() if colour_result is not None else None,
        "vehicle_body_type": body_type_result.to_report_payload() if body_type_result is not None else None,
        "anpr": anpr_payload,
    }
    (image_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
