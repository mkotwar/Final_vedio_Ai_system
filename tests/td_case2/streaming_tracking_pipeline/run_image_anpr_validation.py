from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import FlorenceConfig, PlateDetectionConfig
from .florence_inference import FlorenceInferenceEngine
from .image_anpr_validation import ImageAnprValidationConfig, ImageAnprValidator
from .plate_detection import UltralyticsPlateDetectionStage


DEFAULT_INPUT_DIR = Path(r"debug_runs\test pictures")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_optional_path(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (_repo_root() / path).resolve())


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("At least one diagnostic threshold is required.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Diagnostic thresholds must be between 0 and 1.")
    if len(set(values)) != len(values):
        raise ValueError("Diagnostic thresholds must be unique.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate plate YOLO and Florence OCR on an image folder.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--plate-detector-model", default=os.environ.get("TD_CASE2_STREAM_PLATE_MODEL_PATH", "OCR_MUKUL/license_plate_weights.pt"))
    parser.add_argument("--florence-model", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_MODEL_PATH") or os.environ.get("TD_CASE2_FLORENCE_MODEL_PATH"))
    parser.add_argument("--florence-adapter", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_ADAPTER_PATH") or os.environ.get("TD_CASE2_FLORENCE_ADAPTER_PATH") or "OCR_MUKUL/adaptor_florance_baseFT")
    parser.add_argument("--device", default=os.environ.get("TD_CASE2_STREAM_IMAGE_ANPR_DEVICE", "cpu"))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--normal-plate-confidence", type=float, default=0.25)
    parser.add_argument("--diagnostic-thresholds", default="0.25,0.15,0.10,0.05")
    parser.add_argument("--minimum-plate-width", type=int, default=6)
    parser.add_argument("--minimum-plate-height", type=int, default=4)
    parser.add_argument("--max-plate-candidates", type=int, default=3)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--direct-ocr-on-input", action="store_true")
    parser.add_argument("--stop-after-first-non-empty-ocr", action="store_true")
    parser.add_argument("--save-annotations", action="store_true")
    parser.add_argument("--save-rejected-plate-crops", action="store_true")
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--plate-iou", type=float, default=0.45)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    resolved_input = input_dir if input_dir.is_absolute() else (_repo_root() / input_dir).resolve()
    if not resolved_input.exists() or not resolved_input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {resolved_input}")
    thresholds = parse_thresholds(args.diagnostic_thresholds)
    run_dir = Path(args.output_root) / f"image_anpr_validation_{time.strftime('%Y%m%d_%H%M%S')}"
    plate_config = PlateDetectionConfig(
        model_path=resolve_optional_path(args.plate_detector_model),
        confidence_threshold=args.normal_plate_confidence,
        iou_threshold=args.plate_iou,
        device=args.device,
        image_size=args.image_size,
        max_plate_detections_per_vehicle_crop=args.max_plate_candidates,
        minimum_plate_width=args.minimum_plate_width,
        minimum_plate_height=args.minimum_plate_height,
        save_plate_crops=True,
    )
    florence_engine = None
    if args.run_ocr or args.direct_ocr_on_input:
        florence_config = FlorenceConfig(
            enabled=True,
            base_model_path=resolve_optional_path(args.florence_model),
            adapter_path=resolve_optional_path(args.florence_adapter),
            device=args.device,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        florence_engine = FlorenceInferenceEngine(florence_config, run_dir=run_dir)
    validator = ImageAnprValidator(
        config=ImageAnprValidationConfig(
            input_dir=str(resolved_input),
            recursive=args.recursive,
            normal_plate_confidence=args.normal_plate_confidence,
            diagnostic_thresholds=thresholds,
            minimum_plate_width=args.minimum_plate_width,
            minimum_plate_height=args.minimum_plate_height,
            maximum_plate_candidates_per_image=args.max_plate_candidates,
            save_annotations=args.save_annotations,
            save_rejected_plate_crops=args.save_rejected_plate_crops,
            run_florence_ocr=args.run_ocr,
            direct_ocr_on_input=args.direct_ocr_on_input,
            stop_after_first_non_empty_ocr=args.stop_after_first_non_empty_ocr,
            max_images=args.max_images,
        ),
        plate_detector=UltralyticsPlateDetectionStage(plate_config, output_dir=run_dir, run_dir=run_dir),
        florence_engine=florence_engine,
        output_dir=run_dir,
    )
    payload = validator.run()
    return {
        "status": "passed",
        "run_dir": str(run_dir),
        "summary": payload["summary"],
        "config": {
            "input_dir": str(resolved_input),
            "plate_model": plate_config.model_path,
            "florence_model": resolve_optional_path(args.florence_model),
            "florence_adapter": resolve_optional_path(args.florence_adapter),
            "device": args.device,
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "normal_plate_confidence": args.normal_plate_confidence,
            "diagnostic_thresholds": list(thresholds),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
