from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LICENSE_PLATE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}
OCR_MUKUL_BASE_MODEL_ID = "microsoft/Florence-2-base-ft"
VALID_INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ",
    "HR", "HP", "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ",
    "NL", "OR", "PY", "PB", "RJ", "SK", "TN", "TR", "UP", "WB", "TS",
    "UK", "LA", "CG", "JH",
}
ENV_ENABLE_PLATE_COLOR_ENRICHMENT = "TENDER_DEMO_ENABLE_PLATE_COLOR_ENRICHMENT"
ENV_FLORENCE_LOCAL_FILES_ONLY = "TENDER_DEMO_FLORENCE_LOCAL_FILES_ONLY"
ENV_FLORENCE_MODEL_PATH = "TENDER_DEMO_FLORENCE_MODEL_PATH"

_PLATE_DETECTOR_MODEL: Any | None = None
_PLATE_DETECTOR_ATTEMPTED = False
_FLORENCE_STATE: dict[str, Any] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


OCR_MUKUL_DIR = _repo_root() / "object_yolo" / "OCR_MUKUL"
OCR_MUKUL_PLATE_MODEL_PATH = OCR_MUKUL_DIR / "license_plate_weights.pt"
OCR_MUKUL_FLORENCE_ADAPTER_PATH = OCR_MUKUL_DIR / "adaptor_florance_baseFT"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except Exception:
        return str(path)


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Ultralytics is required. Install it with: pip install ultralytics") from exc
    return YOLO(model_name)


def _load_optional_plate_detector():
    global _PLATE_DETECTOR_MODEL, _PLATE_DETECTOR_ATTEMPTED
    if _PLATE_DETECTOR_MODEL is not None:
        return _PLATE_DETECTOR_MODEL
    if _PLATE_DETECTOR_ATTEMPTED:
        return None
    _PLATE_DETECTOR_ATTEMPTED = True
    if not OCR_MUKUL_PLATE_MODEL_PATH.exists():
        return None
    try:
        _PLATE_DETECTOR_MODEL = _load_yolo(str(OCR_MUKUL_PLATE_MODEL_PATH))
    except Exception:
        _PLATE_DETECTOR_MODEL = None
    return _PLATE_DETECTOR_MODEL


def _load_optional_florence_bundle() -> dict[str, Any]:
    global _FLORENCE_STATE
    if _FLORENCE_STATE is not None:
        return _FLORENCE_STATE

    state: dict[str, Any] = {
        "available": False,
        "status": "unavailable",
        "device": "cpu",
        "model": None,
        "processor": None,
        "using_adapter": False,
        "adapter_status": "not_attempted",
        "processor_source": None,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception:
        _FLORENCE_STATE = state
        return state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    local_model_raw = os.environ.get(ENV_FLORENCE_MODEL_PATH, "").strip()
    if not local_model_raw:
        state["status"] = "skipped_local_model_path_not_set"
        _FLORENCE_STATE = state
        return state

    local_model_path = Path(local_model_raw).expanduser()
    if not local_model_path.is_absolute():
        local_model_path = (_repo_root() / local_model_path).resolve()
    else:
        local_model_path = local_model_path.resolve()
    state["processor_source"] = str(local_model_path)

    if not local_model_path.exists():
        state["status"] = "skipped_local_model_missing"
        _FLORENCE_STATE = state
        return state

    try:
        processor_source = str(local_model_path)
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            processor_source,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True,
        ).to(device)
        model.eval()
        state.update(
            {
                "available": True,
                "status": "loaded_offline",
                "model": model,
                "processor": processor,
                "using_adapter": False,
                "processor_source": processor_source,
                "adapter_path": str(OCR_MUKUL_FLORENCE_ADAPTER_PATH),
            }
        )
    except Exception as exc:
        state["status"] = f"load_failed:{exc.__class__.__name__}"
        state["error_message"] = str(exc).strip()[:500]
    _FLORENCE_STATE = state
    return state


def _resize_proportionally_if_needed(image: np.ndarray, target_width: int = 640, target_height: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    if w < target_width or h < target_height:
        scale_factor = max(target_width / w, target_height / h)
        return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return image


def _is_valid_indian_plate(text: str) -> bool:
    clean_text = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    if len(clean_text) > 10:
        return False
    if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", clean_text):
        return clean_text[:2] in VALID_INDIAN_STATE_CODES
    return bool(re.match(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$", clean_text))


def _prepare_plate_crop(plate_crop_raw: np.ndarray) -> np.ndarray | None:
    if plate_crop_raw.size == 0:
        return None
    plate_crop = _resize_proportionally_if_needed(plate_crop_raw)
    ycrcb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_channel)
    plate_crop = cv2.cvtColor(cv2.merge([y_eq, cr_channel, cb_channel]), cv2.COLOR_YCrCb2BGR)
    return cv2.GaussianBlur(plate_crop, (3, 3), 0)


def _detect_plate_crop_from_vehicle(vehicle_crop: np.ndarray) -> tuple[np.ndarray | None, float | None, str]:
    if vehicle_crop.size == 0:
        return None, None, "vehicle_crop_unavailable"
    plate_model = _load_optional_plate_detector()
    if plate_model is None:
        return None, None, "plate_detector_unavailable"
    try:
        plate_results = plate_model(vehicle_crop, conf=0.5, verbose=False)
        if not plate_results or len(plate_results[0].boxes) == 0:
            return None, None, "no_plate_detection"
        best_plate = max(plate_results[0].boxes, key=lambda box: float(box.conf[0]))
        px1, py1, px2, py2 = map(int, best_plate.xyxy[0])
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(vehicle_crop.shape[1], px2)
        py2 = min(vehicle_crop.shape[0], py2)
        plate_crop_raw = vehicle_crop[py1:py2, px1:px2]
        prepared_crop = _prepare_plate_crop(plate_crop_raw)
        if prepared_crop is None:
            return None, None, "plate_crop_invalid"
        return prepared_crop, float(best_plate.conf[0]), "detected"
    except Exception:
        return None, None, "plate_detection_failed"


def _run_florence_inference(image_cv: np.ndarray, task_prompt: str, text_input: str | None = None) -> str | None:
    florence_state = _load_optional_florence_bundle()
    if not florence_state.get("available"):
        return None
    try:
        import torch
        from PIL import Image
    except Exception:
        return None

    model = florence_state["model"]
    processor = florence_state["processor"]
    device = str(florence_state.get("device", "cpu"))
    prompt = task_prompt + (text_input or "")
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_pil.width, image_pil.height),
    )
    value = parsed.get(task_prompt)
    return str(value).strip() if value is not None else None


def _format_seconds(value: float) -> str:
    total = max(0.0, float(value))
    minutes = int(total // 60)
    seconds = total - (minutes * 60)
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{minutes:02d}:{int(round(seconds)):02d}"
    return f"{minutes:02d}:{seconds:04.1f}"


def _extract_vehicle_plate_metadata(
    *,
    class_name: str,
    crop: np.ndarray,
    plate_crops_dir: Path,
    detection_key: str,
    timestamp_seconds: float,
) -> dict[str, Any]:
    result = {
        "plate_ocr_status": "not_applicable",
        "plate_crop_path": None,
        "plate_detection_status": "not_applicable",
        "license_plate_confidence": None,
        "plate_detected_text": [],
        "license_plates": [],
        "best_license_plate": None,
        "vehicle_color_florence": None,
        "vehicle_color_status": "not_applicable",
        "best_plate_timestamp_seconds": round(float(timestamp_seconds), 3),
        "best_plate_timestamp_text": _format_seconds(float(timestamp_seconds)),
    }
    if class_name not in LICENSE_PLATE_CLASS_NAMES:
        return result

    vehicle_color = _run_florence_inference(crop, "<VQA>", "What is the primary color of the vehicle?")
    if vehicle_color:
        result["vehicle_color_florence"] = vehicle_color.strip().lower()
        result["vehicle_color_status"] = "success"
    else:
        result["vehicle_color_status"] = "florence_unavailable_or_failed"

    plate_region, plate_confidence, plate_detection_status = _detect_plate_crop_from_vehicle(crop)
    result["plate_detection_status"] = plate_detection_status
    result["license_plate_confidence"] = round(float(plate_confidence), 4) if plate_confidence is not None else None
    if plate_region is None:
        result["plate_ocr_status"] = plate_detection_status
        return result

    plate_crop_path = plate_crops_dir / f"{detection_key}_plate.jpg"
    cv2.imwrite(str(plate_crop_path), plate_region)
    result["plate_crop_path"] = _to_repo_relative(plate_crop_path)
    extracted_text = _run_florence_inference(plate_region, "<OCR>")
    if not extracted_text:
        result["plate_ocr_status"] = "florence_ocr_unavailable_or_failed"
        return result

    normalized_text = re.sub(r"[^A-Z0-9]", "", extracted_text.upper())
    detected_text = [str(extracted_text).strip()] if str(extracted_text).strip() else []
    license_plates = [normalized_text] if normalized_text and _is_valid_indian_plate(normalized_text) else []
    result["plate_detected_text"] = list(dict.fromkeys(detected_text + ([normalized_text] if normalized_text else [])))
    result["license_plates"] = list(dict.fromkeys(license_plates))
    result["best_license_plate"] = result["license_plates"][0] if result["license_plates"] else None
    result["plate_ocr_status"] = "success" if result["license_plates"] else "ocr_attempted_no_match"
    return result


def _read_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_PLATE_COLOR_ENRICHMENT, "true").strip().lower() == "true"


def run_plate_ocr_color_enrichment(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 11C: plate OCR and Florence color enrichment")
    output_path = run_dir / "11c_plate_ocr_color_enrichment.json"
    report_path = run_dir / "11c_plate_ocr_color_report.json"

    if not _read_enabled():
        report = {
            "enabled": False,
            "status": "disabled_by_env",
            "frames_processed": 0,
            "vehicle_detections_considered": 0,
            "enriched_vehicle_detections": 0,
            "detections_with_plate_crop": 0,
            "detections_with_license_plate": 0,
            "detections_with_vehicle_color": 0,
            "plate_detector_available": OCR_MUKUL_PLATE_MODEL_PATH.exists(),
            "florence_status": "skipped",
        }
        output_path.write_text(json.dumps([], indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    yolo_scores_path = run_dir / "11_yolo_object_scores.json"
    if not yolo_scores_path.exists():
        raise FileNotFoundError(f"Missing YOLO object scores file: {yolo_scores_path}")

    yolo_scores = json.loads(yolo_scores_path.read_text(encoding="utf-8"))
    if not isinstance(yolo_scores, list):
        raise ValueError(f"Expected a list in {yolo_scores_path}")

    plate_crops_dir = _ensure_dir(run_dir / "11c_plate_crops")
    enriched_items: list[dict[str, Any]] = []
    vehicle_detections_considered = 0
    detections_with_plate_crop = 0
    detections_with_license_plate = 0
    detections_with_vehicle_color = 0
    enriched_vehicle_detections = 0

    for score_item in yolo_scores:
        if not isinstance(score_item, dict):
            continue
        detections = score_item.get("detections", [])
        if not isinstance(detections, list):
            continue

        frame_idx = int(score_item.get("frame_idx", 0) or 0)
        timestamp_seconds = float(score_item.get("timestamp_seconds", 0.0) or 0.0)
        enriched_detections: list[dict[str, Any]] = []

        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            class_name = str(detection.get("class_name", "")).strip().lower()
            if class_name not in LICENSE_PLATE_CLASS_NAMES:
                continue
            vehicle_detections_considered += 1

            crop_path = _to_abs_path(detection.get("crop_path"))
            crop = cv2.imread(str(crop_path)) if crop_path and crop_path.exists() else None
            detection_key = f"frame_{frame_idx:06d}_{detection_index:02d}_{class_name}"

            metadata = {
                "detection_key": detection_key,
                "frame_idx": frame_idx,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "object_class_name": class_name,
                "object_confidence": round(float(detection.get("confidence", 0.0) or 0.0), 4),
                "object_crop_path": detection.get("crop_path"),
                "annotated_frame_path": score_item.get("annotated_frame_path"),
            }
            if crop is None or crop.size == 0:
                metadata.update(
                    {
                        "plate_ocr_status": "object_crop_unavailable",
                        "plate_crop_path": None,
                        "plate_detection_status": "object_crop_unavailable",
                        "license_plate_confidence": None,
                        "plate_detected_text": [],
                        "license_plates": [],
                        "best_license_plate": None,
                        "vehicle_color_florence": None,
                        "vehicle_color_status": "object_crop_unavailable",
                    }
                )
            else:
                enrichment = _extract_vehicle_plate_metadata(
                    class_name=class_name,
                    crop=crop,
                    plate_crops_dir=plate_crops_dir,
                    detection_key=detection_key,
                    timestamp_seconds=timestamp_seconds,
                )
                metadata.update(enrichment)
                if metadata.get("plate_crop_path"):
                    detections_with_plate_crop += 1
                if metadata.get("best_license_plate"):
                    detections_with_license_plate += 1
                if metadata.get("vehicle_color_florence"):
                    detections_with_vehicle_color += 1
            enriched_vehicle_detections += 1
            enriched_detections.append(metadata)

        if enriched_detections:
            enriched_items.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "frame_path": score_item.get("frame_path"),
                    "annotated_frame_path": score_item.get("annotated_frame_path"),
                    "detections": enriched_detections,
                }
            )

    florence_state = _load_optional_florence_bundle()
    report = {
        "enabled": True,
        "status": "success",
        "frames_processed": len(enriched_items),
        "vehicle_detections_considered": vehicle_detections_considered,
        "enriched_vehicle_detections": enriched_vehicle_detections,
        "detections_with_plate_crop": detections_with_plate_crop,
        "detections_with_license_plate": detections_with_license_plate,
        "detections_with_vehicle_color": detections_with_vehicle_color,
        "plate_detector_available": OCR_MUKUL_PLATE_MODEL_PATH.exists(),
        "ocr_assets_dir": str(OCR_MUKUL_DIR),
        "plate_detector_path": str(OCR_MUKUL_PLATE_MODEL_PATH),
        "florence_processor_source": florence_state.get("processor_source"),
        "florence_adapter_path": florence_state.get("adapter_path", str(OCR_MUKUL_FLORENCE_ADAPTER_PATH)),
        "florence_adapter_status": florence_state.get("adapter_status", "unknown"),
        "florence_status": florence_state.get("status", "unavailable"),
        "florence_device": florence_state.get("device", "cpu"),
        "florence_error_message": florence_state.get("error_message"),
        "sample_detections": [
            item
            for frame_item in enriched_items[:5]
            for item in frame_item.get("detections", [])[:2]
        ][:10],
    }

    if florence_state.get("status") != "loaded_offline":
        report["detections_with_license_plate"] = 0
        report["detections_with_vehicle_color"] = 0

    output_path.write_text(json.dumps(enriched_items, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Vehicle detections considered: {vehicle_detections_considered}")
    print(f"[tender-demo] Detections with vehicle color: {detections_with_vehicle_color}")
    print(f"[tender-demo] Detections with license plate: {detections_with_license_plate}")
    print(f"[tender-demo] Plate/color enrichment output path: {output_path}")
    return report
