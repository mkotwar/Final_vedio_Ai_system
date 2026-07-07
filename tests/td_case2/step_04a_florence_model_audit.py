from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import cv2

from stage_checks import read_json, write_json


PREFERRED_VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bike",
    "bus",
    "truck",
    "auto",
    "van",
    "vehicle",
}
COLOR_WORDS = [
    "white",
    "black",
    "red",
    "blue",
    "yellow",
    "grey",
    "gray",
    "silver",
    "green",
    "brown",
    "orange",
]
BASE_MODEL_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processing_florence2.py",
    "modeling_florence2.py",
]
ADAPTER_EXPECTED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processing_florence2.py",
]


def build_json_prompt(task_mode: str) -> str:
    """Build the optional legacy JSON prompt test."""

    mode_line = {
        "ocr": "Focus mainly on OCR text and visibility.",
        "color": "Focus mainly on the dominant visible color.",
        "ocr_and_color": "Return both OCR text and the main visible color.",
    }[task_mode]
    return (
        "You are analyzing a vehicle/object crop from a road CCTV camera.\n"
        "Return only valid JSON.\n\n"
        "Detect:\n"
        "1. Any visible license plate text or readable OCR text.\n"
        "2. Main visible vehicle/object color.\n\n"
        "Rules:\n"
        "- Do not guess plate text.\n"
        '- If plate text is not readable, return "not_visible".\n'
        '- If color is unclear, return "unknown".\n'
        "- Use simple color names like white, black, red, blue, yellow, grey, silver, green, brown, orange.\n"
        "- Return only one JSON object.\n"
        f"{mode_line}\n\n"
        "JSON format:\n"
        "{\n"
        '  "ocr_text": "...",\n'
        '  "vehicle_color": "...",\n'
        '  "visibility": "clear|partial|not_visible",\n'
        '  "notes": "short observable note"\n'
        "}"
    )


def select_audit_crops(run_dir: Path, audit_limit: int) -> list[dict[str, Any]]:
    """Select a small set of useful YOLO crops for Florence auditing."""

    yolo_payload = read_json(run_dir / "03_yolo_detections.json")
    frame_items = list(yolo_payload.get("detections", []))
    candidates: list[dict[str, Any]] = []

    for frame_item in frame_items:
        for detection in list(frame_item.get("detections", [])):
            crop_path_value = str(detection.get("crop_path", ""))
            crop_path = (run_dir / Path(crop_path_value)).resolve() if crop_path_value else None
            candidates.append(
                {
                    "source_frame_id": str(frame_item.get("frame_id", "")),
                    "frame_idx": int(frame_item.get("frame_idx", 0) or 0),
                    "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
                    "detection_id": str(detection.get("detection_id", "")),
                    "class_name": str(detection.get("class_name", "")),
                    "yolo_confidence": float(detection.get("confidence", 0.0) or 0.0),
                    "bbox_xyxy": list(detection.get("bbox_xyxy", [])),
                    "bbox_area_ratio": float(detection.get("bbox_area_ratio", 0.0) or 0.0),
                    "crop_path": crop_path_value,
                    "crop_exists": bool(crop_path and crop_path.exists()),
                    "absolute_crop_path": crop_path,
                }
            )

    vehicle_candidates = [item for item in candidates if item["class_name"].lower() in PREFERRED_VEHICLE_CLASSES]
    selected_pool = vehicle_candidates if vehicle_candidates else candidates
    selected_pool = [item for item in selected_pool if item["crop_path"]]
    selected_pool.sort(
        key=lambda item: (
            1 if item["crop_exists"] else 0,
            item["yolo_confidence"],
            item["bbox_area_ratio"],
        ),
        reverse=True,
    )
    return selected_pool[:audit_limit]


def copy_audit_input(audit_inputs_dir: Path, crop_item: dict[str, Any], enabled: bool) -> str:
    """Copy audit inputs for easier manual review."""

    if not enabled:
        return ""
    source_crop_path = crop_item.get("absolute_crop_path")
    if not isinstance(source_crop_path, Path) or not source_crop_path.exists():
        return ""
    output_name = f"{crop_item['audit_index']:03d}_{crop_item['detection_id']}_{source_crop_path.name}"
    output_path = audit_inputs_dir / output_name
    shutil.copy2(source_crop_path, output_path)
    return str(Path("04A_florence_audit_inputs") / output_name).replace("\\", "/")


def inspect_path_files(path: Path | None, expected_files: list[str]) -> dict[str, Any]:
    """Inspect required files under a model or adapter directory."""

    if path is None:
        return {
            "path_value": None,
            "path_exists": False,
            "found_files": [],
            "missing_files": list(expected_files),
        }

    found_files = [name for name in expected_files if (path / name).exists()]
    return {
        "path_value": str(path),
        "path_exists": path.exists(),
        "found_files": found_files,
        "missing_files": [name for name in expected_files if name not in found_files],
    }


def load_florence_model(model_path: Path, device: str) -> tuple[Any, Any, str]:
    """Load Florence processor and model from local files only."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import Florence dependencies: {exc}") from exc

    if device == "auto":
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_used = device

    # local_files_only=True prevents any online Hugging Face download during this isolated audit.
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device_used)
    model.eval()
    return processor, model, device_used


def run_florence_generation(
    *,
    image_path: Path,
    processor: Any,
    model: Any,
    device_used: str,
    task_prompt: str,
    max_new_tokens: int,
    num_beams: int,
) -> dict[str, Any]:
    """Run a single Florence-native task or optional JSON prompt."""

    import torch

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read crop image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    started_at = time.perf_counter()
    inputs = processor(text=task_prompt, images=image_rgb, return_tensors="pt")
    inputs = {key: value.to(device_used) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    decoded_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
    raw_decoded_text = decoded_list[0].strip() if decoded_list else ""
    post_processed_output: Any = None
    try:
        post_processed_output = processor.post_process_generation(
            raw_decoded_text,
            task=task_prompt,
            image_size=(width, height),
        )
    except Exception:
        post_processed_output = None

    elapsed_seconds = time.perf_counter() - started_at
    generated_token_count = 0
    generated_ids_shape = []
    if hasattr(generated_ids, "shape"):
        generated_ids_shape = list(generated_ids.shape)
        if len(generated_ids_shape) >= 2:
            generated_token_count = int(generated_ids_shape[-1])

    status = "success" if raw_decoded_text or (post_processed_output not in (None, "", {}, [])) else "failed"
    return {
        "task_name": task_prompt.strip("<>"),
        "task_prompt": task_prompt,
        "raw_decoded_text": raw_decoded_text,
        "post_processed_output": post_processed_output,
        "generated_token_count": generated_token_count,
        "generated_ids_shape": generated_ids_shape,
        "status": status,
        "error_message": None,
        "processing_seconds": round(elapsed_seconds, 6),
    }


def parse_florence_json(raw_output: str) -> dict[str, str]:
    """Parse optional JSON-prompt Florence output."""

    json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if not json_match:
        raise ValueError("Florence output did not contain a JSON object.")
    payload = json.loads(json_match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Florence JSON output was not an object.")
    return {
        "ocr_text": str(payload.get("ocr_text", "not_visible")),
        "vehicle_color": str(payload.get("vehicle_color", "unknown")),
        "visibility": str(payload.get("visibility", "not_visible")),
        "notes": str(payload.get("notes", "")),
    }


def extract_color_from_text(text: str) -> str:
    """Pick a simple color word from caption-like output."""

    normalized = text.lower()
    for color in COLOR_WORDS:
        if re.search(rf"\b{re.escape(color)}\b", normalized):
            return "grey" if color == "gray" else color
    return "unknown"


def stringify_result_content(result: dict[str, Any]) -> str:
    """Flatten native-task outputs into a simple string for heuristics."""

    raw_text = str(result.get("raw_decoded_text", "") or "").strip()
    post_processed_output = result.get("post_processed_output")
    if isinstance(post_processed_output, str):
        return f"{raw_text} {post_processed_output}".strip()
    if isinstance(post_processed_output, dict):
        return f"{raw_text} {json.dumps(post_processed_output, ensure_ascii=False)}".strip()
    return raw_text


def best_ocr_from_results(native_task_results: list[dict[str, Any]], json_prompt_result: dict[str, Any] | None) -> tuple[str, str]:
    """Choose the best available OCR source from native or JSON tasks."""

    for item in native_task_results:
        if item.get("task_prompt") == "<OCR>":
            text = stringify_result_content(item).strip()
            if text:
                return text, "native_ocr"
    if json_prompt_result and json_prompt_result.get("status") == "success":
        text = str(json_prompt_result.get("parsed_ocr_text", "not_visible")).strip()
        if text and text != "not_visible":
            return text, "json_prompt"
    return "not_visible", "none"


def best_color_from_results(native_task_results: list[dict[str, Any]], json_prompt_result: dict[str, Any] | None) -> tuple[str, str]:
    """Choose the best available color source from captions or optional JSON."""

    for item in native_task_results:
        if item.get("task_prompt") in {"<DETAILED_CAPTION>", "<CAPTION>"}:
            color = extract_color_from_text(stringify_result_content(item))
            if color != "unknown":
                return color, item["task_name"].lower()
    if json_prompt_result and json_prompt_result.get("status") == "success":
        color = str(json_prompt_result.get("parsed_vehicle_color", "unknown")).strip().lower()
        if color and color != "unknown":
            return color, "json_prompt"
    return "unknown", "none"


def load_plate_detector(model_path: Path | None) -> tuple[str, Any | None]:
    """Load an optional plate detector model without failing the whole audit."""

    if model_path is None:
        return "not_provided", None
    if not model_path.exists():
        return "failed", None
    try:
        from ultralytics import YOLO  # type: ignore
        return "success", YOLO(str(model_path))
    except Exception:
        return "failed", None


def run_plate_detector_on_crop(
    *,
    crop_path: Path,
    plate_detector: Any,
    plate_audit_dir: Path,
    detection_id: str,
) -> dict[str, Any]:
    """Run optional plate detection on a vehicle crop and save plate crops."""

    image = cv2.imread(str(crop_path))
    if image is None:
        raise RuntimeError(f"Failed to read crop image for plate detection: {crop_path}")

    results = plate_detector.predict(source=str(crop_path), conf=0.25, iou=0.45, verbose=False)
    plate_crop_paths: list[str] = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            for index, bbox_xyxy in enumerate(boxes.xyxy.tolist(), start=1):
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                plate_crop = image[y1:y2, x1:x2]
                plate_name = f"{detection_id}_plate_{index:03d}.jpg"
                plate_output_path = plate_audit_dir / plate_name
                if cv2.imwrite(str(plate_output_path), plate_crop):
                    plate_crop_paths.append(str(Path("04A_plate_audit_crops") / plate_name).replace("\\", "/"))
    return {
        "plate_crop_paths": plate_crop_paths,
        "plate_crop_count": len(plate_crop_paths),
    }


def run_florence_audit(
    *,
    run_dir: Path,
    florence_model_path: Path,
    florence_adapter_path: Path | None,
    task_mode: str,
    audit_limit: int,
    max_new_tokens: int,
    num_beams: int,
    native_tasks: list[str],
    run_json_prompt_test: bool,
    save_audit_inputs: bool,
    device: str,
    plate_detector_model_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run an offline Florence health-check audit on a few YOLO crops."""

    audit_inputs_dir = run_dir / "04A_florence_audit_inputs"
    audit_inputs_dir.mkdir(parents=True, exist_ok=True)
    plate_audit_dir = run_dir / "04A_plate_audit_crops"
    plate_audit_dir.mkdir(parents=True, exist_ok=True)

    base_model_files = inspect_path_files(florence_model_path, BASE_MODEL_REQUIRED_FILES)
    adapter_files_summary = inspect_path_files(florence_adapter_path, ADAPTER_EXPECTED_FILES)
    adapter_path_exists = bool(florence_adapter_path and florence_adapter_path.exists())
    adapter_load_status = "not_provided" if florence_adapter_path is None else "failed"
    if adapter_path_exists:
        adapter_load_status = "success" if not adapter_files_summary["missing_files"] else "failed"

    selected_crops = select_audit_crops(run_dir, audit_limit)
    plate_detector_load_status, plate_detector = load_plate_detector(plate_detector_model_path)

    if not florence_model_path.exists():
        raise FileNotFoundError(
            "Base Florence model folder is missing. Adapter folder alone is not enough. "
            'Suggested setting: TD_CASE2_FLORENCE_MODEL_PATH="C:\\Mukul K\\models\\Florence-2-base-ft"'
        )

    processor, model, device_used = load_florence_model(florence_model_path, device)
    results: list[dict[str, Any]] = []
    successful_outputs = 0
    failed_outputs = 0
    missing_crop_count = 0
    native_task_success_count = 0
    ocr_native_success_count = 0
    caption_success_count = 0
    detailed_caption_success_count = 0
    json_prompt_success_count = 0
    plate_crop_count = 0
    plate_ocr_success_count = 0
    processing_seconds_values: list[float] = []

    for audit_index, crop_item in enumerate(selected_crops, start=1):
        crop_item["audit_index"] = audit_index
        crop_path = crop_item.get("absolute_crop_path")
        audit_input_copy_path = copy_audit_input(audit_inputs_dir, crop_item, save_audit_inputs)
        result_payload = {
            "audit_index": audit_index,
            "source_frame_id": crop_item["source_frame_id"],
            "frame_idx": crop_item["frame_idx"],
            "timestamp_seconds": crop_item["timestamp_seconds"],
            "detection_id": crop_item["detection_id"],
            "class_name": crop_item["class_name"],
            "yolo_confidence": crop_item["yolo_confidence"],
            "bbox_xyxy": crop_item["bbox_xyxy"],
            "bbox_area_ratio": crop_item["bbox_area_ratio"],
            "crop_path": crop_item["crop_path"],
            "audit_input_copy_path": audit_input_copy_path,
            "native_task_results": [],
            "json_prompt_result": None,
            "plate_detector_result": None,
            "parsed_ocr_text": "not_visible",
            "parsed_vehicle_color": "unknown",
            "best_available_ocr_source": "none",
            "best_available_color_source": "none",
            "status": "failed",
            "error_message": None,
            "processing_seconds": 0.0,
        }

        if not isinstance(crop_path, Path) or not crop_path.exists():
            missing_crop_count += 1
            failed_outputs += 1
            result_payload["error_message"] = f"Crop file does not exist: {crop_item['crop_path']}"
            results.append(result_payload)
            continue

        crop_started_at = time.perf_counter()
        try:
            for task_prompt in native_tasks:
                task_result = run_florence_generation(
                    image_path=crop_path,
                    processor=processor,
                    model=model,
                    device_used=device_used,
                    task_prompt=task_prompt,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                result_payload["native_task_results"].append(task_result)
                if task_result["status"] == "success":
                    native_task_success_count += 1
                    if task_prompt == "<OCR>":
                        ocr_native_success_count += 1
                    elif task_prompt == "<CAPTION>":
                        caption_success_count += 1
                    elif task_prompt == "<DETAILED_CAPTION>":
                        detailed_caption_success_count += 1

            if run_json_prompt_test:
                json_prompt_started = time.perf_counter()
                json_prompt_result = {
                    "task_name": "JSON_PROMPT",
                    "task_prompt": build_json_prompt(task_mode),
                    "raw_decoded_text": "",
                    "post_processed_output": None,
                    "generated_token_count": 0,
                    "generated_ids_shape": [],
                    "status": "failed",
                    "error_message": None,
                    "processing_seconds": 0.0,
                    "parsed_ocr_text": "not_visible",
                    "parsed_vehicle_color": "unknown",
                    "parsed_visibility": "not_visible",
                    "parsed_notes": "",
                }
                try:
                    generation_result = run_florence_generation(
                        image_path=crop_path,
                        processor=processor,
                        model=model,
                        device_used=device_used,
                        task_prompt=json_prompt_result["task_prompt"],
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                    json_prompt_result.update(generation_result)
                    parsed_json = parse_florence_json(json_prompt_result["raw_decoded_text"])
                    json_prompt_result["parsed_ocr_text"] = parsed_json["ocr_text"]
                    json_prompt_result["parsed_vehicle_color"] = parsed_json["vehicle_color"]
                    json_prompt_result["parsed_visibility"] = parsed_json["visibility"]
                    json_prompt_result["parsed_notes"] = parsed_json["notes"]
                    json_prompt_result["status"] = "success"
                    json_prompt_success_count += 1
                except Exception as exc:
                    json_prompt_result["error_message"] = str(exc)
                json_prompt_result["processing_seconds"] = round(time.perf_counter() - json_prompt_started, 6)
                result_payload["json_prompt_result"] = json_prompt_result

            if plate_detector is not None:
                plate_result = run_plate_detector_on_crop(
                    crop_path=crop_path,
                    plate_detector=plate_detector,
                    plate_audit_dir=plate_audit_dir,
                    detection_id=result_payload["detection_id"],
                )
                plate_crop_count += int(plate_result["plate_crop_count"])
                plate_result["plate_ocr_results"] = []
                for plate_crop_relative_path in plate_result["plate_crop_paths"]:
                    plate_crop_path = (run_dir / Path(plate_crop_relative_path)).resolve()
                    plate_ocr_task = run_florence_generation(
                        image_path=plate_crop_path,
                        processor=processor,
                        model=model,
                        device_used=device_used,
                        task_prompt="<OCR>",
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                    plate_result["plate_ocr_results"].append(plate_ocr_task)
                    if plate_ocr_task["status"] == "success":
                        plate_ocr_success_count += 1
                result_payload["plate_detector_result"] = plate_result

            best_ocr_text, best_ocr_source = best_ocr_from_results(
                result_payload["native_task_results"],
                result_payload["json_prompt_result"],
            )
            best_color_text, best_color_source = best_color_from_results(
                result_payload["native_task_results"],
                result_payload["json_prompt_result"],
            )
            result_payload["parsed_ocr_text"] = best_ocr_text
            result_payload["parsed_vehicle_color"] = best_color_text
            result_payload["best_available_ocr_source"] = best_ocr_source
            result_payload["best_available_color_source"] = best_color_source

            result_payload["processing_seconds"] = round(time.perf_counter() - crop_started_at, 6)
            if any(item["status"] == "success" for item in result_payload["native_task_results"]):
                result_payload["status"] = "success"
                successful_outputs += 1
                processing_seconds_values.append(result_payload["processing_seconds"])
            else:
                failed_outputs += 1
                result_payload["error_message"] = "All native Florence tasks returned empty outputs."
        except Exception as exc:
            failed_outputs += 1
            result_payload["processing_seconds"] = round(time.perf_counter() - crop_started_at, 6)
            result_payload["error_message"] = str(exc)

        results.append(result_payload)

    avg_seconds_per_crop = round(sum(processing_seconds_values) / len(processing_seconds_values), 6) if processing_seconds_values else 0.0
    recommendation = (
        "Florence native tasks produced useful output. Proceed to tracking-aware OCR/color later."
        if (native_task_success_count > 0 and not base_model_files["missing_files"])
        else "Fix Florence prompt/task configuration or base model files before full OCR/color."
    )
    ready_for_full_ocr_color_after_tracking = bool(
        base_model_files["path_exists"]
        and not base_model_files["missing_files"]
        and native_task_success_count > 0
    )

    audit_summary = {
        "status": "success",
        "run_dir": str(run_dir),
        "florence_model_path": str(florence_model_path),
        "florence_adapter_path": str(florence_adapter_path) if florence_adapter_path is not None else None,
        "model_path_exists": base_model_files["path_exists"],
        "base_model_required_files": base_model_files["found_files"],
        "base_model_missing_files": base_model_files["missing_files"],
        "adapter_path_exists": adapter_path_exists,
        "adapter_files_summary": adapter_files_summary,
        "model_load_status": "success",
        "adapter_load_status": adapter_load_status,
        "device_used": device_used,
        "task_mode": task_mode,
        "audit_crop_limit": audit_limit,
        "selected_crop_count": len(selected_crops),
        "successful_outputs": successful_outputs,
        "failed_outputs": failed_outputs,
        "missing_crop_count": missing_crop_count,
        "avg_seconds_per_crop": avg_seconds_per_crop,
        "native_tasks": native_tasks,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "run_json_prompt_test": run_json_prompt_test,
        "native_task_success_count": native_task_success_count,
        "ocr_native_success_count": ocr_native_success_count,
        "caption_success_count": caption_success_count,
        "detailed_caption_success_count": detailed_caption_success_count,
        "json_prompt_success_count": json_prompt_success_count,
        "plate_detector_load_status": plate_detector_load_status,
        "plate_crop_count": plate_crop_count,
        "plate_ocr_success_count": plate_ocr_success_count,
        "ready_for_full_ocr_color_after_tracking": ready_for_full_ocr_color_after_tracking,
        "recommendation": recommendation,
        "error_message": None,
    }

    audit_results = {
        "status": "success",
        "input_yolo_detections_file": "03_yolo_detections.json",
        "selected_crop_count": len(selected_crops),
        "results": results,
    }

    write_json(run_dir / "04A_florence_model_audit.json", audit_summary)
    write_json(run_dir / "04A_florence_audit_results.json", audit_results)
    return audit_summary, audit_results
