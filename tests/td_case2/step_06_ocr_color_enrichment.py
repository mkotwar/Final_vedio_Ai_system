from __future__ import annotations

import ast
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from device_manager import cuda_memory_allocated_mb, cuda_memory_reserved_mb
from stage_checks import read_json, write_json
from step_04a_florence_model_audit import (
    ADAPTER_EXPECTED_FILES,
    BASE_MODEL_REQUIRED_FILES,
    inspect_path_files,
    load_florence_model,
    run_florence_generation,
)
from vehicle_color import extract_florence_vehicle_color, resolve_vehicle_color


VEHICLE_MAKES = (
    "audi",
    "bmw",
    "chevrolet",
    "ford",
    "honda",
    "hyundai",
    "kia",
    "mahindra",
    "maruti",
    "mercedes",
    "nissan",
    "renault",
    "skoda",
    "suzuki",
    "tata",
    "toyota",
    "volkswagen",
    "volvo",
)
VEHICLE_MODELS = (
    "alto",
    "baleno",
    "bolero",
    "brezza",
    "creta",
    "dzire",
    "ecosport",
    "ertiga",
    "fortuner",
    "innova",
    "nexon",
    "scorpio",
    "swift",
    "thar",
    "verna",
    "wagonr",
    "xuv",
)
BODY_TYPES = (
    "hatchback",
    "sedan",
    "suv",
    "pickup truck",
    "truck",
    "bus",
    "minibus",
    "van",
    "motorcycle",
    "scooter",
    "bicycle",
    "auto rickshaw",
)
SCENE_ROAD_TYPES = ("highway", "motorway", "residential road", "city street", "service road", "dirt road")
CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SUPPORTED_PROCESS_GROUPS = {"primary", "fallback"}
INVALID_PLATE_WORDS = {
    "UNANSWERABLE",
    "ANSWERABLE",
    "UNKNOWN",
    "NONE",
    "NULL",
    "NIL",
    "STOP",
    "VICTORY",
    "DEV",
    "CITY",
    "RAIL",
    "AIR",
    "AMBULANCE",
    "LANCERAIL",
    "AIRAMBULANCE",
    "LANCERAILAIRAMBULANCE",
    "TOY",
    "VEHICLE",
    "CAR",
    "TRUCK",
    "BUS",
    "MOTORCYCLE",
}
INVALID_PLATE_PHRASES = {
    "LANCERAILAIRAMBULANCE",
    "UNANSWERABLE",
    "STOP",
}
INDIAN_STATE_UT_CODES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}
INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"),
]
STRICT_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")


def _resolve_run_relative(run_dir: Path, path_value: str) -> Path | None:
    """Resolve a run-relative path safely."""

    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    """Convert an absolute path back to a run-relative POSIX path when possible."""

    if path is None:
        return None
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def _stringify_generation_result(result: dict[str, Any] | None) -> str:
    """Flatten Florence generation output into a simple string."""

    if not result:
        return ""
    raw_text = str(result.get("raw_decoded_text", "") or "").strip()
    post_processed_output = result.get("post_processed_output")
    if isinstance(post_processed_output, str):
        return f"{raw_text} {post_processed_output}".strip()
    if isinstance(post_processed_output, dict):
        return f"{raw_text} {post_processed_output}".strip()
    if isinstance(post_processed_output, list):
        return f"{raw_text} {post_processed_output}".strip()
    return raw_text


def clean_florence_ocr_text(raw_text: str) -> str:
    """Extract and normalize OCR text from Florence mixed plain/dict-like output."""

    raw_value = str(raw_text or "").strip()
    if not raw_value:
        return ""

    parts: list[str] = []
    leading_text = raw_value.split("{", 1)[0].strip()
    if leading_text:
        parts.append(leading_text)

    dict_match = re.search(r"\{.*\}", raw_value, flags=re.DOTALL)
    if dict_match:
        try:
            parsed = ast.literal_eval(dict_match.group(0))
            if isinstance(parsed, dict):
                for value in parsed.values():
                    value_text = str(value or "").strip()
                    if value_text:
                        parts.append(value_text)
        except Exception:
            parts.append(dict_match.group(0))

    if not parts:
        parts.append(raw_value)

    combined = " ".join(parts).upper()
    combined = re.sub(r"<\s*OCR\s*>", " ", combined)
    combined = re.sub(r"<\s*CAPTION\s*>", " ", combined)
    combined = re.sub(r"<\s*DETAILED_CAPTION\s*>", " ", combined)
    combined = re.sub(r"\bDETAILED_CAPTION\b", " ", combined)
    combined = re.sub(r"\bCAPTION\b", " ", combined)
    combined = re.sub(r"[{}\[\]:'\",]", " ", combined)
    combined = re.sub(r"[^A-Z0-9]+", " ", combined)
    tokens = [token for token in combined.split() if token]

    deduped_tokens: list[str] = []
    for token in tokens:
        if not deduped_tokens or deduped_tokens[-1] != token:
            deduped_tokens.append(token)

    return "".join(deduped_tokens)


def remove_repeated_plate_artifacts(text: str) -> str:
    """Remove repeated OCR artifacts such as ABC123OCRABC123."""

    normalized = str(text or "").upper()
    if not normalized:
        return ""

    normalized = normalized.replace(" ", "")
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    if "OCR" in normalized:
        candidates = [part for part in normalized.split("OCR") if part]
        if candidates:
            unique_candidates: list[str] = []
            for candidate in candidates:
                if candidate not in unique_candidates:
                    unique_candidates.append(candidate)
            candidates = unique_candidates
            candidates.sort(
                key=lambda item: (
                    any(pattern.match(item) for pattern in INDIAN_PLATE_PATTERNS),
                    any(char.isdigit() for char in item),
                    any(char.isalpha() for char in item),
                    len(item),
                ),
                reverse=True,
            )
            normalized = candidates[0]

    half_length = len(normalized) // 2
    if half_length > 0 and len(normalized) % 2 == 0 and normalized[:half_length] == normalized[half_length:]:
        normalized = normalized[:half_length]

    return normalized


def is_valid_license_plate_candidate(text: str) -> tuple[bool, str]:
    """Validate a cleaned OCR candidate against conservative plate rules."""

    candidate = str(text or "").upper()
    if not candidate:
        return False, "empty"
    if candidate in INVALID_PLATE_WORDS or any(phrase in candidate for phrase in INVALID_PLATE_PHRASES):
        return False, "invalid_word"
    if not re.fullmatch(r"[A-Z0-9]+", candidate):
        return False, "invalid_format"
    if len(candidate) < 6:
        return False, "too_short"
    if len(candidate) > 12:
        return False, "too_long"
    if not any(char.isdigit() for char in candidate):
        return False, "no_digit"
    if not any(char.isalpha() for char in candidate):
        return False, "no_letter"
    if len(set(candidate)) <= 2 and len(candidate) >= 6:
        return False, "too_repetitive"
    if re.fullmatch(r"([A-Z0-9])\1{5,}", candidate):
        return False, "too_repetitive"
    for pattern in INDIAN_PLATE_PATTERNS:
        if pattern.match(candidate) and candidate[:2] in INDIAN_STATE_UT_CODES:
            return True, "high"
    return True, "medium"


def _clean_and_validate_plate_candidate(raw_text: str) -> dict[str, Any]:
    """Clean one OCR source and return validation metadata."""

    cleaned = clean_florence_ocr_text(raw_text)
    artifact_cleaned = remove_repeated_plate_artifacts(cleaned)
    is_valid, reason = is_valid_license_plate_candidate(artifact_cleaned)
    return {
        "raw_text": str(raw_text or ""),
        "cleaned_text": artifact_cleaned,
        "is_valid": is_valid,
        "plate_format_confidence": reason if is_valid else "none",
        "reject_reason": None if is_valid else reason,
    }


def is_verified_license_plate(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a candidate is a strict verified license plate."""

    text = str(candidate.get("text", "") or "").upper()
    source = str(candidate.get("source", "none") or "none")
    confidence_level = str(candidate.get("plate_format_confidence", "none") or "none")
    if not text:
        return False, "empty"
    if text in INVALID_PLATE_WORDS or any(phrase in text for phrase in INVALID_PLATE_PHRASES):
        return False, "invalid_word"
    if source != "plate_crop_ocr":
        return False, "vehicle_crop_ocr_not_verified"
    if confidence_level != "high":
        return False, "medium_confidence_not_verified"
    if not STRICT_INDIAN_PLATE_PATTERN.match(text):
        return False, "strict_regex_failed"
    if text[:2] not in INDIAN_STATE_UT_CODES:
        return False, "unknown_region_code"
    if len(text) < 8:
        return False, "too_short"
    if len(text) > 10:
        return False, "too_long"
    return True, "strict_plate_crop_high_confidence"


def classify_plate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify a plate candidate as verified, possible, or invalid."""

    text = str(candidate.get("text", "") or "").upper()
    source = str(candidate.get("source", "none") or "none")
    if not text:
        return {
            "text": text,
            "source": source,
            "classification": "invalid",
            "reason": "empty",
        }

    is_valid, valid_reason = is_valid_license_plate_candidate(text)
    if not is_valid:
        return {
            "text": text,
            "source": source,
            "classification": "invalid",
            "reason": valid_reason,
        }

    is_verified, verified_reason = is_verified_license_plate(
        {
            "text": text,
            "source": source,
            "plate_format_confidence": candidate.get("plate_format_confidence", "none"),
        }
    )
    if is_verified:
        return {
            "text": text,
            "source": source,
            "classification": "verified",
            "reason": verified_reason,
        }

    return {
        "text": text,
        "source": source,
        "classification": "possible",
        "reason": verified_reason if verified_reason != "empty" else "possible_candidate",
    }


def _parse_color_from_text(text: str) -> str:
    """Return a canonical vehicle-linked color from a free Florence caption."""

    _raw_phrase, canonical = extract_florence_vehicle_color(text)
    return canonical or "unknown"


def _first_explicit_term(text: str, terms: tuple[str, ...]) -> str | None:
    """Return the first explicitly present vocabulary term, preferring longer terms."""

    normalized = str(text or "").lower()
    for term in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            return term
    return None


def _explicit_flag(text: str, terms: tuple[str, ...]) -> bool | None:
    """Return True only when a caption explicitly states an attribute."""

    normalized = str(text or "").lower()
    return True if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms) else None


def extract_structured_florence_metadata(
    *,
    caption_text: str,
    ocr_text: str,
    plate_found: bool,
    plate_confidence: float,
    plate_text: str,
    plate_valid: bool,
    resolved_color: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract conservative structured metadata from Florence OCR/CAPTION outputs."""

    caption = str(caption_text or "").strip()
    normalized = caption.lower()
    color = str(dict(resolved_color or {}).get("color") or _parse_color_from_text(caption))
    body_type = _first_explicit_term(caption, BODY_TYPES)
    vehicle_make = _first_explicit_term(caption, VEHICLE_MAKES)
    vehicle_model = _first_explicit_term(caption, VEHICLE_MODELS)
    vehicle_category = _first_explicit_term(
        caption,
        ("emergency vehicle", "commercial vehicle", "passenger vehicle", "public transport", "delivery vehicle"),
    )
    approximate_size = _first_explicit_term(caption, ("very large", "large", "mid-size", "compact", "small"))
    visible_damage = _first_explicit_term(caption, ("heavily damaged", "damaged", "dented", "cracked", "broken"))
    doors_match = re.search(r"\b(two|three|four|five|2|3|4|5)[ -]door\b", normalized)
    doors = doors_match.group(1) if doors_match else None
    markings = _first_explicit_term(
        caption,
        ("police", "ambulance", "fire service", "emergency", "taxi", "school bus"),
    )
    commercial_branding = _first_explicit_term(caption, ("delivery", "company branding", "advertising", "commercial"))
    orientation = _first_explicit_term(
        caption,
        ("front-left", "front-right", "rear-left", "rear-right", "facing left", "facing right", "facing forward", "facing away"),
    )
    view = _first_explicit_term(caption, ("front view", "rear view", "side view", "three-quarter view"))
    cargo = _first_explicit_term(caption, ("shipping container", "construction material", "luggage", "boxes", "cargo", "goods"))
    scene_road_type = _first_explicit_term(caption, SCENE_ROAD_TYPES)
    scene_place = _first_explicit_term(caption, ("parking lot", "car park", "intersection", "roundabout", "toll plaza"))
    weather = _first_explicit_term(caption, ("heavy rain", "rainy", "foggy", "snowy", "overcast", "sunny"))
    lighting = _first_explicit_term(caption, ("night", "daytime", "dusk", "dawn"))
    environment = _first_explicit_term(caption, ("indoors", "indoor", "outdoors", "outdoor"))

    region_hint = None
    if plate_valid and STRICT_INDIAN_PLATE_PATTERN.fullmatch(plate_text or ""):
        region_hint = f"India state/territory code {plate_text[:2]}"
    plate_visibility = "readable" if plate_valid else "detected_unreadable" if plate_found else "not_detected"
    plate_style = _first_explicit_term(caption, ("white license plate", "yellow license plate", "green license plate", "red license plate"))

    vehicle_attributes = {
        "color": None if color == "unknown" else color,
        "color_raw_prediction": dict(resolved_color or {}).get("raw_prediction"),
        "color_source": dict(resolved_color or {}).get("source", "florence_caption_normalized"),
        "color_image_confidence": dict(resolved_color or {}).get("image_confidence"),
        "make": vehicle_make,
        "model": vehicle_model,
        "body_type": body_type,
        "vehicle_category": vehicle_category,
        "approximate_size": approximate_size,
        "visible_damage": visible_damage,
        "doors": doors,
        "special_markings": markings,
        "commercial_branding": commercial_branding,
        "company_logo": vehicle_make if vehicle_make and "logo" in normalized else None,
        "roof_rack": _explicit_flag(caption, ("roof rack", "roof carrier")),
        "trailer": _explicit_flag(caption, ("trailer", "towing")),
        "cargo": cargo,
        "orientation": orientation,
        "view": view,
        "source": "florence_caption",
        "confidence": "medium" if any((color != "unknown", body_type, vehicle_make, vehicle_model)) else "none",
    }
    license_plate_attributes = {
        "country_region_hint": region_hint,
        "plate_style": plate_style,
        "visibility": plate_visibility,
        "detector_confidence": round(float(plate_confidence or 0.0), 6),
        "ocr_confidence": "high" if plate_valid else "low" if ocr_text else "none",
        "source": "plate_detector_and_florence_ocr" if plate_found else "florence_ocr",
    }
    scene_attributes = {
        "road_type": scene_road_type,
        "place": scene_place,
        "weather": weather,
        "lighting": lighting,
        "environment": environment,
        "source": "florence_caption",
        "confidence": "medium" if any((scene_road_type, scene_place, weather, lighting, environment)) else "none",
    }
    return {
        "vehicle_attributes": vehicle_attributes,
        "license_plate_attributes": license_plate_attributes,
        "scene_attributes": scene_attributes,
    }


def _prepare_selected_crops(
    selection_payload: dict[str, Any],
    process_groups: set[str],
    primary_limit: int,
    fallback_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten Step 05 selected detections into a processing queue."""

    queue: list[dict[str, Any]] = []
    counts = {"primary": 0, "fallback": 0}
    group_limits = {"primary": primary_limit, "fallback": fallback_limit}

    for track in list(selection_payload.get("tracks", [])):
        track_level = {
            "track_id": str(track.get("track_id", "")),
            "selection_group": str(track.get("selection_group", "")),
            "quality_label": str(track.get("quality_label", "")),
            "dominant_class_name": str(track.get("dominant_class_name", "")),
            "track_quality": str(track.get("track_quality", "")),
        }
        selection_group = track_level["selection_group"]
        if selection_group not in process_groups:
            continue

        for crop_item in list(track.get("selected_detections", [])):
            if group_limits[selection_group] > 0 and counts[selection_group] >= group_limits[selection_group]:
                break
            queue.append({**track_level, **crop_item})
            counts[selection_group] += 1

    queue.sort(key=lambda item: (item.get("selection_group") != "primary", item.get("track_id", ""), item.get("rank", 0)))
    return queue, counts


def _load_plate_detector(model_path: Path | None) -> tuple[str, Any | None]:
    """Load an optional local license plate detector."""

    if model_path is None:
        return "not_provided", None
    if not model_path.exists():
        return "missing", None
    try:
        from ultralytics import YOLO  # type: ignore

        return "success", YOLO(str(model_path))
    except Exception:
        return "failed", None


def _maybe_apply_adapter(model: Any, adapter_path: Path | None) -> tuple[Any, str]:
    """Apply a local PEFT adapter if available."""

    if adapter_path is None:
        return model, "not_provided"
    if not adapter_path.exists():
        return model, "missing"
    try:
        from peft import PeftModel  # type: ignore

        return PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True), "success"
    except Exception:
        return model, "failed"


def _select_best_plate_box(result: Any, min_plate_confidence: float) -> tuple[bool, float, list[float] | None]:
    """Choose the highest-confidence plate box from a YOLO result."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return False, 0.0, None

    xyxy_values = boxes.xyxy.tolist()
    confidence_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy_values)
    best_confidence = 0.0
    best_bbox = None
    for bbox_xyxy, confidence in zip(xyxy_values, confidence_values):
        confidence_value = float(confidence)
        if confidence_value < min_plate_confidence:
            continue
        if confidence_value > best_confidence:
            best_confidence = confidence_value
            best_bbox = [float(value) for value in bbox_xyxy]
    return bool(best_bbox), best_confidence, best_bbox


def _run_plate_detection(
    *,
    run_dir: Path,
    crop_item: dict[str, Any],
    plate_detector: Any | None,
    plate_dir: Path,
    debug_dir: Path,
    save_plate_crops: bool,
    save_debug_images: bool,
    min_plate_confidence: float,
    device: str,
) -> dict[str, Any]:
    """Run optional plate detection on one selected vehicle crop."""

    if plate_detector is None:
        return {
            "plate_detector_status": "skipped",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "debug_image_path": None,
        }

    crop_path = _resolve_run_relative(run_dir, str(crop_item.get("selected_crop_path", "") or ""))
    if crop_path is None or not crop_path.exists():
        return {
            "plate_detector_status": "failed",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "debug_image_path": None,
        }

    image = cv2.imread(str(crop_path))
    if image is None:
        return {
            "plate_detector_status": "failed",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "debug_image_path": None,
        }

    results = plate_detector.predict(
        source=str(crop_path),
        conf=min_plate_confidence,
        iou=0.50,
        imgsz=960,
        device=device,
        verbose=False,
    )
    if not results:
        return {
            "plate_detector_status": "success",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "debug_image_path": None,
        }

    plate_found, plate_confidence, plate_bbox_xyxy = _select_best_plate_box(results[0], min_plate_confidence)
    if not plate_found or plate_bbox_xyxy is None:
        return {
            "plate_detector_status": "success",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "debug_image_path": None,
        }

    raw_x1, raw_y1, raw_x2, raw_y2 = [int(round(value)) for value in plate_bbox_xyxy]
    box_width = max(1, raw_x2 - raw_x1)
    box_height = max(1, raw_y2 - raw_y1)
    pad_x = max(3, int(round(box_width * 0.10)))
    pad_y = max(3, int(round(box_height * 0.20)))
    x1 = max(0, raw_x1 - pad_x)
    y1 = max(0, raw_y1 - pad_y)
    x2 = min(image.shape[1], raw_x2 + pad_x)
    y2 = min(image.shape[0], raw_y2 + pad_y)
    plate_crop = image[y1:y2, x1:x2]
    plate_crop_path: Path | None = None
    plate_enhanced_crop_path: Path | None = None
    debug_image_path: Path | None = None

    if save_plate_crops and plate_crop.size > 0:
        output_name = f"{crop_item['track_id']}_{crop_item['detection_id']}_plate.jpg"
        plate_crop_path = plate_dir / output_name
        cv2.imwrite(str(plate_crop_path), plate_crop)
        target_width = max(320, int(plate_crop.shape[1]))
        scale = target_width / max(1, plate_crop.shape[1])
        resized = cv2.resize(
            plate_crop,
            (target_width, max(64, int(round(plate_crop.shape[0] * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        enhanced = cv2.cvtColor(cv2.merge((enhanced_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
        enhanced_name = f"{crop_item['track_id']}_{crop_item['detection_id']}_plate_enhanced.jpg"
        plate_enhanced_crop_path = plate_dir / enhanced_name
        cv2.imwrite(str(plate_enhanced_crop_path), enhanced)

    if save_debug_images:
        debug_output_name = f"{crop_item['track_id']}_{crop_item['detection_id']}_debug.jpg"
        debug_image_path = debug_dir / debug_output_name
        debug_image = image.copy()
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            f"plate {plate_confidence:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_image_path), debug_image)

    return {
        "plate_detector_status": "success",
        "plate_crop_found": True,
        "plate_crop_path": _relative_to_run(run_dir, plate_crop_path),
        "plate_crop_paths": [
            path_value
            for path_value in (
                _relative_to_run(run_dir, plate_crop_path),
                _relative_to_run(run_dir, plate_enhanced_crop_path),
            )
            if path_value
        ],
        "plate_enhanced_crop_path": _relative_to_run(run_dir, plate_enhanced_crop_path),
        "plate_confidence": round(plate_confidence, 6),
        "plate_bbox_xyxy": [round(value, 3) for value in plate_bbox_xyxy],
        "plate_crop_bbox_xyxy": [x1, y1, x2, y2],
        "plate_crop_padding_pixels": {"x": pad_x, "y": pad_y},
        "debug_image_path": _relative_to_run(run_dir, debug_image_path),
    }


def _select_best_plate_candidate(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Select the best OCR-derived license plate candidate for one crop."""

    source_candidates = [
        ("plate_crop_ocr", str(result_payload.get("plate_crop_ocr_raw", "") or "")),
        ("vehicle_crop_ocr", str(result_payload.get("vehicle_crop_ocr_raw", "") or "")),
    ]

    best_candidate: dict[str, Any] | None = None
    rejected_candidates: list[dict[str, Any]] = []
    possible_candidates: list[dict[str, Any]] = []
    weak_ocr_text: list[str] = []
    verified_license_plate = "not_visible"
    verified_license_plate_valid = False
    verified_license_plate_reason = "none"

    for source_name, raw_text in source_candidates:
        cleaned_payload = _clean_and_validate_plate_candidate(raw_text)
        candidate_payload = {
            "source": source_name,
            **cleaned_payload,
        }
        if source_name == "plate_crop_ocr":
            result_payload["plate_crop_ocr_cleaned"] = cleaned_payload["cleaned_text"]
        else:
            result_payload["vehicle_crop_ocr_cleaned"] = cleaned_payload["cleaned_text"]

        if cleaned_payload["is_valid"]:
            classification = classify_plate_candidate(
                {
                    "text": cleaned_payload["cleaned_text"],
                    "source": source_name,
                    "plate_format_confidence": cleaned_payload["plate_format_confidence"],
                }
            )
            ranking = (
                0 if source_name == "plate_crop_ocr" else 1,
                0 if cleaned_payload["plate_format_confidence"] == "high" else 1,
                -len(cleaned_payload["cleaned_text"]),
            )
            if best_candidate is None or ranking < best_candidate["ranking"]:
                best_candidate = {
                    "ranking": ranking,
                    "source": source_name,
                    "cleaned_text": cleaned_payload["cleaned_text"],
                    "plate_format_confidence": cleaned_payload["plate_format_confidence"],
                }
            if classification["classification"] == "verified":
                verified_license_plate = cleaned_payload["cleaned_text"]
                verified_license_plate_valid = True
                verified_license_plate_reason = classification["reason"]
            elif classification["classification"] == "possible":
                possible_candidates.append(
                    {
                        "text": cleaned_payload["cleaned_text"],
                        "source": source_name,
                        "reason": classification["reason"],
                        "plate_format_confidence": cleaned_payload["plate_format_confidence"],
                    }
                )
                weak_ocr_text.append(cleaned_payload["cleaned_text"])
        elif cleaned_payload["cleaned_text"]:
            rejected_candidates.append(candidate_payload)
            weak_ocr_text.append(cleaned_payload["cleaned_text"])

    if best_candidate is not None:
        return {
            "parsed_license_plate_text": best_candidate["cleaned_text"],
            "parsed_license_plate_valid": True,
            "parsed_license_plate_source": best_candidate["source"],
            "plate_format_confidence": best_candidate["plate_format_confidence"],
            "license_plate_reject_reason": None,
            "rejected_plate_candidates": rejected_candidates,
            "verified_license_plate": verified_license_plate,
            "verified_license_plate_valid": verified_license_plate_valid,
            "verified_license_plate_reason": verified_license_plate_reason,
            "possible_license_plate_candidates": possible_candidates,
            "weak_ocr_text": weak_ocr_text,
        }

    reject_reason = rejected_candidates[0]["reject_reason"] if rejected_candidates else "not_visible"
    return {
        "parsed_license_plate_text": "not_visible",
        "parsed_license_plate_valid": False,
        "parsed_license_plate_source": "none",
        "plate_format_confidence": "none",
        "license_plate_reject_reason": reject_reason,
        "rejected_plate_candidates": rejected_candidates,
        "verified_license_plate": "not_visible",
        "verified_license_plate_valid": False,
        "verified_license_plate_reason": reject_reason,
        "possible_license_plate_candidates": possible_candidates,
        "weak_ocr_text": weak_ocr_text,
    }


def _select_best_plate_ocr_variant(variant_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the OCR variant most consistent with a plausible plate format."""

    if not variant_results:
        return None

    def ranking(item: dict[str, Any]) -> tuple[int, int, int, int]:
        cleaned = _clean_and_validate_plate_candidate(str(item.get("ocr_raw", "") or ""))
        confidence = str(cleaned.get("plate_format_confidence", "none"))
        text = str(cleaned.get("cleaned_text", ""))
        return (
            0 if bool(cleaned.get("is_valid")) else 1,
            0 if confidence == "high" else 1,
            0 if str(item.get("variant", "")) == "padded_original" else 1,
            abs(len(text) - 9),
        )

    return sorted(variant_results, key=ranking)[0]


def _aggregate_attribute_group(track_items: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    """Merge crop-level structured attributes without inventing missing values."""

    merged: dict[str, Any] = {}
    ranked_items = sorted(
        track_items,
        key=lambda item: (
            CONFIDENCE_RANK.get(str(dict(item.get(group_name, {})).get("confidence", "none")), 0),
            float(item.get("final_selection_score", 0.0) or 0.0),
            float(item.get("plate_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for item in ranked_items:
        attributes = dict(item.get(group_name, {}))
        for key, value in attributes.items():
            if key not in merged and value not in (None, "", [], {}, "unknown", "none"):
                merged[key] = value
    return merged


def _aggregate_track_results(track_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate crop-level OCR/color outputs into one track-level summary."""

    first_item = track_items[0]
    clear_color_counter: Counter[str] = Counter()
    color_source_by_color: dict[str, str] = {}
    valid_plate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_plate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_plate_candidates: list[dict[str, Any]] = []
    possible_license_plate_candidates: list[dict[str, Any]] = []
    weak_ocr_text: list[str] = []
    invalid_ocr_text: list[str] = []

    successful_crop_results = 0
    raw_plate_text_found = False
    for item in track_items:
        if item.get("status") == "success":
            successful_crop_results += 1

        if str(item.get("vehicle_crop_ocr_cleaned", "")) or str(item.get("plate_crop_ocr_cleaned", "")):
            raw_plate_text_found = True

        if bool(item.get("parsed_license_plate_valid")):
            valid_plate_groups[str(item.get("parsed_license_plate_text", ""))].append(item)
        if bool(item.get("verified_license_plate_valid")):
            verified_plate_groups[str(item.get("verified_license_plate", ""))].append(item)
        else:
            for candidate in list(item.get("rejected_plate_candidates", [])):
                rejected_plate_candidates.append(
                    {
                        "text": str(candidate.get("cleaned_text", "")),
                        "source": str(candidate.get("source", "none")),
                        "reject_reason": str(candidate.get("reject_reason", "unknown")),
                        "selection_group": str(item.get("selection_group", "")),
                        "crop_path": str(item.get("selected_crop_path", "")),
                    }
                )
                if str(candidate.get("cleaned_text", "")):
                    invalid_ocr_text.append(str(candidate.get("cleaned_text", "")))
        for candidate in list(item.get("possible_license_plate_candidates", [])):
            possible_license_plate_candidates.append(candidate)
            if str(candidate.get("text", "")):
                weak_ocr_text.append(str(candidate.get("text", "")))
        for text in list(item.get("weak_ocr_text", [])):
            if text:
                weak_ocr_text.append(str(text))

        parsed_color = str(item.get("parsed_vehicle_color", "unknown"))
        if parsed_color != "unknown":
            weight = 2 if str(item.get("selection_group", "")) == "primary" else 1
            clear_color_counter[parsed_color] += weight
            color_source_by_color.setdefault(parsed_color, str(item.get("parsed_color_source", "caption")))

    best_plate_text = "not_visible"
    best_plate_valid = False
    best_plate_source = "none"
    best_plate_confidence_level = "none"
    best_plate_evidence_count = 0
    best_ocr_crop_path = None
    all_valid_candidate_plates: list[dict[str, Any]] = []
    verified_license_plate = "not_visible"
    verified_license_plate_valid = False
    verified_license_plate_source = "none"
    verified_license_plate_confidence_level = "none"
    verified_license_plate_evidence_count = 0
    verified_license_plate_crop_path = None
    best_license_plate_promoted_to_verified = False
    best_license_plate_verification_reject_reason = None

    if valid_plate_groups:
        ranked_candidates: list[tuple[tuple[int, int, int, float, float], str, list[dict[str, Any]]]] = []
        for plate_text, items in valid_plate_groups.items():
            evidence_count = len(items)
            best_item = sorted(
                items,
                key=lambda item: (
                    0 if str(item.get("selection_group", "")) == "primary" else 1,
                    0 if str(item.get("parsed_license_plate_source", "")) == "plate_crop_ocr" else 1,
                    0 if str(item.get("plate_format_confidence", "")) == "high" else 1,
                    -float(item.get("plate_confidence", 0.0) or 0.0),
                    -float(item.get("final_selection_score", 0.0) or 0.0),
                ),
            )[0]
            ranking = (
                0 if str(best_item.get("selection_group", "")) == "primary" else 1,
                0 if str(best_item.get("parsed_license_plate_source", "")) == "plate_crop_ocr" else 1,
                0 if str(best_item.get("plate_format_confidence", "")) == "high" else 1,
                -float(evidence_count),
                -float(best_item.get("plate_confidence", 0.0) or 0.0),
            )
            ranked_candidates.append((ranking, plate_text, items))
            all_valid_candidate_plates.append(
                {
                    "plate_text": plate_text,
                    "evidence_count": evidence_count,
                    "best_source": str(best_item.get("parsed_license_plate_source", "none")),
                    "best_selection_group": str(best_item.get("selection_group", "")),
                    "best_plate_format_confidence": str(best_item.get("plate_format_confidence", "none")),
                }
            )

        ranked_candidates.sort(key=lambda item: item[0])
        _ranking, best_plate_text, best_items = ranked_candidates[0]
        best_item = sorted(
            best_items,
            key=lambda item: (
                0 if str(item.get("selection_group", "")) == "primary" else 1,
                0 if str(item.get("parsed_license_plate_source", "")) == "plate_crop_ocr" else 1,
                0 if str(item.get("plate_format_confidence", "")) == "high" else 1,
                -float(item.get("plate_confidence", 0.0) or 0.0),
                -float(item.get("final_selection_score", 0.0) or 0.0),
            ),
        )[0]
        best_plate_valid = True
        best_plate_source = str(best_item.get("parsed_license_plate_source", "none"))
        best_plate_confidence_level = str(best_item.get("plate_format_confidence", "none"))
        best_plate_evidence_count = len(best_items)
        best_ocr_crop_path = str(best_item.get("selected_crop_path", ""))

    if verified_plate_groups:
        ranked_verified: list[tuple[tuple[int, float, float, int], str, list[dict[str, Any]]]] = []
        for plate_text, items in verified_plate_groups.items():
            evidence_count = len(items)
            best_item = sorted(
                items,
                key=lambda item: (
                    -float(item.get("plate_confidence", 0.0) or 0.0),
                    -float(item.get("final_selection_score", 0.0) or 0.0),
                    0 if str(item.get("selection_group", "")) == "primary" else 1,
                ),
            )[0]
            ranked_verified.append(
                (
                    (
                        -evidence_count,
                        -float(best_item.get("plate_confidence", 0.0) or 0.0),
                        -float(best_item.get("final_selection_score", 0.0) or 0.0),
                        0 if str(best_item.get("selection_group", "")) == "primary" else 1,
                    ),
                    plate_text,
                    items,
                )
            )
        ranked_verified.sort(key=lambda item: item[0])
        _rank, verified_license_plate, verified_items = ranked_verified[0]
        verified_best_item = sorted(
            verified_items,
            key=lambda item: (
                -float(item.get("plate_confidence", 0.0) or 0.0),
                -float(item.get("final_selection_score", 0.0) or 0.0),
                0 if str(item.get("selection_group", "")) == "primary" else 1,
            ),
        )[0]
        verified_license_plate_valid = True
        verified_license_plate_source = str(verified_best_item.get("parsed_license_plate_source", "none"))
        verified_license_plate_confidence_level = str(verified_best_item.get("plate_format_confidence", "none"))
        verified_license_plate_evidence_count = len(verified_items)
        verified_license_plate_crop_path = str(verified_best_item.get("selected_crop_path", ""))

    if best_plate_valid and verified_license_plate_valid and best_plate_text == verified_license_plate:
        best_license_plate_promoted_to_verified = True
    elif best_plate_valid:
        best_license_plate_promoted_to_verified = False
        best_license_plate_verification_reject_reason = "not_strictly_verified"

    best_vehicle_color = "unknown"
    best_color_source = "unknown"
    best_color_crop_path = None
    if clear_color_counter:
        best_vehicle_color = clear_color_counter.most_common(1)[0][0]
        best_color_source = color_source_by_color.get(best_vehicle_color, "caption")
        for item in track_items:
            if str(item.get("parsed_vehicle_color", "unknown")) == best_vehicle_color:
                best_color_crop_path = str(item.get("selected_crop_path", ""))
                break

    return {
        "track_id": first_item["track_id"],
        "selection_group": first_item["selection_group"],
        "quality_label": first_item["quality_label"],
        "dominant_class_name": first_item["dominant_class_name"],
        "selected_crop_count": len(track_items),
        "successful_crop_results": successful_crop_results,
        "has_raw_plate_text": raw_plate_text_found,
        "best_license_plate_text": best_plate_text,
        "best_license_plate_valid": best_plate_valid,
        "best_license_plate_source": best_plate_source,
        "best_license_plate_confidence_level": best_plate_confidence_level,
        "best_license_plate_evidence_count": best_plate_evidence_count,
        "verified_license_plate": verified_license_plate,
        "verified_license_plate_valid": verified_license_plate_valid,
        "verified_license_plate_source": verified_license_plate_source,
        "verified_license_plate_confidence_level": verified_license_plate_confidence_level,
        "verified_license_plate_evidence_count": verified_license_plate_evidence_count,
        "verified_license_plate_crop_path": verified_license_plate_crop_path,
        "possible_license_plate_candidates": possible_license_plate_candidates,
        "weak_ocr_text": sorted({text for text in weak_ocr_text if text}),
        "invalid_ocr_text": sorted({text for text in invalid_ocr_text if text}),
        "best_license_plate_promoted_to_verified": best_license_plate_promoted_to_verified,
        "best_license_plate_verification_reject_reason": best_license_plate_verification_reject_reason,
        "best_vehicle_color": best_vehicle_color,
        "best_color_source": best_color_source,
        "best_ocr_crop_path": best_ocr_crop_path,
        "best_color_crop_path": best_color_crop_path,
        "vehicle_attributes": _aggregate_attribute_group(track_items, "vehicle_attributes"),
        "license_plate_attributes": _aggregate_attribute_group(track_items, "license_plate_attributes"),
        "scene_attributes": _aggregate_attribute_group(track_items, "scene_attributes"),
        "confidence_summary": {
            "min": round(min(float(item.get("confidence", 0.0) or 0.0) for item in track_items), 6),
            "max": round(max(float(item.get("confidence", 0.0) or 0.0) for item in track_items), 6),
            "avg": round(sum(float(item.get("confidence", 0.0) or 0.0) for item in track_items) / len(track_items), 6),
        },
        "all_valid_candidate_plates": all_valid_candidate_plates,
        "all_rejected_candidate_plates": rejected_plate_candidates,
        "all_candidate_colors": [
            {
                "color": str(item.get("parsed_vehicle_color", "unknown")),
                "source": str(item.get("parsed_color_source", "unknown")),
                "selection_group": str(item.get("selection_group", "")),
                "crop_path": str(item.get("selected_crop_path", "")),
            }
            for item in track_items
            if str(item.get("parsed_vehicle_color", "unknown")) != "unknown"
        ],
        "recommendation": (
            "Track has validated OCR/color evidence. Prefer this track for later search index enrichment."
            if best_plate_valid or best_vehicle_color != "unknown"
            else "Track has weak OCR/color evidence. Keep as supporting vehicle context."
        ),
        "crop_results": track_items,
    }


def _summarize_results(
    *,
    crop_results: list[dict[str, Any]],
    track_results: list[dict[str, Any]],
    queue_counts: dict[str, int],
    plate_crops_found: int,
    plate_detector_load_status: str,
    adapter_load_status: str,
    base_model_files: dict[str, Any],
    adapter_files: dict[str, Any],
    processing_seconds_values: list[float],
    selected_crop_count: int,
    source_selected_track_count: int,
) -> dict[str, Any]:
    """Build the Step 06 report payload with strict plate metrics."""

    colour_counts: Counter[str] = Counter(
        item["best_vehicle_color"]
        for item in track_results
        if str(item.get("best_vehicle_color", "unknown")) != "unknown"
    )
    tracks_with_raw_plate_text = sum(1 for item in track_results if bool(item.get("has_raw_plate_text")))
    tracks_with_valid_plate_text = sum(1 for item in track_results if bool(item.get("best_license_plate_valid")))
    tracks_with_invalid_plate_text_only = sum(
        1
        for item in track_results
        if bool(item.get("has_raw_plate_text")) and not bool(item.get("best_license_plate_valid"))
    )
    tracks_with_vehicle_color = sum(1 for item in track_results if item["best_vehicle_color"] != "unknown")
    primary_tracks_with_valid_plate_text = sum(
        1 for item in track_results if item["selection_group"] == "primary" and bool(item.get("best_license_plate_valid"))
    )
    fallback_tracks_with_valid_plate_text = sum(
        1 for item in track_results if item["selection_group"] == "fallback" and bool(item.get("best_license_plate_valid"))
    )
    primary_tracks_with_color = sum(
        1 for item in track_results if item["selection_group"] == "primary" and item["best_vehicle_color"] != "unknown"
    )
    fallback_tracks_with_color = sum(
        1 for item in track_results if item["selection_group"] == "fallback" and item["best_vehicle_color"] != "unknown"
    )

    valid_unique_license_plates = sorted(
        {
            str(item.get("best_license_plate_text", ""))
            for item in track_results
            if bool(item.get("best_license_plate_valid"))
        }
    )
    rejected_plate_reason_counts: Counter[str] = Counter()
    rejected_plate_examples: list[dict[str, Any]] = []
    rejected_plate_candidate_count = 0
    verified_unique_license_plates = sorted(
        {
            str(item.get("verified_license_plate", ""))
            for item in track_results
            if bool(item.get("verified_license_plate_valid"))
        }
    )
    possible_unique_license_plates = sorted(
        {
            str(candidate.get("text", ""))
            for item in track_results
            for candidate in list(item.get("possible_license_plate_candidates", []))
            if str(candidate.get("text", ""))
        }
    )
    for track_item in track_results:
        for rejected_item in list(track_item.get("all_rejected_candidate_plates", [])):
            rejected_plate_candidate_count += 1
            rejected_plate_reason_counts[str(rejected_item.get("reject_reason", "unknown"))] += 1
            if len(rejected_plate_examples) < 10:
                rejected_plate_examples.append(
                    {
                        "track_id": track_item["track_id"],
                        "text": str(rejected_item.get("text", "")),
                        "source": str(rejected_item.get("source", "none")),
                        "reject_reason": str(rejected_item.get("reject_reason", "unknown")),
                    }
                )
        if (
            bool(track_item.get("has_raw_plate_text"))
            and not bool(track_item.get("verified_license_plate_valid"))
            and not bool(track_item.get("possible_license_plate_candidates"))
        ):
            rejected_plate_reason_counts["no_verified_or_possible_candidate"] += 1

    successful_crop_count = sum(1 for item in crop_results if item.get("status") == "success")
    failed_crop_count = len(crop_results) - successful_crop_count
    vehicle_attribute_counts: Counter[str] = Counter()
    for item in track_results:
        for key, value in dict(item.get("vehicle_attributes", {})).items():
            if key not in {"source", "confidence"} and value not in (None, "", [], {}, "unknown", "none"):
                vehicle_attribute_counts[key] += 1

    return {
        "status": "success",
        "selected_crop_count": selected_crop_count,
        "processed_crop_count": len(crop_results),
        "primary_processed_count": queue_counts["primary"],
        "fallback_processed_count": queue_counts["fallback"],
        "successful_crop_count": successful_crop_count,
        "failed_crop_count": failed_crop_count,
        "plate_detector_available": plate_detector_load_status == "success",
        "plate_detector_load_status": plate_detector_load_status,
        "adapter_load_status": adapter_load_status,
        "base_model_missing_files": base_model_files["missing_files"],
        "adapter_missing_files": adapter_files["missing_files"],
        "plate_crops_found": plate_crops_found,
        "tracks_with_raw_plate_text": tracks_with_raw_plate_text,
        "tracks_with_valid_plate_text": tracks_with_valid_plate_text,
        "tracks_with_invalid_plate_text_only": tracks_with_invalid_plate_text_only,
        "tracks_with_vehicle_color": tracks_with_vehicle_color,
        "tracks_with_vehicle_make": vehicle_attribute_counts.get("make", 0),
        "tracks_with_vehicle_model": vehicle_attribute_counts.get("model", 0),
        "tracks_with_vehicle_body_type": vehicle_attribute_counts.get("body_type", 0),
        "structured_vehicle_attribute_counts": dict(sorted(vehicle_attribute_counts.items())),
        "florence_task_counts": {
            "ocr": len(crop_results),
            "caption": len(crop_results),
            "detailed_caption": 0,
            "plate_ocr_variants": sum(len(list(item.get("plate_ocr_variants", []))) for item in crop_results),
        },
        "tracks_with_verified_license_plate": sum(1 for item in track_results if bool(item.get("verified_license_plate_valid"))),
        "verified_unique_license_plate_count": len(verified_unique_license_plates),
        "verified_unique_license_plates": verified_unique_license_plates,
        "tracks_with_possible_license_plate_only": sum(
            1
            for item in track_results
            if not bool(item.get("verified_license_plate_valid")) and bool(item.get("possible_license_plate_candidates"))
        ),
        "possible_unique_license_plate_count": len(possible_unique_license_plates),
        "possible_unique_license_plates": possible_unique_license_plates,
        "invalid_ocr_text_count": sum(len(list(item.get("invalid_ocr_text", []))) for item in track_results),
        "primary_tracks_with_plate_text": primary_tracks_with_valid_plate_text,
        "fallback_tracks_with_plate_text": fallback_tracks_with_valid_plate_text,
        "primary_tracks_with_color": primary_tracks_with_color,
        "fallback_tracks_with_color": fallback_tracks_with_color,
        "valid_unique_license_plate_count": len(valid_unique_license_plates),
        "valid_unique_license_plates": valid_unique_license_plates,
        "rejected_plate_candidate_count": rejected_plate_candidate_count,
        "rejected_plate_reason_counts": dict(sorted(rejected_plate_reason_counts.items())),
        "colour_counts": dict(sorted(colour_counts.items())),
        "top_valid_plate_results": [
            {
                "track_id": item["track_id"],
                "plate_text": item["best_license_plate_text"],
                "source": item["best_license_plate_source"],
                "selection_group": item["selection_group"],
                "confidence_level": item["best_license_plate_confidence_level"],
                "evidence_count": item["best_license_plate_evidence_count"],
            }
            for item in track_results
            if bool(item.get("best_license_plate_valid"))
        ][:10],
        "top_verified_plate_results": [
            {
                "track_id": item["track_id"],
                "verified_license_plate": item["verified_license_plate"],
                "source": item["verified_license_plate_source"],
                "selection_group": item["selection_group"],
                "evidence_count": item["verified_license_plate_evidence_count"],
                "confidence_level": item["verified_license_plate_confidence_level"],
            }
            for item in track_results
            if bool(item.get("verified_license_plate_valid"))
        ][:10],
        "top_possible_plate_results": [
            {
                "track_id": item["track_id"],
                "text": str(candidate.get("text", "")),
                "source": str(candidate.get("source", "none")),
                "reason": str(candidate.get("reason", "possible_candidate")),
            }
            for item in track_results
            for candidate in list(item.get("possible_license_plate_candidates", []))
        ][:10],
        "top_rejected_plate_examples": rejected_plate_examples,
        "failed_examples": [
            {
                "track_id": item["track_id"],
                "detection_id": item["detection_id"],
                "selected_crop_path": item["selected_crop_path"],
                "error_message": item["error_message"],
            }
            for item in crop_results
            if item["status"] != "success"
        ][:10],
        "avg_processing_seconds_per_crop": round(sum(processing_seconds_values) / len(processing_seconds_values), 6)
        if processing_seconds_values
        else 0.0,
        "total_processing_seconds": round(sum(processing_seconds_values), 6),
        "recommendation": "Proceed to Step 07 object search index enrichment.",
        "source_selected_track_count": source_selected_track_count,
    }


def _write_outputs(
    *,
    run_dir: Path,
    reuse_existing_raw_results: bool,
    results_payload: dict[str, Any],
    report_payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write normal or cleaned Step 06 outputs."""

    if reuse_existing_raw_results:
        results_path = run_dir / "06_ocr_color_results_cleaned.json"
        report_path = run_dir / "06_ocr_color_report_cleaned.json"
    else:
        results_path = run_dir / "06_ocr_color_results.json"
        report_path = run_dir / "06_ocr_color_report.json"
    write_json(results_path, results_payload)
    write_json(report_path, report_payload)
    return results_path, report_path


def _write_verified_outputs(
    *,
    run_dir: Path,
    results_payload: dict[str, Any],
    report_payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write strict verified Step 06 outputs."""

    results_path = run_dir / "06_ocr_color_results_verified.json"
    report_path = run_dir / "06_ocr_color_report_verified.json"
    write_json(results_path, results_payload)
    write_json(report_path, report_payload)
    return results_path, report_path


def verify_existing_step06_outputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Build strict verified outputs from existing cleaned/raw Step 06 results."""

    cleaned_results_path = run_dir / "06_ocr_color_results_cleaned.json"
    raw_results_path = run_dir / "06_ocr_color_results.json"
    cleaned_report_path = run_dir / "06_ocr_color_report_cleaned.json"
    raw_report_path = run_dir / "06_ocr_color_report.json"

    if cleaned_results_path.exists():
        source_results_path = cleaned_results_path
        source_report_path = cleaned_report_path if cleaned_report_path.exists() else raw_report_path
    elif raw_results_path.exists():
        source_results_path = raw_results_path
        source_report_path = raw_report_path
    else:
        raise FileNotFoundError(
            "No Step 06 source results file found for verification. Expected one of: "
            f"{cleaned_results_path} or {raw_results_path}"
        )

    results_payload = read_json(source_results_path)
    report_payload = read_json(source_report_path) if source_report_path.exists() else {}
    verified_results_payload, verified_report_payload, _tmp_results, _tmp_report = _clean_existing_results(
        run_dir=run_dir,
        results_payload=results_payload,
        report_payload=report_payload,
        reuse_existing_raw_results=False,
    )
    verified_report_payload["source_report"] = source_report_path.name
    verified_results_payload["source_results_file"] = source_results_path.name
    results_path, report_path = _write_verified_outputs(
        run_dir=run_dir,
        results_payload=verified_results_payload,
        report_payload=verified_report_payload,
    )
    return verified_results_payload, verified_report_payload, results_path, report_path


def _clean_existing_results(
    *,
    run_dir: Path,
    results_payload: dict[str, Any],
    report_payload: dict[str, Any] | None,
    reuse_existing_raw_results: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Reprocess saved raw OCR fields without rerunning Florence."""

    track_results = list(results_payload.get("track_results", []))
    all_crop_results: list[dict[str, Any]] = []
    processing_seconds_values: list[float] = []
    plate_crops_found = 0

    for track_item in track_results:
        cleaned_crop_results: list[dict[str, Any]] = []
        for crop_item in list(track_item.get("crop_results", [])):
            crop_copy = dict(crop_item)
            plate_selection = _select_best_plate_candidate(crop_copy)
            crop_copy.update(plate_selection)
            crop_path = _resolve_run_relative(run_dir, str(crop_copy.get("selected_crop_path", "") or ""))
            if crop_path is None or not crop_path.exists():
                raise FileNotFoundError(f"Selected crop does not exist: {crop_copy.get('selected_crop_path')}")
            color_result = resolve_vehicle_color(str(crop_copy.get("caption_raw", "") or ""), crop_path)
            crop_copy["parsed_vehicle_color"] = color_result["color"]
            crop_copy["parsed_vehicle_color_raw"] = color_result["raw_prediction"]
            crop_copy["parsed_color_source"] = color_result["source"]
            crop_copy["parsed_color_confidence"] = color_result["confidence"]
            crop_copy["image_color_confidence"] = color_result["image_confidence"]
            crop_copy.update(
                extract_structured_florence_metadata(
                    caption_text=str(crop_copy.get("caption_raw", "") or ""),
                    ocr_text=str(crop_copy.get("plate_crop_ocr_raw") or crop_copy.get("vehicle_crop_ocr_raw") or ""),
                    plate_found=bool(crop_copy.get("plate_crop_found")),
                    plate_confidence=float(crop_copy.get("plate_confidence", 0.0) or 0.0),
                    plate_text=str(crop_copy.get("parsed_license_plate_text", "not_visible") or "not_visible"),
                    plate_valid=bool(crop_copy.get("parsed_license_plate_valid")),
                    resolved_color=color_result,
                )
            )
            if crop_copy.get("plate_crop_found"):
                plate_crops_found += 1
            cleaned_crop_results.append(crop_copy)
            all_crop_results.append(crop_copy)
            processing_seconds_values.append(float(crop_copy.get("processing_seconds", 0.0) or 0.0))

        aggregated_track = _aggregate_track_results(cleaned_crop_results)
        track_item.clear()
        track_item.update(aggregated_track)

    track_results.sort(key=lambda item: (item["selection_group"] != "primary", item["track_id"]))
    queue_counts = {
        "primary": sum(1 for item in all_crop_results if item.get("selection_group") == "primary"),
        "fallback": sum(1 for item in all_crop_results if item.get("selection_group") == "fallback"),
    }

    results_payload["status"] = "success"
    results_payload["processed_crop_count"] = len(all_crop_results)
    results_payload["successful_crop_count"] = sum(1 for item in all_crop_results if item.get("status") == "success")
    results_payload["failed_crop_count"] = len(all_crop_results) - results_payload["successful_crop_count"]
    results_payload["track_results"] = track_results

    base_model_files = {"missing_files": []}
    adapter_files = {"missing_files": []}
    updated_report = _summarize_results(
        crop_results=all_crop_results,
        track_results=track_results,
        queue_counts=queue_counts,
        plate_crops_found=plate_crops_found,
        plate_detector_load_status=str((report_payload or {}).get("plate_detector_load_status", "unknown")),
        adapter_load_status=str((report_payload or {}).get("adapter_load_status", "unknown")),
        base_model_files=base_model_files,
        adapter_files=adapter_files,
        processing_seconds_values=processing_seconds_values,
        selected_crop_count=int(results_payload.get("selected_crop_count", len(all_crop_results)) or len(all_crop_results)),
        source_selected_track_count=int((report_payload or {}).get("source_selected_track_count", len(track_results)) or len(track_results)),
    )

    results_path, report_path = _write_outputs(
        run_dir=run_dir,
        reuse_existing_raw_results=reuse_existing_raw_results,
        results_payload=results_payload,
        report_payload=updated_report,
    )
    return results_payload, updated_report, results_path, report_path


def test_clean_plate_text_examples() -> list[str]:
    """Run small internal OCR cleaning tests and return warnings."""

    warnings: list[str] = []
    positive_cases = {
        "DL12CL4316 {'<OCR>': 'DL12CL4316'}": "DL12CL4316",
        "DL12CL4316OCRDL12CL4316": "DL12CL4316",
        "DL4SAE0084 {'<OCR>': 'DL4SAE0084'}": "DL4SAE0084",
        "HR38AH0181 {'<OCR>': 'HR38AH0181'}": "HR38AH0181",
    }
    negative_cases = [
        "UNANSWERABLEOCRUNANSWERABLE",
        "STOPOCRSTOP",
        "LANCERAILAIRAMBULANCEOCRLANCERAILAIRAMBULANCE",
        "COCRC",
        "ROCRR",
    ]

    for raw_text, expected in positive_cases.items():
        cleaned = remove_repeated_plate_artifacts(clean_florence_ocr_text(raw_text))
        if cleaned != expected:
            warnings.append(f"Cleaning test mismatch for {raw_text!r}: expected {expected!r}, got {cleaned!r}")

    for raw_text in negative_cases:
        cleaned = remove_repeated_plate_artifacts(clean_florence_ocr_text(raw_text))
        is_valid, _reason = is_valid_license_plate_candidate(cleaned)
        if is_valid:
            warnings.append(f"Negative cleaning test should be invalid for {raw_text!r}, got {cleaned!r}")

    return warnings


def test_verified_plate_classification_examples() -> list[str]:
    """Run internal verification tests and return warnings."""

    warnings: list[str] = []
    verified_cases = ["DL12CL4316", "DL4SAE0084", "HR47F3216", "HR38AH0181", "DL1LR9174", "UP16ED1448"]
    possible_cases = ["CITYDL1FT", "CGHTRCTAX1", "DALESK1", "DET4X0", "Z0BEST", "VETB95", "PATBR1699", "AS14UP", "1426LK", "DT3581"]
    invalid_cases = ["UNANSWERABLE", "STOP", "VICTORY", "DEV", "C", "F", "0", "JE"]

    for text in verified_cases:
        result = classify_plate_candidate({"text": text, "source": "plate_crop_ocr", "plate_format_confidence": "high"})
        if result["classification"] != "verified":
            warnings.append(f"Verified classification mismatch for {text!r}: got {result}")

    for text in possible_cases:
        result = classify_plate_candidate({"text": text, "source": "vehicle_crop_ocr", "plate_format_confidence": "medium"})
        if result["classification"] != "possible":
            warnings.append(f"Possible classification mismatch for {text!r}: got {result}")

    for text in invalid_cases:
        result = classify_plate_candidate({"text": text, "source": "plate_crop_ocr", "plate_format_confidence": "high"})
        if result["classification"] != "invalid":
            warnings.append(f"Invalid classification mismatch for {text!r}: got {result}")

    return warnings


def run_ocr_color_enrichment(
    *,
    run_dir: Path,
    florence_model_path: Path,
    florence_adapter_path: Path | None,
    plate_detector_model_path: Path | None,
    process_groups: list[str],
    primary_limit: int,
    fallback_limit: int,
    device: str,
    max_new_tokens: int,
    num_beams: int,
    save_plate_crops: bool,
    save_debug_images: bool,
    min_plate_confidence: float,
    reuse_existing_raw_results: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Run Step 06 OCR/color enrichment on Step 05 selected crops or clean existing raw results."""

    if reuse_existing_raw_results:
        existing_results_path = run_dir / "06_ocr_color_results.json"
        if not existing_results_path.exists():
            raise FileNotFoundError(
                f"Reuse mode requested, but the raw Step 06 results file is missing: {existing_results_path}"
            )
        existing_report_path = run_dir / "06_ocr_color_report.json"
        existing_results_payload = read_json(existing_results_path)
        existing_report_payload = read_json(existing_report_path) if existing_report_path.exists() else None
        return _clean_existing_results(
            run_dir=run_dir,
            results_payload=existing_results_payload,
            report_payload=existing_report_payload,
            reuse_existing_raw_results=True,
        )

    selection_payload = read_json(run_dir / "05_best_track_frames.json")
    selection_report = read_json(run_dir / "05_best_track_frames_report.json")

    plate_dir = run_dir / "06_plate_crops"
    debug_dir = run_dir / "06_debug_images"
    plate_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    base_model_files = inspect_path_files(florence_model_path, BASE_MODEL_REQUIRED_FILES)
    adapter_files = inspect_path_files(florence_adapter_path, ADAPTER_EXPECTED_FILES)
    queue, queue_counts = _prepare_selected_crops(
        selection_payload,
        set(process_groups),
        primary_limit,
        fallback_limit,
    )

    if selection_payload.get("status") == "no_vehicle_tracks" or not queue:
        empty_results_payload = {
            "status": "no_selected_crops",
            "input_best_frames_file": "05_best_track_frames.json",
            "florence_model_path": str(florence_model_path),
            "florence_adapter_path": str(florence_adapter_path) if florence_adapter_path is not None else None,
            "plate_detector_model_path": str(plate_detector_model_path) if plate_detector_model_path is not None else None,
            "device_used": None,
            "process_groups": process_groups,
            "selected_crop_count": 0,
            "processed_crop_count": 0,
            "successful_crop_count": 0,
            "failed_crop_count": 0,
            "track_results": [],
        }
        empty_report_payload = {
            "status": "no_selected_crops",
            "selected_crop_count": 0,
            "processed_crop_count": 0,
            "primary_processed_count": 0,
            "fallback_processed_count": 0,
            "successful_crop_count": 0,
            "failed_crop_count": 0,
            "plate_detector_available": False,
            "plate_crops_found": 0,
            "tracks_with_raw_plate_text": 0,
            "tracks_with_valid_plate_text": 0,
            "tracks_with_invalid_plate_text_only": 0,
            "valid_unique_license_plate_count": 0,
            "valid_unique_license_plates": [],
            "rejected_plate_candidate_count": 0,
            "rejected_plate_reason_counts": {},
            "tracks_with_vehicle_color": 0,
            "top_valid_plate_results": [],
            "top_rejected_plate_examples": [],
            "failed_examples": [],
            "avg_processing_seconds_per_crop": 0.0,
            "recommendation": "No selected vehicle crops were available for Step 06.",
        }
        results_path, report_path = _write_outputs(
            run_dir=run_dir,
            reuse_existing_raw_results=False,
            results_payload=empty_results_payload,
            report_payload=empty_report_payload,
        )
        return empty_results_payload, empty_report_payload, results_path, report_path

    processor, model, device_used = load_florence_model(florence_model_path, device)
    model, adapter_load_status = _maybe_apply_adapter(model, florence_adapter_path)
    plate_detector_load_status, plate_detector = _load_plate_detector(plate_detector_model_path)

    crop_results: list[dict[str, Any]] = []
    plate_crops_found = 0
    processing_seconds_values: list[float] = []

    for crop_item in queue:
        started_at = time.perf_counter()
        result_payload = {
            "track_id": crop_item["track_id"],
            "selection_group": crop_item["selection_group"],
            "quality_label": crop_item["quality_label"],
            "detection_id": crop_item["detection_id"],
            "frame_id": crop_item["frame_id"],
            "frame_idx": int(crop_item.get("frame_idx", 0) or 0),
            "timestamp_seconds": float(crop_item.get("timestamp_seconds", 0.0) or 0.0),
            "class_name": str(crop_item.get("class_name", "")),
            "dominant_class_name": str(crop_item.get("dominant_class_name", "")),
            "confidence": round(float(crop_item.get("confidence", 0.0) or 0.0), 6),
            "final_selection_score": round(float(crop_item.get("final_selection_score", 0.0) or 0.0), 6),
            "selected_crop_path": str(crop_item.get("selected_crop_path", "")),
            "plate_detector_status": "not_started",
            "plate_crop_found": False,
            "plate_crop_path": None,
            "plate_crop_paths": [],
            "plate_enhanced_crop_path": None,
            "plate_confidence": 0.0,
            "plate_bbox_xyxy": None,
            "plate_crop_bbox_xyxy": None,
            "plate_crop_padding_pixels": None,
            "vehicle_crop_ocr_raw": "",
            "plate_crop_ocr_raw": "",
            "plate_ocr_variants": [],
            "vehicle_crop_ocr_cleaned": "",
            "plate_crop_ocr_cleaned": "",
            "parsed_license_plate_text": "not_visible",
            "parsed_license_plate_valid": False,
            "parsed_license_plate_source": "none",
            "plate_format_confidence": "none",
            "license_plate_reject_reason": None,
            "rejected_plate_candidates": [],
            "caption_raw": "",
            "detailed_caption_raw": "",
            "parsed_vehicle_color": "unknown",
            "parsed_vehicle_color_raw": None,
            "parsed_color_source": "unknown",
            "parsed_color_confidence": "none",
            "image_color_confidence": None,
            "vehicle_attributes": {},
            "license_plate_attributes": {},
            "scene_attributes": {},
            "status": "failed",
            "error_message": None,
            "processing_seconds": 0.0,
        }

        try:
            crop_path = _resolve_run_relative(run_dir, result_payload["selected_crop_path"])
            if crop_path is None or not crop_path.exists():
                raise FileNotFoundError(f"Selected crop does not exist: {result_payload['selected_crop_path']}")

            plate_detection_payload = _run_plate_detection(
                run_dir=run_dir,
                crop_item=crop_item,
                plate_detector=plate_detector,
                plate_dir=plate_dir,
                debug_dir=debug_dir,
                save_plate_crops=save_plate_crops,
                save_debug_images=save_debug_images,
                min_plate_confidence=min_plate_confidence,
                device=device_used,
            )
            result_payload.update(plate_detection_payload)
            if result_payload["plate_crop_found"]:
                plate_crops_found += 1

            vehicle_ocr = run_florence_generation(
                image_path=crop_path,
                processor=processor,
                model=model,
                device_used=device_used,
                task_prompt="<OCR>",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            caption_result = run_florence_generation(
                image_path=crop_path,
                processor=processor,
                model=model,
                device_used=device_used,
                task_prompt="<CAPTION>",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            result_payload["vehicle_crop_ocr_raw"] = _stringify_generation_result(vehicle_ocr)
            result_payload["caption_raw"] = _stringify_generation_result(caption_result)
            result_payload["detailed_caption_raw"] = ""

            if result_payload["plate_crop_found"]:
                plate_ocr_variants: list[dict[str, Any]] = []
                for plate_variant_path_value in list(result_payload.get("plate_crop_paths", [])):
                    plate_crop_path = _resolve_run_relative(run_dir, str(plate_variant_path_value))
                    if plate_crop_path is None or not plate_crop_path.exists():
                        continue
                    plate_ocr = run_florence_generation(
                        image_path=plate_crop_path,
                        processor=processor,
                        model=model,
                        device_used=device_used,
                        task_prompt="<OCR>",
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                    plate_ocr_variants.append(
                        {
                            "variant": "enhanced" if "enhanced" in plate_crop_path.stem else "padded_original",
                            "image_path": str(plate_variant_path_value),
                            "ocr_raw": _stringify_generation_result(plate_ocr),
                        }
                    )
                result_payload["plate_ocr_variants"] = plate_ocr_variants
                best_plate_variant = _select_best_plate_ocr_variant(plate_ocr_variants)
                if best_plate_variant is not None:
                    result_payload["plate_crop_ocr_raw"] = str(best_plate_variant.get("ocr_raw", "") or "")

            result_payload.update(_select_best_plate_candidate(result_payload))

            color_result = resolve_vehicle_color(result_payload["caption_raw"], crop_path)
            result_payload["parsed_vehicle_color"] = color_result["color"]
            result_payload["parsed_vehicle_color_raw"] = color_result["raw_prediction"]
            result_payload["parsed_color_source"] = color_result["source"]
            result_payload["parsed_color_confidence"] = color_result["confidence"]
            result_payload["image_color_confidence"] = color_result["image_confidence"]
            result_payload.update(
                extract_structured_florence_metadata(
                    caption_text=result_payload["caption_raw"],
                    ocr_text=result_payload["plate_crop_ocr_raw"] or result_payload["vehicle_crop_ocr_raw"],
                    plate_found=bool(result_payload["plate_crop_found"]),
                    plate_confidence=float(result_payload["plate_confidence"] or 0.0),
                    plate_text=str(result_payload["parsed_license_plate_text"]),
                    plate_valid=bool(result_payload["parsed_license_plate_valid"]),
                    resolved_color=color_result,
                )
            )

            result_payload["status"] = "success"
        except Exception as exc:
            result_payload["error_message"] = str(exc)

        result_payload["processing_seconds"] = round(time.perf_counter() - started_at, 6)
        processing_seconds_values.append(result_payload["processing_seconds"])
        crop_results.append(result_payload)

    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in crop_results:
        grouped_results[str(item["track_id"])].append(item)

    track_results = [_aggregate_track_results(track_items) for track_items in grouped_results.values()]
    track_results.sort(key=lambda item: (item["selection_group"] != "primary", item["track_id"]))

    results_payload = {
        "status": "success",
        "input_best_frames_file": "05_best_track_frames.json",
        "florence_model_path": str(florence_model_path),
        "florence_adapter_path": str(florence_adapter_path) if florence_adapter_path is not None else None,
        "plate_detector_model_path": str(plate_detector_model_path) if plate_detector_model_path is not None else None,
        "device_used": device_used,
        "model_device": str(getattr(model, "device", device_used)),
        "cuda_memory_allocated_mb": cuda_memory_allocated_mb(),
        "cuda_memory_reserved_mb": cuda_memory_reserved_mb(),
        "process_groups": process_groups,
        "selected_crop_count": len(queue),
        "processed_crop_count": len(crop_results),
        "successful_crop_count": sum(1 for item in crop_results if item.get("status") == "success"),
        "failed_crop_count": sum(1 for item in crop_results if item.get("status") != "success"),
        "track_results": track_results,
    }
    report_payload = _summarize_results(
        crop_results=crop_results,
        track_results=track_results,
        queue_counts=queue_counts,
        plate_crops_found=plate_crops_found,
        plate_detector_load_status=plate_detector_load_status,
        adapter_load_status=adapter_load_status,
        base_model_files=base_model_files,
        adapter_files=adapter_files,
        processing_seconds_values=processing_seconds_values,
        selected_crop_count=len(queue),
        source_selected_track_count=int(selection_report.get("selected_track_count", 0) or 0),
    )

    results_path, report_path = _write_outputs(
        run_dir=run_dir,
        reuse_existing_raw_results=False,
        results_payload=results_payload,
        report_payload=report_payload,
    )
    return results_payload, report_payload, results_path, report_path
