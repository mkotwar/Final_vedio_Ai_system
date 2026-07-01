from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.tracker_adapter import (
    crop_bbox_from_image,
    to_absolute_repo_path,
    to_repo_relative_path,
)
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_OCR_BACKEND = "FINAL_DEMO_OCR_BACKEND"
ENV_FINAL_DEMO_OCR_DEVICE = "FINAL_DEMO_OCR_DEVICE"
ENV_FINAL_DEMO_OCR_REQUIRE_GPU = "FINAL_DEMO_OCR_REQUIRE_GPU"
ENV_FINAL_DEMO_OCR_GPU_ID = "FINAL_DEMO_OCR_GPU_ID"
ENV_FINAL_DEMO_OCR_MIN_CONFIDENCE = "FINAL_DEMO_OCR_MIN_CONFIDENCE"
ENV_FINAL_DEMO_OCR_STRONG_CONFIDENCE = "FINAL_DEMO_OCR_STRONG_CONFIDENCE"
ENV_FINAL_DEMO_OCR_MIN_STRONG_TEXT_LENGTH = "FINAL_DEMO_OCR_MIN_STRONG_TEXT_LENGTH"
ENV_FINAL_DEMO_OCR_STRONG_MIN_OCR_CONF = "FINAL_DEMO_OCR_STRONG_MIN_OCR_CONF"
ENV_FINAL_DEMO_OCR_SAVE_DEBUG_CROPS = "FINAL_DEMO_OCR_SAVE_DEBUG_CROPS"
ENV_FINAL_DEMO_OCR_NORMALIZE_TEXT = "FINAL_DEMO_OCR_NORMALIZE_TEXT"
ENV_FINAL_DEMO_OCR_MAX_TRACKS = "FINAL_DEMO_OCR_MAX_TRACKS"
ENV_FINAL_DEMO_OCR_FRAMES_PER_TRACK = "FINAL_DEMO_OCR_FRAMES_PER_TRACK"
ENV_FINAL_DEMO_OCR_USE_ORIGINAL_FRAME = "FINAL_DEMO_OCR_USE_ORIGINAL_FRAME"
ENV_FINAL_DEMO_OCR_USE_LEGACY_GUESSED_REGIONS = "FINAL_DEMO_OCR_USE_LEGACY_GUESSED_REGIONS"
ENV_FINAL_DEMO_OCR_MAX_FRAME_SCAN_CANDIDATES = "FINAL_DEMO_OCR_MAX_FRAME_SCAN_CANDIDATES"
ENV_FINAL_DEMO_OCR_FRAME_SCAN_TOPK_PER_FRAME = "FINAL_DEMO_OCR_FRAME_SCAN_TOPK_PER_FRAME"
ENV_FINAL_DEMO_OCR_FRAME_SCAN_MIN_CANDIDATE_SCORE = "FINAL_DEMO_OCR_FRAME_SCAN_MIN_CANDIDATE_SCORE"

DEFAULT_OCR_BACKEND = "auto"
DEFAULT_OCR_DEVICE = "auto"
DEFAULT_OCR_MIN_CONFIDENCE = 0.45
DEFAULT_OCR_STRONG_CONFIDENCE = 0.70
DEFAULT_OCR_MIN_STRONG_TEXT_LENGTH = 6
DEFAULT_OCR_STRONG_MIN_OCR_CONF = 0.45
DEFAULT_OCR_SAVE_DEBUG_CROPS = True
DEFAULT_OCR_NORMALIZE_TEXT = True
DEFAULT_OCR_FRAMES_PER_TRACK = 5
DEFAULT_OCR_USE_ORIGINAL_FRAME = False
DEFAULT_OCR_USE_LEGACY_GUESSED_REGIONS = False
DEFAULT_OCR_MAX_FRAME_SCAN_CANDIDATES = 100
DEFAULT_OCR_FRAME_SCAN_TOPK_PER_FRAME = 3
DEFAULT_OCR_FRAME_SCAN_MIN_CANDIDATE_SCORE = 0.35

ALLOWED_OCR_BACKENDS = {"auto", "paddleocr", "easyocr", "disabled"}
ALLOWED_OCR_DEVICES = {"auto", "gpu", "cpu"}
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB",
    "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}
INDIAN_SERIES_DIGIT_TO_LETTER = {
    "4": "A",
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "6": "G",
    "2": "Z",
}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}"
    )


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def read_non_negative_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_optional_positive_int_env(env_name: str) -> int | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return read_positive_int_env(env_name, 1)


def read_ocr_settings() -> dict[str, Any]:
    backend = os.environ.get(ENV_FINAL_DEMO_OCR_BACKEND, DEFAULT_OCR_BACKEND).strip().lower()
    if backend not in ALLOWED_OCR_BACKENDS:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_OCR_BACKEND} must be one of "
            f"{sorted(ALLOWED_OCR_BACKENDS)}. Received: {backend!r}"
        )
    device = os.environ.get(ENV_FINAL_DEMO_OCR_DEVICE, DEFAULT_OCR_DEVICE).strip().lower()
    if device not in ALLOWED_OCR_DEVICES:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_OCR_DEVICE} must be one of "
            f"{sorted(ALLOWED_OCR_DEVICES)}. Received: {device!r}"
        )
    min_confidence = round(
        read_non_negative_float_env(ENV_FINAL_DEMO_OCR_MIN_CONFIDENCE, DEFAULT_OCR_MIN_CONFIDENCE),
        3,
    )
    strong_confidence = round(
        read_non_negative_float_env(ENV_FINAL_DEMO_OCR_STRONG_CONFIDENCE, DEFAULT_OCR_STRONG_CONFIDENCE),
        3,
    )
    if strong_confidence < min_confidence:
        raise ValueError(
            f"{ENV_FINAL_DEMO_OCR_STRONG_CONFIDENCE} must be greater than or equal to "
            f"{ENV_FINAL_DEMO_OCR_MIN_CONFIDENCE}."
        )
    return {
        "ocr_backend_requested": backend,
        "ocr_device_requested": device,
        "require_gpu": read_bool_env(ENV_FINAL_DEMO_OCR_REQUIRE_GPU, False),
        "ocr_gpu_id": read_non_negative_int_env(ENV_FINAL_DEMO_OCR_GPU_ID, 0),
        "min_confidence": min_confidence,
        "strong_confidence": strong_confidence,
        "min_strong_text_length": read_positive_int_env(
            ENV_FINAL_DEMO_OCR_MIN_STRONG_TEXT_LENGTH,
            DEFAULT_OCR_MIN_STRONG_TEXT_LENGTH,
        ),
        "strong_min_ocr_confidence": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_OCR_STRONG_MIN_OCR_CONF,
                DEFAULT_OCR_STRONG_MIN_OCR_CONF,
            ),
            3,
        ),
        "save_debug_crops": read_bool_env(
            ENV_FINAL_DEMO_OCR_SAVE_DEBUG_CROPS,
            DEFAULT_OCR_SAVE_DEBUG_CROPS,
        ),
        "normalize_text": read_bool_env(
            ENV_FINAL_DEMO_OCR_NORMALIZE_TEXT,
            DEFAULT_OCR_NORMALIZE_TEXT,
        ),
        "max_tracks": read_optional_positive_int_env(ENV_FINAL_DEMO_OCR_MAX_TRACKS),
        "frames_per_track": read_positive_int_env(
            ENV_FINAL_DEMO_OCR_FRAMES_PER_TRACK,
            DEFAULT_OCR_FRAMES_PER_TRACK,
        ),
        "use_original_frame": read_bool_env(
            ENV_FINAL_DEMO_OCR_USE_ORIGINAL_FRAME,
            DEFAULT_OCR_USE_ORIGINAL_FRAME,
        ),
        "use_legacy_guessed_regions": read_bool_env(
            ENV_FINAL_DEMO_OCR_USE_LEGACY_GUESSED_REGIONS,
            DEFAULT_OCR_USE_LEGACY_GUESSED_REGIONS,
        ),
        "max_frame_scan_candidates": read_positive_int_env(
            ENV_FINAL_DEMO_OCR_MAX_FRAME_SCAN_CANDIDATES,
            DEFAULT_OCR_MAX_FRAME_SCAN_CANDIDATES,
        ),
        "frame_scan_topk_per_frame": read_positive_int_env(
            ENV_FINAL_DEMO_OCR_FRAME_SCAN_TOPK_PER_FRAME,
            DEFAULT_OCR_FRAME_SCAN_TOPK_PER_FRAME,
        ),
        "frame_scan_min_candidate_score": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_OCR_FRAME_SCAN_MIN_CANDIDATE_SCORE,
                DEFAULT_OCR_FRAME_SCAN_MIN_CANDIDATE_SCORE,
            ),
            3,
        ),
    }


def build_device_context(warnings: list[str], settings: dict[str, Any]) -> dict[str, Any]:
    context = {
        "ocr_device_requested": settings["ocr_device_requested"],
        "ocr_device_used": "cpu",
        "cuda_available": False,
        "cuda_device_name": None,
        "gpu_enabled_for_ocr": False,
        "ocr_gpu_id": int(settings["ocr_gpu_id"]),
        "ocr_backend_init_status": "not_started",
        "ocr_backend_init_error": None,
        "torch_module": None,
    }
    try:
        import torch  # type: ignore

        context["torch_module"] = torch
        context["cuda_available"] = bool(torch.cuda.is_available())
        if context["cuda_available"]:
            gpu_id = min(int(settings["ocr_gpu_id"]), max(0, int(torch.cuda.device_count()) - 1))
            context["ocr_gpu_id"] = gpu_id
            context["cuda_device_name"] = str(torch.cuda.get_device_name(gpu_id))
    except Exception as exc:
        warnings.append(f"Could not inspect CUDA device information: {exc}")

    if settings["ocr_device_requested"] == "gpu":
        if context["cuda_available"]:
            context["ocr_device_used"] = f"cuda:{context['ocr_gpu_id']}"
            context["gpu_enabled_for_ocr"] = True
        else:
            context["ocr_device_used"] = "cpu"
    elif settings["ocr_device_requested"] == "auto" and context["cuda_available"]:
        context["ocr_device_used"] = f"cuda:{context['ocr_gpu_id']}"
        context["gpu_enabled_for_ocr"] = True
    else:
        context["ocr_device_used"] = "cpu"
    return context


def load_ocr_backend(
    requested_backend: str,
    settings: dict[str, Any],
    device_context: dict[str, Any],
    warnings: list[str],
) -> tuple[str | None, Any | None, bool]:
    if requested_backend == "disabled":
        device_context["ocr_backend_init_status"] = "disabled"
        return None, None, False

    if settings["require_gpu"] and not device_context["cuda_available"]:
        device_context["ocr_backend_init_status"] = "gpu_required_but_unavailable"
        device_context["ocr_backend_init_error"] = "CUDA unavailable while FINAL_DEMO_OCR_REQUIRE_GPU=1."
        return None, None, False

    backend_order = {
        "auto": ["easyocr", "paddleocr"],
        "paddleocr": ["paddleocr"],
        "easyocr": ["easyocr"],
    }[requested_backend]

    for backend_name in backend_order:
        if backend_name == "easyocr":
            try:
                import easyocr  # type: ignore

                use_gpu = bool(device_context["gpu_enabled_for_ocr"])
                reader = easyocr.Reader(["en"], gpu=use_gpu)
                device_context["ocr_backend_init_status"] = "ready"
                return "easyocr", reader, True
            except Exception as exc:
                device_context["ocr_backend_init_error"] = str(exc)
                warnings.append(f"Could not import or initialize EasyOCR: {exc}")
        if backend_name == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore

                use_gpu = bool(device_context["gpu_enabled_for_ocr"])
                reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu=use_gpu)
                device_context["ocr_backend_init_status"] = "ready"
                return "paddleocr", reader, True
            except Exception as exc:
                device_context["ocr_backend_init_error"] = str(exc)
                warnings.append(f"Could not import or initialize PaddleOCR: {exc}")

    if device_context["ocr_backend_init_status"] == "not_started":
        device_context["ocr_backend_init_status"] = "backend_missing"
    return None, None, False


def detect_track_source(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    clean_tracks_path = run_dir / "05B_clean_tracks.json"
    if clean_tracks_path.exists():
        payload = read_json(clean_tracks_path)
        return "05B_clean_tracks", list(payload.get("clean_tracks") or [])
    raw_tracks_path = run_dir / "05_tracks.json"
    if raw_tracks_path.exists():
        payload = read_json(raw_tracks_path)
        return "05_tracks", list(payload.get("tracks") or [])
    return "missing", []


def build_track_lookup(tracks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for track in tracks:
        track_id = str(track.get("clean_track_id") or track.get("local_track_id") or track.get("source_track_id") or "")
        if track_id:
            lookup[track_id] = track
    return lookup


def build_frame_lookup(frames_index_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(frames_index_payload, dict):
        return lookup
    for frame in list(frames_index_payload.get("frames") or []):
        if isinstance(frame, dict):
            frame_id = str(frame.get("frame_id") or "")
            if frame_id:
                lookup[frame_id] = frame
    return lookup


def build_plate_candidate_lookup(plate_candidates_payload: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(plate_candidates_payload, dict):
        return lookup
    for candidate in list(plate_candidates_payload.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("candidate_source") or "track_based") == "frame_scan":
            continue
        attribute_track_id = str(candidate.get("attribute_track_id") or "")
        if not attribute_track_id:
            continue
        lookup.setdefault(attribute_track_id, []).append(candidate)
    for attribute_track_id in list(lookup.keys()):
        lookup[attribute_track_id].sort(
            key=lambda item: (
                -float(item.get("plate_candidate_score", 0.0) or 0.0),
                str(item.get("frame_id") or ""),
            )
        )
    return lookup


def build_frame_scan_plate_candidates(plate_candidates_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(plate_candidates_payload, dict):
        return candidates
    for candidate in list(plate_candidates_payload.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("candidate_source") or "") != "frame_scan":
            continue
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item.get("timestamp") or 0.0),
            str(item.get("frame_id") or ""),
            -float(item.get("plate_candidate_score") or 0.0),
        )
    )
    return candidates


def filter_frame_scan_plate_candidates(
    candidates: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    per_frame_counts: dict[str, int] = {}
    skipped_missing_crop = 0
    skipped_low_score = 0
    skipped_limit = 0
    for candidate in candidates:
        plate_crop_path = str(candidate.get("plate_candidate_crop_path") or "")
        if not plate_crop_path or not to_absolute_repo_path(plate_crop_path).exists():
            skipped_missing_crop += 1
            continue
        if str(candidate.get("candidate_status") or "").lower() == "rejected":
            skipped_low_score += 1
            continue
        if float(candidate.get("plate_candidate_score") or 0.0) < float(settings["frame_scan_min_candidate_score"]):
            skipped_low_score += 1
            continue
        frame_id = str(candidate.get("frame_id") or "")
        if per_frame_counts.get(frame_id, 0) >= int(settings["frame_scan_topk_per_frame"]):
            skipped_limit += 1
            continue
        if len(selected) >= int(settings["max_frame_scan_candidates"]):
            skipped_limit += 1
            continue
        per_frame_counts[frame_id] = per_frame_counts.get(frame_id, 0) + 1
        selected.append(candidate)
    return selected, {
        "frame_scan_candidates_available": len(candidates),
        "frame_scan_candidates_after_filter": len(selected),
        "frame_scan_candidates_skipped_missing_crop": skipped_missing_crop,
        "frame_scan_candidates_skipped_low_score": skipped_low_score,
        "frame_scan_candidates_skipped_limit": skipped_limit,
        "frame_scan_candidates_with_matched_track": sum(
            1 for item in selected if item.get("matched_source_track_id")
        ),
        "frame_scan_candidates_without_matched_track": sum(
            1 for item in selected if not item.get("matched_source_track_id")
        ),
    }


def normalize_text(raw_text: str, apply_normalization: bool) -> str:
    text = str(raw_text or "").upper().strip()
    text = re.sub(r"[\s\-_:/\\|.{}[\](),]+", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not apply_normalization:
        return text
    return text


def has_letters_and_digits(text: str) -> bool:
    return any(character.isalpha() for character in text) and any(character.isdigit() for character in text)


def length_score(text: str) -> float:
    length = len(text)
    if 9 <= length <= 10:
        return 1.0
    if length == 8:
        return 0.85
    if 6 <= length <= 7:
        return 0.45
    if 1 <= length <= 5:
        return 0.10
    return 0.0


def format_priority_score(status: str) -> float:
    return {
        "valid_indian_plate": 1.0,
        "possible_indian_plate": 0.75,
        "partial_indian_plate": 0.50,
        "non_plate_text": 0.05,
        "weak_pattern": 0.15,
        "unreadable": 0.0,
        "not_available": 0.0,
    }.get(status, 0.0)


def indian_plate_score(text: str) -> tuple[float, str, str]:
    if not text:
        return 0.0, "unreadable", "empty_text"

    if text.isdigit():
        if re.fullmatch(r"[0-9]{2}0000[A-Z]{2}", text):
            return 0.12, "weak_pattern", "unlikely_bh_numeric_variant"
        if 7 <= len(text) <= 12:
            return 0.0, "non_plate_text", "pure_digit_vehicle_text_without_state_prefix"
        return 0.0, "unreadable", "pure_digits_too_short"

    if re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}", text):
        if text[:2] in INDIAN_STATE_CODES:
            return 1.0, "valid_indian_plate", "state_series_four_digit"
    if re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}", text):
        if text[:2] in INDIAN_STATE_CODES:
            return 0.98, "valid_indian_plate", "state_single_letter_series"
    if re.fullmatch(r"[A-Z]{2}[0-9]{3}[A-Z][0-9]{4}", text):
        if text[:2] in INDIAN_STATE_CODES:
            return 0.78, "possible_indian_plate", "extra_digit_before_single_letter_series"
    if re.fullmatch(r"[A-Z]{2}[0-9]{1}[A-Z]{2}[0-9]{4}", text):
        if text[:2] in INDIAN_STATE_CODES:
            return 0.88, "possible_indian_plate", "short_district_code"
    if re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", text):
        return 0.97, "valid_indian_plate", "bharat_series"
    if len(text) >= 6 and text[:2] in INDIAN_STATE_CODES and has_letters_and_digits(text):
        if 8 <= len(text) <= 10:
            return 0.78, "possible_indian_plate", "state_prefix_mixed_pattern"
        return 0.58, "partial_indian_plate", "state_prefix_partial_pattern"
    if has_letters_and_digits(text) and 6 <= len(text) <= 10:
        return 0.20, "weak_pattern", "mixed_alnum_without_state_prefix"
    return 0.0, "unreadable", "not_plate_like"


def correction_maps() -> tuple[dict[str, str], dict[str, str]]:
    to_digit = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
    to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    return to_digit, to_letter


def apply_pattern(text: str, pattern: str) -> tuple[str, list[str]]:
    to_digit, to_letter = correction_maps()
    chars = list(text)
    notes: list[str] = []
    for index, expected in enumerate(pattern):
        if index >= len(chars):
            break
        current = chars[index]
        if expected == "L" and current.isdigit() and current in to_letter:
            chars[index] = to_letter[current]
            notes.append(f"pos{index + 1}:{current}->{chars[index]}")
        if expected == "D" and current.isalpha() and current in to_digit:
            chars[index] = to_digit[current]
            notes.append(f"pos{index + 1}:{current}->{chars[index]}")
    return "".join(chars), notes


def generate_indian_series_corrections(base_text: str, base_notes: list[str]) -> list[tuple[str, list[str], str | None]]:
    generated: list[tuple[str, list[str], str | None]] = []
    if len(base_text) < 10 or base_text[:2] not in INDIAN_STATE_CODES:
        return generated
    if not base_text[2:4].isdigit():
        return generated

    # Example OCR confusion:
    # HR384E1442 -> HR38AE1442
    # This demo parser is still heuristic; production should use a stronger plate parser/ReID-backed validation.
    if (
        len(base_text) == 10
        and base_text[4].isdigit()
        and base_text[5].isalpha()
        and base_text[6:].isdigit()
        and base_text[4] in INDIAN_SERIES_DIGIT_TO_LETTER
    ):
        corrected_letter = INDIAN_SERIES_DIGIT_TO_LETTER[base_text[4]]
        reconstructed = f"{base_text[:4]}{corrected_letter}{base_text[5:]}"
        generated.append(
            (
                reconstructed,
                base_notes + [f"series_position_digit_to_letter:{base_text[4]}->{corrected_letter}"],
                "standard_two_letter_series_position_correction",
            )
        )

    return generated


def reconstruct_indian_plate(normalized_text: str) -> dict[str, Any]:
    if not normalized_text:
        return {
            "indian_plate_candidate_text": "",
            "corrected_plate_text": "",
            "correction_applied": False,
            "correction_notes": [],
            "indian_plate_score": 0.0,
            "plate_format_status": "unreadable",
            "indian_format_reason": "empty_text",
            "all_corrected_candidates": [],
            "selected_indian_candidate_reason": "empty_text",
            "non_plate_text_detected": False,
            "body_text_possible": False,
            "body_text_reason": None,
        }

    if normalized_text.isdigit():
        score, status, reason = indian_plate_score(normalized_text)
        return {
            "indian_plate_candidate_text": normalized_text,
            "corrected_plate_text": normalized_text,
            "correction_applied": False,
            "correction_notes": [],
            "indian_plate_score": round(score, 3),
            "plate_format_status": status,
            "indian_format_reason": reason,
            "all_corrected_candidates": [normalized_text],
            "selected_indian_candidate_reason": reason,
            "non_plate_text_detected": status == "non_plate_text",
            "body_text_possible": status == "non_plate_text",
            "body_text_reason": reason if status == "non_plate_text" else None,
        }

    candidates: list[tuple[str, list[str], str | None]] = [(normalized_text, [], None)]
    for index in (1, 2):
        if len(normalized_text) > index + 1 and normalized_text[index : index + 2] in INDIAN_STATE_CODES:
            candidates.append((normalized_text[index:], [f"trimmed_prefix_{index}"], None))

    for base_text, base_notes, base_reason in list(candidates):
        if len(base_text) >= 10:
            for pattern in ("LLDDLLDDDD", "LLDDLDDDD", "DDBHDDDDLL"):
                corrected, notes = apply_pattern(base_text, pattern)
                candidates.append((corrected, base_notes + notes, base_reason))
        if len(base_text) >= 9 and len(base_text) <= 10 and base_text[:2] in INDIAN_STATE_CODES:
            district = base_text[2:5]
            if district[:2].isdigit() and district[2].isdigit():
                reconstructed = f"{base_text[:4]}{base_text[5:]}"
                candidates.append(
                    (
                        reconstructed,
                        base_notes + ["removed_extra_digit_before_series_letter"],
                        "single_letter_series_after_extra_digit_cleanup",
                    )
                )
        candidates.extend(generate_indian_series_corrections(base_text, base_notes))

    best_payload = None
    unique_candidate_texts: list[str] = []
    for candidate_text, notes, forced_reason in candidates:
        if candidate_text not in unique_candidate_texts:
            unique_candidate_texts.append(candidate_text)
        score, status, reason = indian_plate_score(candidate_text)
        selected_reason = forced_reason or reason
        if best_payload is None or score > float(best_payload["indian_plate_score"]) or (
            score == float(best_payload["indian_plate_score"]) and len(candidate_text) > len(best_payload["corrected_plate_text"])
        ):
            best_payload = {
                "indian_plate_candidate_text": normalized_text,
                "corrected_plate_text": candidate_text,
                "correction_applied": candidate_text != normalized_text,
                "correction_notes": notes,
                "indian_plate_score": round(score, 3),
                "plate_format_status": status,
                "indian_format_reason": reason,
                "all_corrected_candidates": [],
                "selected_indian_candidate_reason": selected_reason,
                "non_plate_text_detected": status == "non_plate_text",
                "body_text_possible": status == "non_plate_text",
                "body_text_reason": reason if status == "non_plate_text" else None,
            }
    if best_payload is not None:
        best_payload["all_corrected_candidates"] = unique_candidate_texts
        return best_payload
    return {
        "indian_plate_candidate_text": normalized_text,
        "corrected_plate_text": normalized_text,
        "correction_applied": False,
        "correction_notes": [],
        "indian_plate_score": 0.0,
        "plate_format_status": "unreadable",
        "indian_format_reason": "not_plate_like",
        "all_corrected_candidates": unique_candidate_texts,
        "selected_indian_candidate_reason": "not_plate_like",
        "non_plate_text_detected": False,
        "body_text_possible": False,
        "body_text_reason": None,
    }


def crop_quality_score(image: Any | None) -> float:
    if image is None or getattr(image, "size", 0) == 0:
        return 0.0
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(edges.mean()) / 255.0
    aspect_ratio = width / max(1.0, float(height))
    aspect_score = 1.0 if 2.0 <= aspect_ratio <= 6.8 else 0.55 if 1.5 <= aspect_ratio <= 8.0 else 0.15
    return round(min(1.0, edge_density * 0.55 + aspect_score * 0.45), 3)


def build_debug_variants(image: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscale_2x = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    upscale_3x = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    sharpened = cv2.filter2D(
        upscale_2x,
        -1,
        kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(upscale_2x)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted = cv2.bitwise_not(adaptive)
    return {
        "original": image,
        "grayscale": gray,
        "upscale_2x": upscale_2x,
        "upscale_3x": upscale_3x,
        "sharpened": sharpened,
        "clahe": clahe,
        "adaptive_threshold": adaptive,
        "inverted_threshold": inverted,
    }


def write_variant_images(output_dir: Path, variants: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, variant_image in variants.items():
        cv2.imwrite(str(output_dir / f"{variant_name}.jpg"), variant_image)


def run_backend_ocr(
    backend_name: str,
    backend_reader: Any,
    image: Any,
) -> list[tuple[str, float]]:
    if backend_name == "easyocr":
        rows = backend_reader.readtext(image, detail=1, paragraph=False) or []
        return [
            (str(row[1] or "").strip(), round(float(row[2] or 0.0), 3))
            for row in rows
            if isinstance(row, (list, tuple)) and len(row) >= 3
        ]
    rows = backend_reader.ocr(image, cls=False) or []
    parsed: list[tuple[str, float]] = []
    for group in rows:
        for row in group or []:
            text_conf = row[1] if len(row) > 1 else None
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                parsed.append((str(text_conf[0] or "").strip(), round(float(text_conf[1] or 0.0), 3)))
    return parsed


def detect_track_frame_info(
    attribute: dict[str, Any],
    track_lookup: dict[str, dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
    frames_per_track: int,
) -> list[dict[str, Any]]:
    source_track_id = str(attribute.get("source_track_id") or "")
    source_track = track_lookup.get(source_track_id, {})
    bbox_sequence = list(source_track.get("bbox_sequence") or [])
    if not bbox_sequence:
        best_frame_id = str(attribute.get("best_frame_id") or "")
        best_image_path = attribute.get("best_image_path")
        if best_frame_id and best_image_path:
            return [{
                "frame_id": best_frame_id,
                "timestamp": float(attribute.get("start_time") or 0.0),
                "bbox_xyxy": None,
                "image_path": str(best_image_path),
            }]
        return []

    indices = {0, len(bbox_sequence) - 1, len(bbox_sequence) // 2}
    if len(bbox_sequence) > 1:
        step = max(1, len(bbox_sequence) // max(1, frames_per_track - 1))
        indices.update(range(0, len(bbox_sequence), step))
    ordered_indices = sorted(indices)[: max(1, frames_per_track)]

    results: list[dict[str, Any]] = []
    seen_frame_ids: set[str] = set()
    for index in ordered_indices:
        item = bbox_sequence[index]
        frame_id = str(item.get("frame_id") or "")
        if frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)
        frame_item = frame_lookup.get(frame_id, {})
        image_path = frame_item.get("image_path") or source_track.get("best_image_path") or attribute.get("best_image_path")
        results.append(
            {
                "frame_id": frame_id,
                "timestamp": round(float(item.get("timestamp") or 0.0), 3),
                "bbox_xyxy": list(item.get("bbox_xyxy") or [])[:4],
                "image_path": str(image_path) if image_path else None,
            }
        )
    return results


def read_image_with_original_fallback(
    frame_info: dict[str, Any],
    *,
    settings: dict[str, Any],
    video_info_payload: dict[str, Any] | None,
    warnings: list[str],
) -> Any | None:
    if settings["use_original_frame"] and isinstance(video_info_payload, dict):
        video_path = Path(str(video_info_payload.get("video_path") or ""))
        if video_path.exists():
            capture = cv2.VideoCapture(str(video_path))
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_info["timestamp"]) * 1000.0)
                success, frame = capture.read()
                if success and frame is not None:
                    return frame
                warnings.append(
                    f"Original-frame OCR fallback failed at {float(frame_info['timestamp']):.3f}s; using sampled frame."
                )
            finally:
                capture.release()

    image_path = frame_info.get("image_path")
    if not image_path:
        return None
    return cv2.imread(str(to_absolute_repo_path(str(image_path))))


def crop_vehicle_from_frame(frame: Any, bbox_xyxy: list[Any] | None) -> Any | None:
    if frame is None or bbox_xyxy is None or len(bbox_xyxy) < 4:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(float(bbox_xyxy[0])))))
    y1 = max(0, min(height, int(round(float(bbox_xyxy[1])))))
    x2 = max(0, min(width, int(round(float(bbox_xyxy[2])))))
    y2 = max(0, min(height, int(round(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return crop


def build_plate_region_proposals_for_vehicle(
    vehicle_crop: Any,
    plate_candidate_image: Any | None,
) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    if plate_candidate_image is not None and getattr(plate_candidate_image, "size", 0) > 0:
        proposals["plate_candidate_from_step6"] = {
            "image": plate_candidate_image,
            "crop_quality_score": crop_quality_score(plate_candidate_image),
        }

    height, width = vehicle_crop.shape[:2]
    regions = [
        ("vehicle_lower_center_wide", 0.54, 0.86, 0.18, 0.82),
        ("vehicle_lower_center_tight", 0.60, 0.82, 0.28, 0.72),
        ("vehicle_lower_left", 0.58, 0.86, 0.10, 0.48),
        ("vehicle_lower_right", 0.58, 0.86, 0.52, 0.90),
        ("vehicle_mid_center", 0.42, 0.72, 0.25, 0.75),
        ("vehicle_front_like_left", 0.46, 0.78, 0.02, 0.42),
        ("vehicle_front_like_right", 0.46, 0.78, 0.58, 0.98),
    ]
    for name, y1r, y2r, x1r, x2r in regions:
        y1 = int(round(height * y1r))
        y2 = int(round(height * y2r))
        x1 = int(round(width * x1r))
        x2 = int(round(width * x2r))
        if y2 <= y1 or x2 <= x1:
            continue
        crop = vehicle_crop[y1:y2, x1:x2]
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        proposals[name] = {"image": crop, "crop_quality_score": crop_quality_score(crop)}

    proposals["full_vehicle_crop_resized"] = {
        "image": cv2.resize(vehicle_crop, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC),
        "crop_quality_score": crop_quality_score(vehicle_crop),
    }
    return proposals


def evaluate_candidate_status(candidate: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str]:
    corrected_text = str(candidate.get("corrected_plate_text") or "")
    plate_format_status = str(candidate.get("plate_format_status") or "unreadable")
    final_plate_confidence = float(candidate.get("final_plate_confidence") or 0.0)
    ocr_confidence = float(candidate.get("ocr_confidence") or 0.0)
    plate_crop_source = str(candidate.get("plate_crop_source") or "")
    if not corrected_text:
        return "unreadable", "no_text_detected"

    has_state_prefix = corrected_text[:2] in INDIAN_STATE_CODES if len(corrected_text) >= 2 else False
    has_bh = bool(re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", corrected_text))
    if (
        plate_format_status == "valid_indian_plate"
        and final_plate_confidence >= float(settings["strong_confidence"])
        and ocr_confidence >= float(settings["strong_min_ocr_confidence"])
        and 8 <= len(corrected_text) <= 10
        and (has_state_prefix or has_bh)
        and has_letters_and_digits(corrected_text)
        and plate_crop_source == "06A_plate_candidate_detector"
    ):
        return "read_strong", "valid_plate_like_text_high_confidence"
    if plate_format_status == "valid_indian_plate":
        return "read_needs_review", "valid_indian_format_low_ocr_confidence"
    if plate_format_status in {"possible_indian_plate", "partial_indian_plate"}:
        return "read_needs_review", "possible_plate_like_text_needs_review"
    if plate_format_status == "non_plate_text":
        return "read_weak", "pure_digit_vehicle_text_without_state_prefix"
    if plate_format_status == "weak_pattern":
        return "read_weak", "high_ocr_confidence_but_weak_plate_pattern"
    return "unreadable", "no_text_detected"


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    filtered = [item for item in candidates if str(item.get("corrected_plate_text") or "")]
    if not filtered:
        return None
    grouped: dict[str, dict[str, Any]] = {}
    for item in filtered:
        key = str(item.get("corrected_plate_text") or item.get("normalized_text") or "")
        current = grouped.get(key)
        if current is None or float(item["candidate_score"]) > float(current["candidate_score"]):
            grouped[key] = item
    return max(
        grouped.values(),
        key=lambda item: (
            float(item["candidate_score"]),
            float(item["indian_plate_score"]),
            -float(bool(item.get("non_plate_text_detected"))),
            float(item["final_plate_confidence"]),
            len(str(item["corrected_plate_text"])),
        ),
    )


def build_empty_result(
    attribute: dict[str, Any],
    vehicle_attributes: dict[str, Any],
    device_context: dict[str, Any],
    *,
    plate_crop_path: str | None,
    status: str,
    status_reason: str,
) -> dict[str, Any]:
    return {
        "attribute_track_id": attribute.get("attribute_track_id"),
        "source_track_id": attribute.get("source_track_id"),
        "class_name": attribute.get("class_name"),
        "vehicle_type": vehicle_attributes.get("vehicle_type"),
        "vehicle_category": vehicle_attributes.get("vehicle_category"),
        "vehicle_color": vehicle_attributes.get("vehicle_color"),
        "start_time": attribute.get("start_time"),
        "end_time": attribute.get("end_time"),
        "best_frame_id": attribute.get("best_frame_id"),
        "plate_candidate_crop_path": plate_crop_path,
        "plate_crop_source": None,
        "plate_candidate_id": None,
        "plate_candidate_score": None,
        "ocr_backend_used": None,
        "ocr_device_used": device_context["ocr_device_used"],
        "gpu_enabled_for_ocr": bool(device_context["gpu_enabled_for_ocr"]),
        "raw_ocr_text": "",
        "normalized_text": "",
        "candidate_plate_text": "",
        "indian_plate_candidate_text": "",
        "corrected_plate_text": "",
        "correction_applied": False,
        "correction_notes": [],
        "indian_plate_score": 0.0,
        "indian_format_reason": "not_available",
        "ocr_confidence": 0.0,
        "plate_like_score": 0.0,
        "final_plate_confidence": 0.0,
        "plate_format_status": "not_available" if status != "unreadable" else "unreadable",
        "plate_ocr_status": status,
        "status_reason": status_reason,
        "needs_review": True,
        "frames_used_for_ocr": [],
        "best_ocr_frame_id": None,
        "best_ocr_timestamp": None,
        "selected_crop_region": None,
        "selected_crop_path": None,
        "ocr_candidates": [],
        "debug_crop_dir": None,
    }


def build_plate_ocr_outputs(run_dir: Path) -> dict[str, Any]:
    attributes_path = run_dir / "06_track_attributes.json"
    if not attributes_path.exists():
        raise FileNotFoundError(f"Missing required Step 7A input: {attributes_path}")

    warnings: list[str] = []
    settings = read_ocr_settings()
    device_context = build_device_context(warnings, settings)
    attributes_payload = read_json(attributes_path)
    attribute_report_payload = read_optional_json(run_dir / "06_attribute_report.json")
    plate_candidates_payload = read_optional_json(run_dir / "06A_plate_candidates.json")
    tracking_focus_payload = read_optional_json(run_dir / "05_tracking_focus.json")
    video_info_payload = read_optional_json(run_dir / "01_video_info.json")
    frames_index_payload = read_optional_json(run_dir / "03_sampled_frames_index.json")
    _, source_tracks = detect_track_source(run_dir)
    track_lookup = build_track_lookup(source_tracks)
    frame_lookup = build_frame_lookup(frames_index_payload)
    plate_candidate_lookup = build_plate_candidate_lookup(plate_candidates_payload)
    frame_scan_plate_candidates = build_frame_scan_plate_candidates(plate_candidates_payload)
    filtered_frame_scan_candidates, frame_scan_stats = filter_frame_scan_plate_candidates(
        frame_scan_plate_candidates,
        settings=settings,
    )

    attributes = [
        item for item in list(attributes_payload.get("attributes") or [])
        if isinstance(item, dict) and isinstance(item.get("vehicle_attributes"), dict)
    ]
    total_vehicle_tracks = len(attributes)
    if settings["max_tracks"] is not None:
        attributes = attributes[: settings["max_tracks"]]
        warnings.append(
            f"OCR processing limited to the first {settings['max_tracks']} vehicle tracks by {ENV_FINAL_DEMO_OCR_MAX_TRACKS}."
        )

    ocr_backend_used, backend_reader, backend_available = load_ocr_backend(
        settings["ocr_backend_requested"],
        settings,
        device_context,
        warnings,
    )

    debug_root = run_dir / "07A_plate_ocr_crops"
    debug_root.mkdir(parents=True, exist_ok=True)

    if total_vehicle_tracks == 0 and not filtered_frame_scan_candidates:
        overall_status = "skipped_no_vehicle_tracks"
    elif settings["ocr_backend_requested"] == "disabled":
        overall_status = "skipped_disabled"
    elif settings["require_gpu"] and not device_context["cuda_available"]:
        overall_status = "skipped_gpu_unavailable"
    elif not backend_available:
        overall_status = "skipped_backend_missing"
    else:
        overall_status = "completed"

    results: list[dict[str, Any]] = []
    results_by_vehicle_type: dict[str, int] = {}
    results_by_status: dict[str, int] = {}
    tracks_with_plate_candidates = 0
    ocr_attempted_count = 0
    ocr_success_count = 0
    ocr_strong_count = 0
    ocr_needs_review_count = 0
    ocr_weak_count = 0
    valid_indian_plate_count = 0
    possible_indian_plate_count = 0
    partial_indian_plate_count = 0
    non_plate_text_count = 0
    body_text_candidate_count = 0
    weak_pattern_count = 0
    unreadable_count = 0
    crops_generated_count = 0
    downgraded_high_conf_weak_pattern_count = 0
    best_plate_candidates: list[dict[str, Any]] = []
    frame_scan_ocr_results = 0

    for attribute in attributes:
        vehicle_attributes = dict(attribute.get("vehicle_attributes") or {})
        attribute_track_id = str(attribute.get("attribute_track_id") or "")
        vehicle_type = str(vehicle_attributes.get("vehicle_type") or "unknown")
        results_by_vehicle_type[vehicle_type] = results_by_vehicle_type.get(vehicle_type, 0) + 1
        plate_crop_path = vehicle_attributes.get("possible_plate_crop_path")
        absolute_plate_crop_path = (
            to_absolute_repo_path(str(plate_crop_path))
            if plate_crop_path
            else None
        )
        if absolute_plate_crop_path and absolute_plate_crop_path.exists():
            tracks_with_plate_candidates += 1

        if attribute.get("count_for_summary") is False:
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="skipped_noise_track",
                status_reason="noise_track_skipped",
            )
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            continue

        if not absolute_plate_crop_path or not absolute_plate_crop_path.exists():
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="not_available",
                status_reason="no_plate_crop",
            )
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            warnings.append("No plate candidate crop available for this vehicle track.")
            continue

        if overall_status in {"skipped_disabled", "skipped_backend_missing", "skipped_gpu_unavailable"}:
            status_reason = (
                "gpu_required_but_unavailable"
                if overall_status == "skipped_gpu_unavailable"
                else "backend_missing" if overall_status == "skipped_backend_missing" else "ocr_disabled"
            )
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="skipped_backend_missing" if overall_status != "skipped_disabled" else "skipped_disabled",
                status_reason=status_reason,
            )
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            continue

        detected_plate_candidates = list(plate_candidate_lookup.get(attribute_track_id, []))[:3]
        use_legacy_guessed_regions = bool(settings["use_legacy_guessed_regions"])
        plate_crop_source = "06A_plate_candidate_detector" if detected_plate_candidates else None
        if not detected_plate_candidates and use_legacy_guessed_regions and absolute_plate_crop_path and absolute_plate_crop_path.exists():
            plate_crop_source = "step6_legacy_guess"
            warnings.append("OCR used legacy guessed vehicle region; result requires review.")

        if not detected_plate_candidates and not use_legacy_guessed_regions:
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="not_available",
                status_reason="no_reliable_plate_candidate",
            )
            result["plate_crop_source"] = "06A_plate_candidate_detector"
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            warnings.append(
                f"No Step 6A plate candidate available for {attribute_track_id}; OCR skipped because {ENV_FINAL_DEMO_OCR_USE_LEGACY_GUESSED_REGIONS}=0."
            )
            continue

        frame_infos = detect_track_frame_info(
            attribute,
            track_lookup,
            frame_lookup,
            int(settings["frames_per_track"]),
        )
        if not frame_infos:
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="unreadable",
                status_reason="no_text_detected",
            )
            results.append(result)
            unreadable_count += 1
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            continue

        debug_track_dir = debug_root / str(attribute.get("attribute_track_id") or "unknown_track")
        candidate_regions_root = debug_track_dir / "candidate_regions"
        ocr_attempted_count += 1
        all_candidates: list[dict[str, Any]] = []
        frames_used_for_ocr: list[dict[str, Any]] = []
        debug_frames: list[dict[str, Any]] = []

        if detected_plate_candidates:
            for detected_candidate in detected_plate_candidates:
                candidate_crop_path = to_absolute_repo_path(str(detected_candidate["plate_candidate_crop_path"]))
                plate_crop = cv2.imread(str(candidate_crop_path))
                if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                    warnings.append(
                        f"Could not read Step 6A plate candidate crop for {detected_candidate.get('plate_candidate_id')}."
                    )
                    continue
                frame_id = str(detected_candidate.get("frame_id") or "")
                timestamp = round(float(detected_candidate.get("timestamp") or 0.0), 3)
                frames_used_for_ocr.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "image_path": detected_candidate.get("vehicle_crop_path"),
                    }
                )
                region_name = str(detected_candidate.get("plate_candidate_id") or "plate_candidate")
                region_quality = float(detected_candidate.get("plate_candidate_score", 0.5) or 0.5)
                variants = build_debug_variants(plate_crop)
                crops_generated_count += 1
                if settings["save_debug_crops"]:
                    write_variant_images(candidate_regions_root / frame_id / region_name, variants)
                selected_crop_path = (
                    to_repo_relative_path(candidate_regions_root / frame_id / region_name / "original.jpg")
                    if settings["save_debug_crops"]
                    else str(detected_candidate.get("plate_candidate_crop_path") or "")
                )
                frame_debug = {"frame_id": frame_id, "timestamp": timestamp, "regions": []}
                for variant_name, variant_image in variants.items():
                    backend_image = variant_image
                    if len(getattr(backend_image, "shape", ())) == 2:
                        backend_image = cv2.cvtColor(backend_image, cv2.COLOR_GRAY2BGR)
                    try:
                        ocr_rows = run_backend_ocr(str(ocr_backend_used), backend_reader, backend_image)
                    except Exception as exc:
                        warnings.append(f"OCR failed for Step 6A candidate {region_name}: {exc}")
                        continue
                    for raw_text, ocr_confidence in ocr_rows:
                        normalized = normalize_text(raw_text, bool(settings["normalize_text"]))
                        reconstruction = reconstruct_indian_plate(normalized)
                        plate_like_score = float(reconstruction["indian_plate_score"])
                        final_plate_confidence = round((float(ocr_confidence) * 0.70) + (plate_like_score * 0.30), 3)
                        candidate_score = round(
                            (plate_like_score * 0.55)
                            + (format_priority_score(str(reconstruction["plate_format_status"])) * 0.20)
                            + (length_score(str(reconstruction["corrected_plate_text"])) * 0.10)
                            + (float(ocr_confidence) * 0.10)
                            + (region_quality * 0.05),
                            3,
                        )
                        candidate = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "region_name": region_name,
                            "variant_name": variant_name,
                            "raw_text": raw_text,
                            "normalized_text": normalized,
                            "candidate_plate_text": reconstruction["corrected_plate_text"],
                            "selected_crop_path": selected_crop_path,
                            "ocr_confidence": float(ocr_confidence),
                            "plate_like_score": plate_like_score,
                            "final_plate_confidence": final_plate_confidence,
                            "candidate_score": candidate_score,
                            "crop_quality_score": region_quality,
                            "plate_candidate_id": detected_candidate.get("plate_candidate_id"),
                            "plate_candidate_score": detected_candidate.get("plate_candidate_score"),
                            "plate_crop_source": "06A_plate_candidate_detector",
                            **reconstruction,
                        }
                        all_candidates.append(candidate)
                        frame_debug["regions"].append(
                            {
                                "region_name": region_name,
                                "variant_name": variant_name,
                                "raw_text": raw_text,
                                "normalized_text": normalized,
                                "corrected_plate_text": reconstruction["corrected_plate_text"],
                                "candidate_score": candidate_score,
                                "indian_plate_score": plate_like_score,
                            }
                        )
                debug_frames.append(frame_debug)
        else:
            step6_plate_crop = cv2.imread(str(absolute_plate_crop_path))
            for frame_info in frame_infos:
                source_frame = read_image_with_original_fallback(
                    frame_info,
                    settings=settings,
                    video_info_payload=video_info_payload,
                    warnings=warnings,
                )
                vehicle_crop = crop_vehicle_from_frame(source_frame, list(frame_info.get("bbox_xyxy") or []))
                if vehicle_crop is None:
                    continue
                frames_used_for_ocr.append(
                    {
                        "frame_id": frame_info["frame_id"],
                        "timestamp": frame_info["timestamp"],
                        "image_path": frame_info.get("image_path"),
                    }
                )
                region_proposals = build_plate_region_proposals_for_vehicle(vehicle_crop, step6_plate_crop)
                frame_debug = {"frame_id": frame_info["frame_id"], "timestamp": frame_info["timestamp"], "regions": []}
                for region_name, region_payload in region_proposals.items():
                    region_image = region_payload["image"]
                    region_quality = float(region_payload["crop_quality_score"])
                    crops_generated_count += 1
                    variants = build_debug_variants(region_image)
                    if settings["save_debug_crops"]:
                        write_variant_images(candidate_regions_root / str(frame_info["frame_id"]) / region_name, variants)
                    selected_crop_path = to_repo_relative_path(
                        candidate_regions_root / str(frame_info["frame_id"]) / region_name / "original.jpg"
                    ) if settings["save_debug_crops"] else None
                    for variant_name, variant_image in variants.items():
                        backend_image = variant_image
                        if len(getattr(backend_image, "shape", ())) == 2:
                            backend_image = cv2.cvtColor(backend_image, cv2.COLOR_GRAY2BGR)
                        try:
                            ocr_rows = run_backend_ocr(str(ocr_backend_used), backend_reader, backend_image)
                        except Exception as exc:
                            warnings.append(f"OCR failed for frame {frame_info['frame_id']} region {region_name}: {exc}")
                            continue
                        for raw_text, ocr_confidence in ocr_rows:
                            normalized = normalize_text(raw_text, bool(settings["normalize_text"]))
                            reconstruction = reconstruct_indian_plate(normalized)
                            plate_like_score = float(reconstruction["indian_plate_score"])
                            final_plate_confidence = round((float(ocr_confidence) * 0.70) + (plate_like_score * 0.30), 3)
                            candidate_score = round(
                                (plate_like_score * 0.55)
                                + (format_priority_score(str(reconstruction["plate_format_status"])) * 0.20)
                                + (length_score(str(reconstruction["corrected_plate_text"])) * 0.10)
                                + (float(ocr_confidence) * 0.10)
                                + (region_quality * 0.05),
                                3,
                            )
                            candidate = {
                                "frame_id": frame_info["frame_id"],
                                "timestamp": frame_info["timestamp"],
                                "region_name": region_name,
                                "variant_name": variant_name,
                                "raw_text": raw_text,
                                "normalized_text": normalized,
                                "candidate_plate_text": reconstruction["corrected_plate_text"],
                                "selected_crop_path": selected_crop_path,
                                "ocr_confidence": float(ocr_confidence),
                                "plate_like_score": plate_like_score,
                                "final_plate_confidence": final_plate_confidence,
                                "candidate_score": candidate_score,
                                "crop_quality_score": region_quality,
                                "plate_candidate_id": None,
                                "plate_candidate_score": None,
                                "plate_crop_source": plate_crop_source or "step7A_legacy_guessed_region",
                                **reconstruction,
                            }
                            all_candidates.append(candidate)
                            frame_debug["regions"].append(
                                {
                                    "region_name": region_name,
                                    "variant_name": variant_name,
                                    "raw_text": raw_text,
                                    "normalized_text": normalized,
                                    "corrected_plate_text": reconstruction["corrected_plate_text"],
                                    "candidate_score": candidate_score,
                                    "indian_plate_score": plate_like_score,
                                }
                            )
                debug_frames.append(frame_debug)

        best_candidate = select_best_candidate(all_candidates)
        if best_candidate is None:
            unreadable_count += 1
            result = build_empty_result(
                attribute,
                vehicle_attributes,
                device_context,
                plate_crop_path=plate_crop_path,
                status="unreadable",
                status_reason="no_text_detected",
            )
            result["frames_used_for_ocr"] = frames_used_for_ocr
            result["ocr_candidates"] = all_candidates
            result["debug_crop_dir"] = to_repo_relative_path(debug_track_dir) if settings["save_debug_crops"] else None
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            if settings["save_debug_crops"]:
                write_json(debug_track_dir / "ocr_debug.json", {
                    "frames_used": frames_used_for_ocr,
                    "ocr_candidates": all_candidates,
                    "selected_candidate": None,
                    "rejected_candidates": all_candidates,
                    "gpu_device_info": {
                        "ocr_device_used": device_context["ocr_device_used"],
                        "gpu_enabled_for_ocr": device_context["gpu_enabled_for_ocr"],
                    },
                })
            continue

        status, status_reason = evaluate_candidate_status(best_candidate, settings)
        if best_candidate.get("plate_crop_source") in {"step6_legacy_guess", "step7A_legacy_guessed_region"}:
            warnings.append("OCR used legacy guessed vehicle region; result requires review.")
            if status == "read_strong":
                status = "read_needs_review"
                status_reason = "legacy_guessed_region_requires_review"
        if best_candidate["plate_format_status"] == "valid_indian_plate":
            valid_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "possible_indian_plate":
            possible_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "partial_indian_plate":
            partial_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "non_plate_text":
            non_plate_text_count += 1
        elif best_candidate["plate_format_status"] == "weak_pattern":
            weak_pattern_count += 1
        else:
            unreadable_count += 1
        if bool(best_candidate.get("body_text_possible")):
            body_text_candidate_count += 1
        if status == "read_strong":
            ocr_success_count += 1
            ocr_strong_count += 1
        elif status == "read_needs_review":
            ocr_success_count += 1
            ocr_needs_review_count += 1
        elif status == "read_weak":
            ocr_weak_count += 1
        if status_reason == "high_ocr_confidence_but_weak_plate_pattern":
            downgraded_high_conf_weak_pattern_count += 1

        result = {
            "candidate_source": best_candidate.get("candidate_source", "track_based"),
            "attribute_track_id": attribute.get("attribute_track_id"),
            "source_track_id": attribute.get("source_track_id"),
            "source_detection_id": best_candidate.get("source_detection_id"),
            "source_detection_class_name": best_candidate.get("source_detection_class_name"),
            "matched_source_track_id": best_candidate.get("matched_source_track_id"),
            "matched_attribute_track_id": best_candidate.get("matched_attribute_track_id"),
            "matched_track_class_name": best_candidate.get("matched_track_class_name"),
            "matched_track_vehicle_type": best_candidate.get("matched_track_vehicle_type"),
            "matched_track_iou": best_candidate.get("matched_track_iou"),
            "matched_track_time_delta": best_candidate.get("matched_track_time_delta"),
            "class_source": best_candidate.get("class_source"),
            "class_name": attribute.get("class_name"),
            "vehicle_type": vehicle_attributes.get("vehicle_type"),
            "vehicle_category": vehicle_attributes.get("vehicle_category"),
            "vehicle_color": vehicle_attributes.get("vehicle_color"),
            "start_time": attribute.get("start_time"),
            "end_time": attribute.get("end_time"),
            "best_frame_id": attribute.get("best_frame_id"),
            "plate_candidate_crop_path": plate_crop_path,
            "plate_crop_source": best_candidate.get("plate_crop_source", plate_crop_source),
            "plate_candidate_id": best_candidate.get("plate_candidate_id"),
            "plate_candidate_score": best_candidate.get("plate_candidate_score"),
            "ocr_backend_used": ocr_backend_used,
            "ocr_device_used": device_context["ocr_device_used"],
            "gpu_enabled_for_ocr": bool(device_context["gpu_enabled_for_ocr"]),
            "raw_ocr_text": best_candidate["raw_text"],
            "normalized_text": best_candidate["normalized_text"],
            "candidate_plate_text": best_candidate["corrected_plate_text"],
            "indian_plate_candidate_text": best_candidate["indian_plate_candidate_text"],
            "corrected_plate_text": best_candidate["corrected_plate_text"],
            "correction_applied": bool(best_candidate["correction_applied"]),
            "correction_notes": list(best_candidate["correction_notes"]),
            "all_corrected_candidates": list(best_candidate.get("all_corrected_candidates") or []),
            "selected_indian_candidate_reason": best_candidate.get("selected_indian_candidate_reason"),
            "non_plate_text_detected": bool(best_candidate.get("non_plate_text_detected")),
            "body_text_possible": bool(best_candidate.get("body_text_possible")),
            "body_text_reason": best_candidate.get("body_text_reason"),
            "indian_plate_score": best_candidate["indian_plate_score"],
            "indian_format_reason": best_candidate["indian_format_reason"],
            "ocr_confidence": best_candidate["ocr_confidence"],
            "plate_like_score": best_candidate["indian_plate_score"],
            "final_plate_confidence": best_candidate["final_plate_confidence"],
            "plate_format_status": best_candidate["plate_format_status"],
            "plate_ocr_status": status,
            "status_reason": status_reason,
            "needs_review": status != "read_strong",
            "frames_used_for_ocr": frames_used_for_ocr,
            "best_ocr_frame_id": best_candidate["frame_id"],
            "best_ocr_timestamp": best_candidate["timestamp"],
            "selected_crop_region": best_candidate["region_name"],
            "selected_crop_path": best_candidate["selected_crop_path"],
            "ocr_candidates": all_candidates,
            "debug_crop_dir": to_repo_relative_path(debug_track_dir) if settings["save_debug_crops"] else None,
        }
        results.append(result)
        results_by_status[status] = results_by_status.get(status, 0) + 1
        best_plate_candidates.append(
            {
                "attribute_track_id": result["attribute_track_id"],
                "candidate_plate_text": result["candidate_plate_text"],
                "plate_format_status": result["plate_format_status"],
                "final_plate_confidence": result["final_plate_confidence"],
            }
        )

        write_json(debug_track_dir / "ocr_debug.json", {
            "frames_used": frames_used_for_ocr,
            "regions_generated": debug_frames,
            "ocr_candidates": all_candidates,
            "selected_candidate": best_candidate,
            "rejected_candidates": [item for item in all_candidates if item is not best_candidate],
            "gpu_device_info": {
                "ocr_device_requested": device_context["ocr_device_requested"],
                "ocr_device_used": device_context["ocr_device_used"],
                "gpu_enabled_for_ocr": device_context["gpu_enabled_for_ocr"],
                "cuda_available": device_context["cuda_available"],
                "cuda_device_name": device_context["cuda_device_name"],
            },
        })

    for candidate_item in filtered_frame_scan_candidates:
        plate_crop_path = str(candidate_item.get("plate_candidate_crop_path") or "")
        if not plate_crop_path:
            continue
        if overall_status in {"skipped_disabled", "skipped_backend_missing", "skipped_gpu_unavailable"}:
            continue
        absolute_plate_crop_path = to_absolute_repo_path(plate_crop_path)
        if not absolute_plate_crop_path.exists():
            warnings.append(
                f"Could not read frame-scan plate candidate crop: {plate_crop_path}"
            )
            continue

        plate_crop = cv2.imread(str(absolute_plate_crop_path))
        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
            warnings.append(
                f"Frame-scan plate crop image is empty or unreadable: {plate_crop_path}"
            )
            continue

        frame_id = str(candidate_item.get("frame_id") or "")
        timestamp = round(float(candidate_item.get("timestamp") or 0.0), 3)
        debug_track_dir = debug_root / "frame_scan" / (frame_id or "unknown_frame")
        candidate_regions_root = debug_track_dir / "candidate_regions"
        ocr_attempted_count += 1
        frame_scan_ocr_results += 1
        frames_used_for_ocr = [{
            "frame_id": frame_id,
            "timestamp": timestamp,
            "image_path": candidate_item.get("vehicle_crop_path"),
        }]
        all_candidates: list[dict[str, Any]] = []
        debug_frames: list[dict[str, Any]] = []
        region_name = str(candidate_item.get("plate_candidate_id") or "frame_scan_plate_candidate")
        region_quality = float(candidate_item.get("plate_candidate_score", 0.5) or 0.5)
        variants = build_debug_variants(plate_crop)
        crops_generated_count += 1
        if settings["save_debug_crops"]:
            write_variant_images(candidate_regions_root / frame_id / region_name, variants)
        selected_crop_path = (
            to_repo_relative_path(candidate_regions_root / frame_id / region_name / "original.jpg")
            if settings["save_debug_crops"]
            else plate_crop_path
        )
        frame_debug = {"frame_id": frame_id, "timestamp": timestamp, "regions": []}
        for variant_name, variant_image in variants.items():
            backend_image = variant_image
            if len(getattr(backend_image, "shape", ())) == 2:
                backend_image = cv2.cvtColor(backend_image, cv2.COLOR_GRAY2BGR)
            try:
                ocr_rows = run_backend_ocr(str(ocr_backend_used), backend_reader, backend_image)
            except Exception as exc:
                warnings.append(f"OCR failed for frame-scan plate candidate {region_name}: {exc}")
                continue
            for raw_text, ocr_confidence in ocr_rows:
                normalized = normalize_text(raw_text, bool(settings["normalize_text"]))
                reconstruction = reconstruct_indian_plate(normalized)
                plate_like_score = float(reconstruction["indian_plate_score"])
                final_plate_confidence = round((float(ocr_confidence) * 0.70) + (plate_like_score * 0.30), 3)
                candidate_score = round(
                    (plate_like_score * 0.55)
                    + (format_priority_score(str(reconstruction["plate_format_status"])) * 0.20)
                    + (length_score(str(reconstruction["corrected_plate_text"])) * 0.10)
                    + (float(ocr_confidence) * 0.10)
                    + (region_quality * 0.05),
                    3,
                )
                candidate = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "region_name": region_name,
                    "variant_name": variant_name,
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                    "candidate_plate_text": reconstruction["corrected_plate_text"],
                    "selected_crop_path": selected_crop_path,
                    "ocr_confidence": float(ocr_confidence),
                    "plate_like_score": plate_like_score,
                    "final_plate_confidence": final_plate_confidence,
                    "candidate_score": candidate_score,
                    "crop_quality_score": region_quality,
                    "plate_candidate_id": candidate_item.get("plate_candidate_id"),
                    "plate_candidate_score": candidate_item.get("plate_candidate_score"),
                    "plate_crop_source": "06A_plate_candidate_detector",
                    "candidate_source": "frame_scan",
                    "source_detection_id": candidate_item.get("source_detection_id"),
                    **reconstruction,
                }
                all_candidates.append(candidate)
                frame_debug["regions"].append(
                    {
                        "region_name": region_name,
                        "variant_name": variant_name,
                        "raw_text": raw_text,
                        "normalized_text": normalized,
                        "corrected_plate_text": reconstruction["corrected_plate_text"],
                        "candidate_score": candidate_score,
                        "indian_plate_score": plate_like_score,
                    }
                )
        debug_frames.append(frame_debug)

        best_candidate = select_best_candidate(all_candidates)
        if best_candidate is None:
            unreadable_count += 1
            result = {
                "candidate_source": "frame_scan",
                "attribute_track_id": candidate_item.get("matched_attribute_track_id"),
                "source_track_id": candidate_item.get("matched_source_track_id"),
                "source_detection_id": candidate_item.get("source_detection_id"),
                "source_detection_class_name": candidate_item.get("source_detection_class_name"),
                "matched_source_track_id": candidate_item.get("matched_source_track_id"),
                "matched_attribute_track_id": candidate_item.get("matched_attribute_track_id"),
                "matched_track_class_name": candidate_item.get("matched_track_class_name"),
                "matched_track_vehicle_type": candidate_item.get("matched_track_vehicle_type"),
                "matched_track_iou": candidate_item.get("matched_track_iou"),
                "matched_track_time_delta": candidate_item.get("matched_track_time_delta"),
                "class_source": candidate_item.get("class_source"),
                "class_name": candidate_item.get("class_name"),
                "vehicle_type": candidate_item.get("vehicle_type"),
                "vehicle_category": None,
                "vehicle_color": candidate_item.get("vehicle_color"),
                "start_time": timestamp,
                "end_time": timestamp,
                "best_frame_id": frame_id,
                "plate_candidate_crop_path": plate_crop_path,
                "plate_crop_source": "06A_plate_candidate_detector",
                "plate_candidate_id": candidate_item.get("plate_candidate_id"),
                "plate_candidate_score": candidate_item.get("plate_candidate_score"),
                "ocr_backend_used": ocr_backend_used,
                "ocr_device_used": device_context["ocr_device_used"],
                "gpu_enabled_for_ocr": bool(device_context["gpu_enabled_for_ocr"]),
                "raw_ocr_text": "",
                "normalized_text": "",
                "candidate_plate_text": "",
                "indian_plate_candidate_text": "",
                "corrected_plate_text": "",
                "correction_applied": False,
                "correction_notes": [],
                "all_corrected_candidates": [],
                "selected_indian_candidate_reason": None,
                "non_plate_text_detected": False,
                "body_text_possible": False,
                "body_text_reason": None,
                "indian_plate_score": 0.0,
                "indian_format_reason": "not_available",
                "ocr_confidence": 0.0,
                "plate_like_score": 0.0,
                "final_plate_confidence": 0.0,
                "plate_format_status": "unreadable",
                "plate_ocr_status": "unreadable",
                "status_reason": "no_text_detected",
                "needs_review": True,
                "frames_used_for_ocr": frames_used_for_ocr,
                "best_ocr_frame_id": frame_id,
                "best_ocr_timestamp": timestamp,
                "selected_crop_region": None,
                "selected_crop_path": selected_crop_path,
                "ocr_candidates": all_candidates,
                "debug_crop_dir": to_repo_relative_path(debug_track_dir) if settings["save_debug_crops"] else None,
            }
            results.append(result)
            results_by_status[result["plate_ocr_status"]] = results_by_status.get(result["plate_ocr_status"], 0) + 1
            if settings["save_debug_crops"]:
                write_json(debug_track_dir / "ocr_debug.json", {
                    "frames_used": frames_used_for_ocr,
                    "regions_generated": debug_frames,
                    "ocr_candidates": all_candidates,
                    "selected_candidate": None,
                    "rejected_candidates": all_candidates,
                })
            continue

        status, status_reason = evaluate_candidate_status(best_candidate, settings)
        if best_candidate["plate_format_status"] == "valid_indian_plate":
            valid_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "possible_indian_plate":
            possible_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "partial_indian_plate":
            partial_indian_plate_count += 1
        elif best_candidate["plate_format_status"] == "non_plate_text":
            non_plate_text_count += 1
        elif best_candidate["plate_format_status"] == "weak_pattern":
            weak_pattern_count += 1
        else:
            unreadable_count += 1
        if bool(best_candidate.get("body_text_possible")):
            body_text_candidate_count += 1
        if status == "read_strong":
            ocr_success_count += 1
            ocr_strong_count += 1
        elif status == "read_needs_review":
            ocr_success_count += 1
            ocr_needs_review_count += 1
        elif status == "read_weak":
            ocr_weak_count += 1
        if status_reason == "high_ocr_confidence_but_weak_plate_pattern":
            downgraded_high_conf_weak_pattern_count += 1

        result = {
            "candidate_source": "frame_scan",
            "attribute_track_id": candidate_item.get("matched_attribute_track_id"),
            "source_track_id": candidate_item.get("matched_source_track_id"),
            "source_detection_id": candidate_item.get("source_detection_id"),
            "source_detection_class_name": candidate_item.get("source_detection_class_name"),
            "matched_source_track_id": candidate_item.get("matched_source_track_id"),
            "matched_attribute_track_id": candidate_item.get("matched_attribute_track_id"),
            "matched_track_class_name": candidate_item.get("matched_track_class_name"),
            "matched_track_vehicle_type": candidate_item.get("matched_track_vehicle_type"),
            "matched_track_iou": candidate_item.get("matched_track_iou"),
            "matched_track_time_delta": candidate_item.get("matched_track_time_delta"),
            "class_source": candidate_item.get("class_source"),
            "class_name": candidate_item.get("class_name"),
            "vehicle_type": candidate_item.get("vehicle_type"),
            "vehicle_category": None,
            "vehicle_color": candidate_item.get("vehicle_color"),
            "start_time": timestamp,
            "end_time": timestamp,
            "best_frame_id": frame_id,
            "plate_candidate_crop_path": plate_crop_path,
            "plate_crop_source": best_candidate.get("plate_crop_source", "06A_plate_candidate_detector"),
            "plate_candidate_id": best_candidate.get("plate_candidate_id"),
            "plate_candidate_score": best_candidate.get("plate_candidate_score"),
            "ocr_backend_used": ocr_backend_used,
            "ocr_device_used": device_context["ocr_device_used"],
            "gpu_enabled_for_ocr": bool(device_context["gpu_enabled_for_ocr"]),
            "raw_ocr_text": best_candidate["raw_text"],
            "normalized_text": best_candidate["normalized_text"],
            "candidate_plate_text": best_candidate["corrected_plate_text"],
            "indian_plate_candidate_text": best_candidate["indian_plate_candidate_text"],
            "corrected_plate_text": best_candidate["corrected_plate_text"],
            "correction_applied": bool(best_candidate["correction_applied"]),
            "correction_notes": list(best_candidate["correction_notes"]),
            "all_corrected_candidates": list(best_candidate.get("all_corrected_candidates") or []),
            "selected_indian_candidate_reason": best_candidate.get("selected_indian_candidate_reason"),
            "non_plate_text_detected": bool(best_candidate.get("non_plate_text_detected")),
            "body_text_possible": bool(best_candidate.get("body_text_possible")),
            "body_text_reason": best_candidate.get("body_text_reason"),
            "indian_plate_score": best_candidate["indian_plate_score"],
            "indian_format_reason": best_candidate["indian_format_reason"],
            "ocr_confidence": best_candidate["ocr_confidence"],
            "plate_like_score": best_candidate["indian_plate_score"],
            "final_plate_confidence": best_candidate["final_plate_confidence"],
            "plate_format_status": best_candidate["plate_format_status"],
            "plate_ocr_status": status,
            "status_reason": status_reason,
            "needs_review": status != "read_strong",
            "frames_used_for_ocr": frames_used_for_ocr,
            "best_ocr_frame_id": best_candidate["frame_id"],
            "best_ocr_timestamp": best_candidate["timestamp"],
            "selected_crop_region": best_candidate["region_name"],
            "selected_crop_path": best_candidate["selected_crop_path"],
            "ocr_candidates": all_candidates,
            "debug_crop_dir": to_repo_relative_path(debug_track_dir) if settings["save_debug_crops"] else None,
        }
        results.append(result)
        results_by_status[status] = results_by_status.get(status, 0) + 1
        best_plate_candidates.append(
            {
                "attribute_track_id": None,
                "candidate_plate_text": result["candidate_plate_text"],
                "plate_format_status": result["plate_format_status"],
                "final_plate_confidence": result["final_plate_confidence"],
            }
        )
        write_json(debug_track_dir / "ocr_debug.json", {
            "frames_used": frames_used_for_ocr,
            "regions_generated": debug_frames,
            "ocr_candidates": all_candidates,
            "selected_candidate": best_candidate,
            "rejected_candidates": [item for item in all_candidates if item is not best_candidate],
        })

    recommendations: list[str] = []
    weakish_count = weak_pattern_count + unreadable_count
    if weakish_count >= max(1, len(results) // 2):
        recommendations.extend(
            [
                "Plate crops are likely missing or too small. Add a dedicated licence plate detector before OCR.",
                "Use original-resolution frames for OCR.",
                "Increase FINAL_DEMO_OCR_FRAMES_PER_TRACK.",
                "Verify EasyOCR GPU is enabled.",
            ]
        )
    if downgraded_high_conf_weak_pattern_count > 0:
        recommendations.append(
            "Some high-confidence OCR strings were downgraded because they were too short or not plate-like."
        )
    if non_plate_text_count > 0:
        recommendations.append(
            "Some detected plate-like regions contained vehicle body text or fleet/phone numbers. These were downgraded."
        )
    if overall_status == "skipped_backend_missing":
        recommendations.append("Install paddleocr or easyocr to enable OCR.")

    results_payload = {
        "created_at": current_timestamp(),
        "ocr_backend_requested": settings["ocr_backend_requested"],
        "ocr_backend_used": ocr_backend_used,
        "ocr_device_requested": device_context["ocr_device_requested"],
        "ocr_device_used": device_context["ocr_device_used"],
        "cuda_available": device_context["cuda_available"],
        "cuda_device_name": device_context["cuda_device_name"],
        "gpu_enabled_for_ocr": device_context["gpu_enabled_for_ocr"],
        "ocr_gpu_id": device_context["ocr_gpu_id"],
        "ocr_backend_init_status": device_context["ocr_backend_init_status"],
        "ocr_backend_init_error": device_context["ocr_backend_init_error"],
        "selected_track_source": "06_track_attributes",
        "total_vehicle_tracks": total_vehicle_tracks,
        "tracks_with_plate_candidates": tracks_with_plate_candidates,
        "frame_scan_candidates_available": frame_scan_stats["frame_scan_candidates_available"],
        "frame_scan_candidates_after_filter": frame_scan_stats["frame_scan_candidates_after_filter"],
        "frame_scan_candidates_input": len(filtered_frame_scan_candidates),
        "frame_scan_ocr_results": frame_scan_ocr_results,
        "frame_scan_candidates_skipped_missing_crop": frame_scan_stats["frame_scan_candidates_skipped_missing_crop"],
        "frame_scan_candidates_skipped_low_score": frame_scan_stats["frame_scan_candidates_skipped_low_score"],
        "frame_scan_candidates_skipped_limit": frame_scan_stats["frame_scan_candidates_skipped_limit"],
        "frame_scan_candidates_with_matched_track": frame_scan_stats["frame_scan_candidates_with_matched_track"],
        "frame_scan_candidates_without_matched_track": frame_scan_stats["frame_scan_candidates_without_matched_track"],
        "ocr_attempted_count": ocr_attempted_count,
        "ocr_success_count": ocr_success_count,
        "ocr_strong_count": ocr_strong_count,
        "ocr_needs_review_count": ocr_needs_review_count,
        "ocr_weak_count": ocr_weak_count,
        "ocr_unreadable_count": unreadable_count,
        "results": results,
        "warnings": list(dict.fromkeys(warnings)),
    }

    report_payload = {
        "created_at": current_timestamp(),
        "ocr_backend_requested": settings["ocr_backend_requested"],
        "ocr_backend_used": ocr_backend_used,
        "ocr_device_requested": device_context["ocr_device_requested"],
        "ocr_device_used": device_context["ocr_device_used"],
        "cuda_available": device_context["cuda_available"],
        "cuda_device_name": device_context["cuda_device_name"],
        "gpu_enabled_for_ocr": device_context["gpu_enabled_for_ocr"],
        "ocr_gpu_id": device_context["ocr_gpu_id"],
        "ocr_backend_init_status": device_context["ocr_backend_init_status"],
        "ocr_backend_init_error": device_context["ocr_backend_init_error"],
        "overall_status": overall_status,
        "total_vehicle_tracks": total_vehicle_tracks,
        "tracks_with_plate_candidates": tracks_with_plate_candidates,
        "frame_scan_candidates_available": frame_scan_stats["frame_scan_candidates_available"],
        "frame_scan_candidates_after_filter": frame_scan_stats["frame_scan_candidates_after_filter"],
        "frame_scan_candidates_input": len(filtered_frame_scan_candidates),
        "frame_scan_ocr_results": frame_scan_ocr_results,
        "frame_scan_candidates_skipped_missing_crop": frame_scan_stats["frame_scan_candidates_skipped_missing_crop"],
        "frame_scan_candidates_skipped_low_score": frame_scan_stats["frame_scan_candidates_skipped_low_score"],
        "frame_scan_candidates_skipped_limit": frame_scan_stats["frame_scan_candidates_skipped_limit"],
        "frame_scan_candidates_with_matched_track": frame_scan_stats["frame_scan_candidates_with_matched_track"],
        "frame_scan_candidates_without_matched_track": frame_scan_stats["frame_scan_candidates_without_matched_track"],
        "ocr_attempted_count": ocr_attempted_count,
        "ocr_success_count": ocr_success_count,
        "ocr_strong_count": ocr_strong_count,
        "ocr_needs_review_count": ocr_needs_review_count,
        "ocr_weak_count": ocr_weak_count,
        "ocr_unreadable_count": unreadable_count,
        "valid_indian_plate_count": valid_indian_plate_count,
        "possible_indian_plate_count": possible_indian_plate_count,
        "partial_indian_plate_count": partial_indian_plate_count,
        "non_plate_text_count": non_plate_text_count,
        "weak_pattern_count": weak_pattern_count,
        "unreadable_count": unreadable_count,
        "body_text_candidate_count": body_text_candidate_count,
        "indian_parser_version": "v2_single_letter_series",
        "best_plate_candidates": best_plate_candidates[:10],
        "candidate_regions_per_vehicle": 8,
        "frames_per_track": int(settings["frames_per_track"]),
        "crops_generated_count": crops_generated_count,
        "false_strong_guard_enabled": True,
        "strong_read_gate": {
            "min_confidence": float(settings["strong_confidence"]),
            "allowed_format_status": ["valid_indian_plate"],
            "min_text_length": int(settings["min_strong_text_length"]),
            "requires_letter_and_digit": True,
        },
        "downgraded_high_conf_weak_pattern_count": downgraded_high_conf_weak_pattern_count,
        "results_by_vehicle_type": dict(sorted(results_by_vehicle_type.items())),
        "results_by_status": dict(sorted(results_by_status.items())),
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recommendations)),
        "selected_track_source": "06_track_attributes",
        "tracking_focus_profile": (
            tracking_focus_payload.get("selected_focus_profile")
            if isinstance(tracking_focus_payload, dict)
            else None
        ),
        "attribute_report_summary": {
            "total_attributes_created": (
                attribute_report_payload.get("total_attributes_created")
                if isinstance(attribute_report_payload, dict)
                else None
            ),
            "vehicle_attribute_count": (
                attribute_report_payload.get("vehicle_attribute_count")
                if isinstance(attribute_report_payload, dict)
                else None
            ),
            "plate_candidate_crop_count": (
                attribute_report_payload.get("plate_candidate_crop_count")
                if isinstance(attribute_report_payload, dict)
                else None
            ),
        },
        "video_path": (
            video_info_payload.get("video_path") if isinstance(video_info_payload, dict) else None
        ),
    }

    return {
        "results_payload": results_payload,
        "report_payload": report_payload,
        "debug_root": debug_root,
    }


def update_run_manifest_for_plate_ocr(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "07A_plate_ocr" not in completed_steps:
        completed_steps.append("07A_plate_ocr")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "07B_event_candidate_generation"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
