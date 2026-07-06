from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

ALLOWED_CLASS_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
}
PERSON_CLASS_NAME = "person"
BAG_CLASS_NAMES = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASS_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck"}
LICENSE_PLATE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}
PERSON_CLASS_FILTER_IDS = [0]
VALID_INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ",
    "HR", "HP", "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ",
    "NL", "OR", "PY", "PB", "RJ", "SK", "TN", "TR", "UP", "WB", "TS",
    "UK", "LA", "CG", "JH",
}
_PLATE_DETECTOR_MODEL: Any | None = None
_PLATE_DETECTOR_ATTEMPTED = False
_FLORENCE_STATE: dict[str, Any] | None = None


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


@dataclass
class FrameDetection:
    frame_id: str
    video_id: str
    timestamp_seconds: float
    frame_width: int
    frame_height: int
    detections: list[Detection]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_LOCAL_PERSON_MODEL = _repo_root() / "Person_detection" / "Person_detection.pt"
DEFAULT_LOCAL_VEHICLE_MODEL = _repo_root() / "object_yolo" / "best_old.pt"
DEFAULT_LOCAL_OCR_MUKUL_DIR = _repo_root() / "object_yolo" / "OCR_MUKUL"
LEGACY_OCR_MUKUL_DIR = Path(r"C:\Mukul K\OCR_MUKUL")
OCR_MUKUL_BASE_MODEL_ID = "microsoft/Florence-2-base-ft"


def _resolve_ocr_mukul_dir() -> Path:
    if DEFAULT_LOCAL_OCR_MUKUL_DIR.exists() and DEFAULT_LOCAL_OCR_MUKUL_DIR.is_dir():
        return DEFAULT_LOCAL_OCR_MUKUL_DIR
    return LEGACY_OCR_MUKUL_DIR


OCR_MUKUL_DIR = _resolve_ocr_mukul_dir()
OCR_MUKUL_PLATE_MODEL_PATH = OCR_MUKUL_DIR / "license_plate_weights.pt"
OCR_MUKUL_FLORENCE_ADAPTER_PATH = OCR_MUKUL_DIR / "adaptor_florance_baseFT"


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))


def _load_benchmark_tracker():
    try:
        from tests.manual_benchmark_case.benchmark_bot_sort_tracker import BenchmarkBoTSORTTracker as tracker_class
        return tracker_class
    except ModuleNotFoundError:
        tracker_path = _repo_root() / "tests" / "manual_benchmark_case" / "benchmark_bot_sort_tracker.py"
        spec = importlib.util.spec_from_file_location("benchmark_bot_sort_tracker", tracker_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load benchmark tracker from: {tracker_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BenchmarkBoTSORTTracker


BenchmarkBoTSORTTracker = _load_benchmark_tracker()


def _debug_root() -> Path:
    return Path(__file__).resolve().parent / "debug_runs"


def _format_seconds(value: float) -> str:
    total = max(0.0, float(value))
    minutes = int(total // 60)
    seconds = total - (minutes * 60)
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{minutes:02d}:{int(round(seconds)):02d}"
    return f"{minutes:02d}:{seconds:04.1f}"


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except Exception:
        return str(path)


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
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception:
        _FLORENCE_STATE = state
        return state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device

    try:
        processor_source = str(OCR_MUKUL_FLORENCE_ADAPTER_PATH) if OCR_MUKUL_FLORENCE_ADAPTER_PATH.exists() else OCR_MUKUL_BASE_MODEL_ID
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, local_files_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            OCR_MUKUL_BASE_MODEL_ID,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=False,
        ).to(device)
        using_adapter = False
        if OCR_MUKUL_FLORENCE_ADAPTER_PATH.exists():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(OCR_MUKUL_FLORENCE_ADAPTER_PATH))
                using_adapter = True
            except Exception:
                using_adapter = False
        model.eval()
        state.update(
            {
                "available": True,
                "status": "ready" if using_adapter else "ready_base_only",
                "model": model,
                "processor": processor,
                "using_adapter": using_adapter,
            }
        )
    except Exception as exc:
        state["status"] = f"load_failed:{exc.__class__.__name__}"
    _FLORENCE_STATE = state
    return state


def _sample_video_frames(
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    max_frames: int | None,
) -> tuple[dict[str, Any], list[tuple[str, str, float, Path]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_dir = _ensure_dir(output_dir / "01_sampled_frames")
    sampled: list[tuple[str, str, float, Path]] = []
    seen_indices: set[int] = set()
    time_value = 0.0
    index = 0

    try:
        while True:
            if max_frames is not None and len(sampled) >= max_frames:
                break
            if duration_seconds > 0 and time_value > duration_seconds + 1e-6:
                break
            frame_idx = int(round(time_value * fps)) if fps > 0 else index
            if frame_count > 0:
                frame_idx = max(0, min(frame_count - 1, frame_idx))
            if frame_idx in seen_indices:
                time_value += sample_every_seconds
                if sample_every_seconds <= 0:
                    break
                continue
            seen_indices.add(frame_idx)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                break
            timestamp_seconds = (frame_idx / fps) if fps > 0 else time_value
            frame_id = f"frame_{index:06d}"
            frame_path = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame)
            sampled.append((frame_id, video_path.name, round(timestamp_seconds, 3), frame_path))
            index += 1
            time_value += sample_every_seconds
            if sample_every_seconds <= 0:
                break
    finally:
        capture.release()

    video_info = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "total_frames": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "width": width,
        "height": height,
        "sample_every_seconds": sample_every_seconds,
        "sampled_frame_count": len(sampled),
    }
    return video_info, sampled


def _detect_frames(
    sampled_frames: list[tuple[str, str, float, Path]],
    *,
    yolo_model_name: str,
    yolo_conf: float,
    yolo_imgsz: int,
    class_filter_ids: list[int] | None = None,
    allowed_class_names: set[str] | None = None,
) -> list[FrameDetection]:
    model = _load_yolo(yolo_model_name)
    allowed_names = allowed_class_names or ALLOWED_CLASS_NAMES
    frame_detections: list[FrameDetection] = []
    for frame_id, video_id, timestamp_seconds, frame_path in sampled_frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        frame_detections.append(
            _run_detection_on_image(
                image=image,
                model=model,
                frame_id=frame_id,
                video_id=video_id,
                timestamp_seconds=timestamp_seconds,
                yolo_conf=yolo_conf,
                yolo_imgsz=yolo_imgsz,
                class_filter_ids=class_filter_ids,
                allowed_names=allowed_names,
            )
        )
    return frame_detections


def _predict_detections_for_image(
    *,
    image: np.ndarray,
    model: Any,
    yolo_conf: float,
    yolo_imgsz: int,
    class_filter_ids: list[int] | None,
    allowed_names: set[str],
) -> list[Detection]:
    predict_kwargs: dict[str, Any] = {
        "source": image,
        "conf": yolo_conf,
        "imgsz": yolo_imgsz,
        "verbose": False,
    }
    if class_filter_ids:
        predict_kwargs["classes"] = class_filter_ids
    predictions = model.predict(**predict_kwargs)
    names = predictions[0].names if predictions else {}
    detections: list[Detection] = []
    if predictions:
        for box in predictions[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names.get(class_id, class_id)).strip().lower()
            if class_name not in allowed_names:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf[0].item()),
                    bbox=[float(value) for value in box.xyxy[0].tolist()],
                )
            )
    return detections


def _run_detection_on_image(
    *,
    image: np.ndarray,
    model: Any,
    frame_id: str,
    video_id: str,
    timestamp_seconds: float,
    yolo_conf: float,
    yolo_imgsz: int,
    class_filter_ids: list[int] | None,
    allowed_names: set[str],
) -> FrameDetection:
    detections = _predict_detections_for_image(
        image=image,
        model=model,
        yolo_conf=yolo_conf,
        yolo_imgsz=yolo_imgsz,
        class_filter_ids=class_filter_ids,
        allowed_names=allowed_names,
    )
    return FrameDetection(
        frame_id=frame_id,
        video_id=video_id,
        timestamp_seconds=float(timestamp_seconds),
        frame_width=int(image.shape[1]),
        frame_height=int(image.shape[0]),
        detections=detections,
    )


def _sample_video_frames_with_detection_queue(
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    max_frames: int | None,
    *,
    queue_size: int,
    unified_detector: dict[str, Any] | None = None,
    person_detector: dict[str, Any] | None = None,
    vehicle_detector: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str, float, Path]], list[FrameDetection]]:
    frame_queue: queue.Queue[tuple[str, str, float, Path] | object] = queue.Queue(maxsize=max(1, queue_size))
    sentinel = object()
    stop_event = threading.Event()
    sampled: list[tuple[str, str, float, Path]] = []
    frame_detections: list[FrameDetection] = []
    errors: list[BaseException] = []
    frames_dir = _ensure_dir(output_dir / "01_sampled_frames")

    metadata_capture = cv2.VideoCapture(str(video_path))
    if not metadata_capture.isOpened():
        metadata_capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(metadata_capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    width = int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    metadata_capture.release()

    if unified_detector:
        unified_model = _load_yolo(str(unified_detector["model"]))
        allowed_names = set(unified_detector.get("allowed_names") or ALLOWED_CLASS_NAMES)
    else:
        unified_model = None
        allowed_names = set()

    if person_detector:
        person_model = _load_yolo(str(person_detector["model"]))
    else:
        person_model = None

    if vehicle_detector:
        vehicle_model = _load_yolo(str(vehicle_detector["model"]))
    else:
        vehicle_model = None

    def _queue_put(item: tuple[str, str, float, Path] | object) -> None:
        while not stop_event.is_set():
            try:
                frame_queue.put(item, block=True, timeout=0.25)
                return
            except queue.Full:
                continue
        raise RuntimeError("Queue pipeline stopped before frame handoff completed.")

    def producer() -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            errors.append(RuntimeError(f"Could not open video: {video_path}"))
            try:
                _queue_put(sentinel)
            except Exception:
                pass
            return

        seen_indices: set[int] = set()
        time_value = 0.0
        index = 0
        try:
            while not stop_event.is_set():
                if max_frames is not None and len(sampled) >= max_frames:
                    break
                if duration_seconds > 0 and time_value > duration_seconds + 1e-6:
                    break
                frame_idx = int(round(time_value * fps)) if fps > 0 else index
                if frame_count > 0:
                    frame_idx = max(0, min(frame_count - 1, frame_idx))
                if frame_idx in seen_indices:
                    time_value += sample_every_seconds
                    if sample_every_seconds <= 0:
                        break
                    continue
                seen_indices.add(frame_idx)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = capture.read()
                if not success or frame is None:
                    break
                timestamp_seconds = (frame_idx / fps) if fps > 0 else time_value
                frame_id = f"frame_{index:06d}"
                frame_path = frames_dir / f"{frame_id}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write sampled frame: {frame_path}")
                frame_tuple = (frame_id, video_path.name, round(timestamp_seconds, 3), frame_path)
                sampled.append(frame_tuple)
                _queue_put(frame_tuple)
                index += 1
                time_value += sample_every_seconds
                if sample_every_seconds <= 0:
                    break
        except BaseException as exc:
            stop_event.set()
            errors.append(exc)
        finally:
            capture.release()
            try:
                _queue_put(sentinel)
            except Exception:
                pass

    def consumer() -> None:
        try:
            while True:
                try:
                    queued_item = frame_queue.get(block=True, timeout=0.25)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if queued_item is sentinel:
                    break
                frame_id, video_id, timestamp_seconds, frame_path = queued_item
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                if unified_model is not None:
                    frame_detections.append(
                        _run_detection_on_image(
                            image=image,
                            model=unified_model,
                            frame_id=frame_id,
                            video_id=video_id,
                            timestamp_seconds=timestamp_seconds,
                            yolo_conf=float(unified_detector["conf"]),
                            yolo_imgsz=int(unified_detector["imgsz"]),
                            class_filter_ids=unified_detector.get("class_filter_ids"),
                            allowed_names=allowed_names,
                        )
                    )
                    continue

                if person_model is None or vehicle_model is None:
                    raise RuntimeError("Queue consumer requires either a unified detector or both split detectors.")

                person_frame = _run_detection_on_image(
                    image=image,
                    model=person_model,
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp_seconds=timestamp_seconds,
                    yolo_conf=float(person_detector["conf"]),
                    yolo_imgsz=int(person_detector["imgsz"]),
                    class_filter_ids=list(person_detector.get("class_filter_ids") or []),
                    allowed_names=set(person_detector.get("allowed_names") or {PERSON_CLASS_NAME}),
                )
                vehicle_frame = _run_detection_on_image(
                    image=image,
                    model=vehicle_model,
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp_seconds=timestamp_seconds,
                    yolo_conf=float(vehicle_detector["conf"]),
                    yolo_imgsz=int(vehicle_detector["imgsz"]),
                    class_filter_ids=vehicle_detector.get("class_filter_ids"),
                    allowed_names=set(vehicle_detector.get("allowed_names") or VEHICLE_CLASS_NAMES),
                )
                frame_detections.extend(_merge_frame_detections([[person_frame], [vehicle_frame]]))
        except BaseException as exc:
            stop_event.set()
            errors.append(exc)

    producer_thread = threading.Thread(target=producer, name="object-search-frame-producer", daemon=True)
    consumer_thread = threading.Thread(target=consumer, name="object-search-frame-consumer", daemon=True)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()

    if errors:
        raise RuntimeError(f"Queue pipeline failed: {errors[0]}") from errors[0]

    video_info = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "total_frames": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "width": width,
        "height": height,
        "sample_every_seconds": sample_every_seconds,
        "sampled_frame_count": len(sampled),
        "queue_pipeline_enabled": True,
        "queue_size": int(queue_size),
    }
    return video_info, sampled, frame_detections


def _merge_frame_detections(
    detection_groups: list[list[FrameDetection]],
    *,
    dedupe_iou_threshold: float = 0.7,
) -> list[FrameDetection]:
    if not detection_groups:
        return []

    merged: list[FrameDetection] = []
    frame_count = len(detection_groups[0])
    for group in detection_groups[1:]:
        if len(group) != frame_count:
            raise ValueError("All detection groups must have the same frame count.")

    for frame_index in range(frame_count):
        base_item = detection_groups[0][frame_index]
        merged_detections: list[Detection] = []

        for group in detection_groups:
            frame_item = group[frame_index]
            for detection in frame_item.detections:
                duplicate = False
                for existing in merged_detections:
                    if (
                        existing.class_name == detection.class_name
                        and _bbox_iou(existing.bbox, detection.bbox) >= dedupe_iou_threshold
                    ):
                        duplicate = True
                        if detection.confidence > existing.confidence:
                            existing.class_id = detection.class_id
                            existing.class_name = detection.class_name
                            existing.confidence = detection.confidence
                            existing.bbox = detection.bbox
                        break
                if not duplicate:
                    merged_detections.append(
                        Detection(
                            class_id=detection.class_id,
                            class_name=detection.class_name,
                            confidence=detection.confidence,
                            bbox=list(detection.bbox),
                        )
                    )

        merged.append(
            FrameDetection(
                frame_id=base_item.frame_id,
                video_id=base_item.video_id,
                timestamp_seconds=base_item.timestamp_seconds,
                frame_width=base_item.frame_width,
                frame_height=base_item.frame_height,
                detections=merged_detections,
            )
        )
    return merged


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _crop_image(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return image[y1:y2, x1:x2]


def _classify_bgr_color(bgr_pixel: np.ndarray) -> str:
    b, g, r = [int(v) for v in bgr_pixel.tolist()]
    sample = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = [int(x) for x in hsv.tolist()]
    if v < 45:
        return "black"
    if s < 35:
        if v > 210:
            return "white"
        if v > 120:
            return "grey"
        return "black"
    if h < 8 or h >= 170:
        return "red"
    if h < 18:
        return "orange"
    if h < 32:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    if h < 170:
        return "pink"
    return "brown"


def _dominant_color_label(image: np.ndarray) -> str:
    if image.size == 0:
        return "unknown"
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3)).astype(np.float32)
    if pixels.shape[0] == 0:
        return "unknown"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cluster_count = min(3, len(pixels))
    _compactness, labels, centers = cv2.kmeans(
        pixels,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=cluster_count)
    dominant_center = centers[int(np.argmax(counts))]
    return _classify_bgr_color(np.asarray(dominant_center, dtype=np.uint8))


def _person_appearance_terms(crop: np.ndarray, carrying_bag: bool) -> list[str]:
    if crop.size == 0:
        return ["person"]
    height = crop.shape[0]
    upper = crop[: max(1, height // 2), :]
    lower = crop[max(0, height // 2) :, :]
    upper_color = _dominant_color_label(upper)
    lower_color = _dominant_color_label(lower)
    terms = ["person", upper_color, f"{upper_color} upper clothing", lower_color, f"{lower_color} lower clothing"]
    if carrying_bag:
        terms.extend(["bag", "person with bag", "carrying bag"])
    return terms


def _object_appearance_terms(class_name: str, crop: np.ndarray) -> list[str]:
    color = _dominant_color_label(crop)
    terms = [class_name]
    if class_name in VEHICLE_CLASS_NAMES:
        terms.extend(["vehicle", color, f"{color} {class_name}", f"{color} vehicle"])
    elif class_name in BAG_CLASS_NAMES:
        terms.extend(["bag", color, f"{color} bag"])
    else:
        terms.extend([color, f"{color} {class_name}"])
    return terms


def _resize_proportionally_if_needed(image: np.ndarray, target_width: int = 200, target_height: int = 150) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
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


def _run_florence_inference(image_cv: np.ndarray, task_prompt: str, text_input: str | None = None, use_adapter: bool = True) -> str | None:
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

    context = torch.no_grad()
    with context:
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


def _extract_vehicle_plate_metadata(
    *,
    class_name: str,
    crop: np.ndarray,
    plate_crops_dir: Path,
    object_id: str,
    frame_id: str,
    timestamp_seconds: float,
) -> dict[str, Any]:
    base_result = {
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
        return base_result

    vehicle_color = _run_florence_inference(crop, "<VQA>", "What is the primary color of the vehicle?", use_adapter=False)
    if vehicle_color:
        normalized_color = vehicle_color.strip().lower()
        base_result["vehicle_color_florence"] = normalized_color
        base_result["vehicle_color_status"] = "success"
    else:
        base_result["vehicle_color_status"] = "florence_unavailable_or_failed"

    plate_region, plate_confidence, plate_detection_status = _detect_plate_crop_from_vehicle(crop)
    base_result["plate_detection_status"] = plate_detection_status
    base_result["license_plate_confidence"] = round(float(plate_confidence), 4) if plate_confidence is not None else None
    if plate_region is None:
        base_result["plate_ocr_status"] = plate_detection_status
        return base_result

    plate_crop_path = plate_crops_dir / f"{object_id}_{frame_id}_plate.jpg"
    cv2.imwrite(str(plate_crop_path), plate_region)
    base_result["plate_crop_path"] = _to_repo_relative(plate_crop_path)
    extracted_text = _run_florence_inference(plate_region, "<OCR>", use_adapter=True)
    if not extracted_text:
        base_result["plate_ocr_status"] = "florence_ocr_unavailable_or_failed"
        return base_result

    normalized_text = re.sub(r"[^A-Z0-9]", "", extracted_text.upper())
    detected_text = [str(extracted_text).strip()] if str(extracted_text).strip() else []
    license_plates = [normalized_text] if normalized_text and _is_valid_indian_plate(normalized_text) else []
    base_result["plate_detected_text"] = list(dict.fromkeys(detected_text + ([normalized_text] if normalized_text else [])))
    base_result["license_plates"] = list(dict.fromkeys(license_plates))
    base_result["best_license_plate"] = base_result["license_plates"][0] if base_result["license_plates"] else None
    base_result["plate_ocr_status"] = "success" if base_result["license_plates"] else "ocr_attempted_no_match"
    return base_result


def _build_track_summaries(
    *,
    frame_detections: list[FrameDetection],
    tracking_results_path: Path,
    sampled_frames: list[tuple[str, str, float, Path]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracking_payload = json.loads(tracking_results_path.read_text(encoding="utf-8"))
    tracking_frames = tracking_payload.get("frames", [])
    frame_detection_map = {item.frame_id: item for item in frame_detections}
    frame_path_map = {frame_id: frame_path for frame_id, _video_id, _ts, frame_path in sampled_frames}
    crops_dir = _ensure_dir(output_dir / "02_object_crops")
    plate_crops_dir = _ensure_dir(output_dir / "03_plate_crops")

    tracks: dict[str, dict[str, Any]] = {}
    search_index: list[dict[str, Any]] = []

    for frame_row in tracking_frames:
        frame_id = str(frame_row.get("frame_id", "")).strip()
        timestamp_seconds = float(frame_row.get("timestamp_seconds", 0.0) or 0.0)
        frame_path = frame_path_map.get(frame_id)
        image = cv2.imread(str(frame_path)) if frame_path else None
        source_detections = frame_detection_map.get(frame_id).detections if frame_id in frame_detection_map else []

        for entity in frame_row.get("tracked_entities", []):
            raw_object_id = str(entity.get("global_actor_id", "")).strip() or f"track_{entity.get('track_id', 'unknown')}"
            class_name = str(entity.get("class_name", "unknown")).strip().lower()
            object_id = f"{raw_object_id}_{class_name}"
            bbox = [float(v) for v in entity.get("bbox", [])]
            confidence = float(entity.get("confidence", 0.0) or 0.0)
            if image is None or not bbox:
                continue
            crop = _crop_image(image, bbox)
            if crop.size == 0:
                continue

            carrying_bag = False
            if class_name == PERSON_CLASS_NAME:
                for det in source_detections:
                    if det.class_name in BAG_CLASS_NAMES and _bbox_iou(bbox, det.bbox) > 0.01:
                        carrying_bag = True
                        break

            appearance_terms = (
                _person_appearance_terms(crop, carrying_bag)
                if class_name == PERSON_CLASS_NAME
                else _object_appearance_terms(class_name, crop)
            )
            plate_metadata = _extract_vehicle_plate_metadata(
                class_name=class_name,
                crop=crop,
                plate_crops_dir=plate_crops_dir,
                object_id=object_id,
                frame_id=frame_id,
                timestamp_seconds=timestamp_seconds,
            )
            if plate_metadata["best_license_plate"]:
                appearance_terms.extend(
                    [
                        "license plate visible",
                        plate_metadata["best_license_plate"],
                        f"plate {plate_metadata['best_license_plate']}",
                    ]
                )
            if plate_metadata.get("vehicle_color_florence"):
                vehicle_color_term = str(plate_metadata["vehicle_color_florence"]).strip().lower()
                appearance_terms.extend([vehicle_color_term, f"{vehicle_color_term} {class_name}"])
            crop_name = f"{object_id}_{frame_id}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
            score = confidence * max(area, 1.0)

            track = tracks.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "source_actor_id": raw_object_id,
                    "class_name": class_name,
                    "track_id_history": [],
                    "start_time": timestamp_seconds,
                    "end_time": timestamp_seconds,
                    "frame_hits": [],
                    "appearance_terms": set(),
                    "best_score": -1.0,
                    "best_frame_path": None,
                    "best_crop_path": None,
                    "best_timestamp_seconds": timestamp_seconds,
                    "best_plate_score": -1.0,
                    "best_license_plate": None,
                    "best_license_plate_frame_path": None,
                    "best_license_plate_crop_path": None,
                    "best_license_plate_timestamp_seconds": None,
                    "best_license_plate_confidence": None,
                    "plate_detected_text": set(),
                    "license_plates": set(),
                    "plate_ocr_statuses": set(),
                    "plate_detection_statuses": set(),
                    "vehicle_color_florence": None,
                    "vehicle_color_statuses": set(),
                    "plate_candidates": [],
                },
            )
            track["track_id_history"].append(int(entity.get("track_id", 0) or 0))
            track["start_time"] = min(float(track["start_time"]), timestamp_seconds)
            track["end_time"] = max(float(track["end_time"]), timestamp_seconds)
            track["frame_hits"].append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "frame_path": _to_repo_relative(frame_path) if frame_path else None,
                    "crop_path": _to_repo_relative(crop_path),
                    "bbox": [round(v, 2) for v in bbox],
                    "confidence": round(confidence, 4),
                    "plate_crop_path": plate_metadata.get("plate_crop_path"),
                    "plate_ocr_status": plate_metadata.get("plate_ocr_status"),
                    "plate_detection_status": plate_metadata.get("plate_detection_status"),
                    "license_plate_confidence": plate_metadata.get("license_plate_confidence"),
                    "plate_detected_text": plate_metadata.get("plate_detected_text", []),
                    "license_plates": plate_metadata.get("license_plates", []),
                    "vehicle_color_florence": plate_metadata.get("vehicle_color_florence"),
                    "vehicle_color_status": plate_metadata.get("vehicle_color_status"),
                }
            )
            track["appearance_terms"].update(str(term).strip().lower() for term in appearance_terms if str(term).strip())
            track["plate_ocr_statuses"].add(str(plate_metadata.get("plate_ocr_status", "not_applicable")))
            track.setdefault("plate_detection_statuses", set()).add(str(plate_metadata.get("plate_detection_status", "not_applicable")))
            track.setdefault("vehicle_color_statuses", set()).add(str(plate_metadata.get("vehicle_color_status", "not_applicable")))
            track["plate_detected_text"].update(
                str(term).strip() for term in plate_metadata.get("plate_detected_text", []) if str(term).strip()
            )
            track["license_plates"].update(
                str(term).strip().upper() for term in plate_metadata.get("license_plates", []) if str(term).strip()
            )
            if plate_metadata.get("vehicle_color_florence") and not track.get("vehicle_color_florence"):
                track["vehicle_color_florence"] = str(plate_metadata.get("vehicle_color_florence")).strip().lower()
            if plate_metadata.get("best_license_plate"):
                plate_candidate = {
                    "license_plate": plate_metadata["best_license_plate"],
                    "frame_id": frame_id,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "timestamp_text": _format_seconds(timestamp_seconds),
                    "frame_path": _to_repo_relative(frame_path) if frame_path else None,
                    "crop_path": _to_repo_relative(crop_path),
                    "plate_crop_path": plate_metadata.get("plate_crop_path"),
                    "license_plate_confidence": plate_metadata.get("license_plate_confidence"),
                }
                track["plate_candidates"].append(plate_candidate)
                if score > float(track["best_plate_score"]):
                    track["best_plate_score"] = score
                    track["best_license_plate"] = plate_metadata["best_license_plate"]
                    track["best_license_plate_frame_path"] = _to_repo_relative(frame_path) if frame_path else None
                    track["best_license_plate_crop_path"] = plate_metadata.get("plate_crop_path")
                    track["best_license_plate_timestamp_seconds"] = round(timestamp_seconds, 3)
                    track["best_license_plate_confidence"] = plate_metadata.get("license_plate_confidence")
            if score > float(track["best_score"]):
                track["best_score"] = score
                track["best_frame_path"] = _to_repo_relative(frame_path) if frame_path else None
                track["best_crop_path"] = _to_repo_relative(crop_path)
                track["best_timestamp_seconds"] = round(timestamp_seconds, 3)

    for track in tracks.values():
        duration_seconds = max(0.0, float(track["end_time"]) - float(track["start_time"]))
        appearance_terms = sorted(set(track["appearance_terms"]))
        plate_detected_text = sorted(set(track["plate_detected_text"]))
        license_plates = sorted(set(track["license_plates"]))
        unique_plate_candidates: list[dict[str, Any]] = []
        seen_plate_keys: set[tuple[str, str]] = set()
        for candidate in sorted(track["plate_candidates"], key=lambda item: (item["license_plate"], item["timestamp_seconds"])):
            candidate_key = (str(candidate["license_plate"]), str(candidate["frame_id"]))
            if candidate_key in seen_plate_keys:
                continue
            seen_plate_keys.add(candidate_key)
            unique_plate_candidates.append(candidate)
        search_text = " ".join(
            [
                track["object_id"],
                track["class_name"],
                " ".join(appearance_terms),
                " ".join(plate_detected_text),
                " ".join(license_plates),
            ]
        ).strip().lower()
        output_track = {
            "object_id": track["object_id"],
            "source_actor_id": track["source_actor_id"],
            "class_name": track["class_name"],
            "track_id_history": sorted(set(int(x) for x in track["track_id_history"] if int(x) > 0)),
            "start_time": round(float(track["start_time"]), 3),
            "end_time": round(float(track["end_time"]), 3),
            "start_time_text": _format_seconds(float(track["start_time"])),
            "end_time_text": _format_seconds(float(track["end_time"])),
            "duration_seconds": round(duration_seconds, 3),
            "frame_hit_count": len(track["frame_hits"]),
            "best_timestamp_seconds": track["best_timestamp_seconds"],
            "best_timestamp_text": _format_seconds(float(track["best_timestamp_seconds"])),
            "best_frame_path": track["best_frame_path"],
            "best_crop_path": track["best_crop_path"],
            "appearance_terms": appearance_terms,
            "plate_ocr_status": (
                "success"
                if license_plates
                else "not_applicable"
                if track["class_name"] not in LICENSE_PLATE_CLASS_NAMES
                else "ocr_attempted_no_match"
            ),
            "plate_ocr_statuses": sorted(set(track["plate_ocr_statuses"])),
            "plate_detection_statuses": sorted(set(track.get("plate_detection_statuses", set()))),
            "plate_detected_text": plate_detected_text,
            "license_plates": license_plates,
            "best_license_plate": track["best_license_plate"],
            "best_license_plate_frame_path": track["best_license_plate_frame_path"],
            "best_license_plate_crop_path": track["best_license_plate_crop_path"],
            "best_license_plate_timestamp_seconds": track["best_license_plate_timestamp_seconds"],
            "best_license_plate_confidence": track.get("best_license_plate_confidence"),
            "best_license_plate_timestamp_text": _format_seconds(track["best_license_plate_timestamp_seconds"])
            if track["best_license_plate_timestamp_seconds"] is not None
            else None,
            "vehicle_color_florence": track.get("vehicle_color_florence"),
            "vehicle_color_statuses": sorted(set(track.get("vehicle_color_statuses", set()))),
            "plate_candidates": unique_plate_candidates[:10],
            "search_text": search_text,
            "frame_hits": sorted(track["frame_hits"], key=lambda item: float(item["timestamp_seconds"])),
        }
        search_index.append(output_track)

    search_index.sort(key=lambda item: (float(item["start_time"]), item["class_name"], item["object_id"]))
    return search_index, tracking_frames


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated YOLO + tracking + object-search testcase")
    parser.add_argument("--video", required=True, help="Absolute or relative path to input video")
    parser.add_argument("--output-dir", default="", help="Optional output directory")
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--use-local-split-models", action="store_true", help="Use Person_detection/Person_detection.pt + object_yolo/best_old.pt together.")
    parser.add_argument("--person-model", default="", help="Optional person detector .pt path. When set with --vehicle-model, detections are merged.")
    parser.add_argument("--vehicle-model", default="", help="Optional vehicle detector model path. When set with --person-model, detections are merged.")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--vehicle-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--person-imgsz", type=int, default=640)
    parser.add_argument("--vehicle-imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--use-frame-queue", action="store_true", help="Overlap frame sampling and detection with a producer-consumer queue.")
    parser.add_argument("--queue-size", type=int, default=16, help="Maximum sampled frames buffered between producer and detector.")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.is_absolute():
        video_path = (_repo_root() / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    person_model_value = str(args.person_model).strip()
    vehicle_model_value = str(args.vehicle_model).strip()
    if args.use_local_split_models:
        person_model_value = str(DEFAULT_LOCAL_PERSON_MODEL)
        vehicle_model_value = str(DEFAULT_LOCAL_VEHICLE_MODEL)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_repo_root() / output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _debug_root() / f"{_safe_name(video_path)}_{timestamp}"
    _ensure_dir(output_dir)

    print(f"[object-search-case] Video: {video_path}")
    print(f"[object-search-case] Output dir: {output_dir}")

    using_split_detectors = bool(person_model_value) and bool(vehicle_model_value)
    detection_metadata: dict[str, Any]
    frame_detections: list[FrameDetection]
    sample_every_seconds = max(0.1, float(args.sample_every_seconds))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None
    if using_split_detectors:
        person_model_path = Path(person_model_value).expanduser()
        vehicle_model_path = Path(vehicle_model_value).expanduser()
        if not person_model_path.is_absolute():
            person_model_path = (_repo_root() / person_model_path).resolve()
        if not vehicle_model_path.is_absolute():
            vehicle_model_path = (_repo_root() / vehicle_model_path).resolve()
        if not person_model_path.exists():
            raise FileNotFoundError(f"Person model not found: {person_model_path}")
        if not vehicle_model_path.exists():
            raise FileNotFoundError(f"Vehicle model not found: {vehicle_model_path}")

        print(f"[object-search-case] Person detector: {person_model_path}")
        print(f"[object-search-case] Vehicle detector: {vehicle_model_path}")
        detection_metadata = {
            "mode": "split_person_vehicle_models",
            "person_model": str(person_model_path),
            "person_conf": float(args.person_conf),
            "person_imgsz": int(args.person_imgsz),
            "person_classes": PERSON_CLASS_FILTER_IDS,
            "vehicle_model": str(vehicle_model_path),
            "vehicle_conf": float(args.vehicle_conf),
            "vehicle_imgsz": int(args.vehicle_imgsz),
            "vehicle_allowed_classes": sorted(VEHICLE_CLASS_NAMES),
            "queue_pipeline_enabled": bool(args.use_frame_queue),
            "queue_size": int(args.queue_size),
        }
        if args.use_frame_queue:
            print(f"[object-search-case] Queue pipeline enabled (size={int(args.queue_size)})")
            video_info, sampled_frames, frame_detections = _sample_video_frames_with_detection_queue(
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                max_frames=max_frames,
                queue_size=int(args.queue_size),
                person_detector={
                    "model": str(person_model_path),
                    "conf": float(args.person_conf),
                    "imgsz": int(args.person_imgsz),
                    "class_filter_ids": PERSON_CLASS_FILTER_IDS,
                    "allowed_names": {PERSON_CLASS_NAME},
                },
                vehicle_detector={
                    "model": str(vehicle_model_path),
                    "conf": float(args.vehicle_conf),
                    "imgsz": int(args.vehicle_imgsz),
                    "allowed_names": VEHICLE_CLASS_NAMES,
                },
            )
        else:
            video_info, sampled_frames = _sample_video_frames(
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                max_frames=max_frames,
            )
            person_detections = _detect_frames(
                sampled_frames,
                yolo_model_name=str(person_model_path),
                yolo_conf=float(args.person_conf),
                yolo_imgsz=int(args.person_imgsz),
                class_filter_ids=PERSON_CLASS_FILTER_IDS,
                allowed_class_names={PERSON_CLASS_NAME},
            )
            vehicle_detections = _detect_frames(
                sampled_frames,
                yolo_model_name=str(vehicle_model_path),
                yolo_conf=float(args.vehicle_conf),
                yolo_imgsz=int(args.vehicle_imgsz),
                allowed_class_names=VEHICLE_CLASS_NAMES,
            )
            frame_detections = _merge_frame_detections([person_detections, vehicle_detections])
    else:
        print(f"[object-search-case] Unified detector: {args.yolo_model}")
        detection_metadata = {
            "mode": "single_model",
            "yolo_model": str(args.yolo_model),
            "yolo_conf": float(args.yolo_conf),
            "yolo_imgsz": int(args.yolo_imgsz),
            "queue_pipeline_enabled": bool(args.use_frame_queue),
            "queue_size": int(args.queue_size),
        }
        if args.use_frame_queue:
            print(f"[object-search-case] Queue pipeline enabled (size={int(args.queue_size)})")
            video_info, sampled_frames, frame_detections = _sample_video_frames_with_detection_queue(
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                max_frames=max_frames,
                queue_size=int(args.queue_size),
                unified_detector={
                    "model": str(args.yolo_model),
                    "conf": float(args.yolo_conf),
                    "imgsz": int(args.yolo_imgsz),
                    "allowed_names": ALLOWED_CLASS_NAMES,
                },
            )
        else:
            video_info, sampled_frames = _sample_video_frames(
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                max_frames=max_frames,
            )
            frame_detections = _detect_frames(
                sampled_frames,
                yolo_model_name=str(args.yolo_model),
                yolo_conf=float(args.yolo_conf),
                yolo_imgsz=int(args.yolo_imgsz),
            )

    _write_json(output_dir / "01_video_info.json", video_info)
    _write_json(
        output_dir / "02_sampled_frames.json",
        {
            "items": [
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_text": _format_seconds(timestamp_seconds),
                    "frame_path": _to_repo_relative(frame_path),
                }
                for frame_id, video_id, timestamp_seconds, frame_path in sampled_frames
            ]
        },
    )
    _write_json(
        output_dir / "03_detections.json",
        {
            "detection_setup": detection_metadata,
            "items": [
                {
                    "frame_id": item.frame_id,
                    "timestamp_seconds": round(item.timestamp_seconds, 3),
                    "timestamp_text": _format_seconds(item.timestamp_seconds),
                    "frame_width": item.frame_width,
                    "frame_height": item.frame_height,
                    "detections": [
                        {
                            "class_id": detection.class_id,
                            "class_name": detection.class_name,
                            "confidence": round(detection.confidence, 4),
                            "bbox": [round(v, 2) for v in detection.bbox],
                        }
                        for detection in item.detections
                    ],
                }
                for item in frame_detections
            ]
        },
    )

    benchmark_inputs = [
        SimpleNamespace(
            frame_id=item.frame_id,
            video_id=item.video_id,
            timestamp_seconds=item.timestamp_seconds,
            frame_width=item.frame_width,
            frame_height=item.frame_height,
            detections=[
                SimpleNamespace(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
                for detection in item.detections
            ],
        )
        for item in frame_detections
    ]
    tracking_dir = _ensure_dir(output_dir / "04_tracking")
    BenchmarkBoTSORTTracker.track_frames(
        benchmark_inputs,
        extracted_tuples=sampled_frames,
        debug_output_dir=tracking_dir,
    )

    search_index, _tracking_frames = _build_track_summaries(
        frame_detections=frame_detections,
        tracking_results_path=tracking_dir / "tracking_results.json",
        sampled_frames=sampled_frames,
        output_dir=output_dir,
    )
    _write_json(output_dir / "05_object_tracks.json", {"items": search_index})
    _write_json(output_dir / "06_searchable_object_index.json", {"items": search_index})

    summary = {
        "video_name": video_info["video_name"],
        "duration_seconds": video_info["duration_seconds"],
        "sample_every_seconds": video_info["sample_every_seconds"],
        "sampled_frame_count": video_info["sampled_frame_count"],
        "detection_setup": detection_metadata,
        "ocr_assets_dir": str(OCR_MUKUL_DIR),
        "plate_detector_path": str(OCR_MUKUL_PLATE_MODEL_PATH),
        "florence_adapter_path": str(OCR_MUKUL_FLORENCE_ADAPTER_PATH),
        "tracked_object_count": len(search_index),
        "tracks_with_license_plate": sum(1 for item in search_index if item.get("best_license_plate")),
        "tracks_with_florence_vehicle_color": sum(1 for item in search_index if item.get("vehicle_color_florence")),
        "tracked_by_class": {
            class_name: sum(1 for item in search_index if item["class_name"] == class_name)
            for class_name in sorted({item["class_name"] for item in search_index})
        },
    }
    _write_json(output_dir / "07_summary.json", summary)

    print("[object-search-case] Done")
    print(f"[object-search-case] Sampled frames: {video_info['sampled_frame_count']}")
    print(f"[object-search-case] Tracked objects: {len(search_index)}")
    print(f"[object-search-case] Search index: {output_dir / '06_searchable_object_index.json'}")


if __name__ == "__main__":
    main()
