from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.td_case2.config import ENV_YOLO_DEVICE
from tests.td_case2.device_manager import cuda_memory_allocated_mb, cuda_memory_reserved_mb, resolve_device
from tests.td_case2.step_03b_yolo_detection import _load_yolo_class

from .fixed_5fps_validation_core import detector_should_run
from .reid_feature_cache import save_feature_cache, write_feature_cache_metadata
from .report_writer import write_json
from .track_state import object_family_for_class
from .video_frame_stream import read_video_info, stream_processed_frames

SUPPORTED_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class LiveReidDetectionConfig:
    video_path: Path
    processing_fps: float
    detector_fps: float
    confidence: float
    iou: float
    model_path: Path
    device: str


def _resolve_device(device: str) -> str:
    if device == "auto":
        decision = resolve_device(component_name="verified_botsort_reid_yolo", override_env_names=(ENV_YOLO_DEVICE,))
        return decision.ultralytics_device
    return device


def _initialize_yolo(model_path: Path, device: str):
    YOLO = _load_yolo_class()
    model = YOLO(str(model_path))
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=0.25, iou=0.45, device=device, verbose=False)
    predictor = model.predictor
    predictor._feats = None

    def pre_hook(module, input):
        predictor._feats = list(input[0])

    hook = model.model.model[-1].register_forward_pre_hook(pre_hook)
    return model, predictor, hook


def build_live_detection_feature_cache(*, config: LiveReidDetectionConfig, shared_dir: Path) -> dict[str, Any]:
    video_info = read_video_info(config.video_path)
    resolved_device = _resolve_device(config.device)
    model, predictor, hook = _initialize_yolo(config.model_path, resolved_device)
    _, _, _, iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    schedule: list[dict[str, Any]] = []
    cached_detector_frames: list[dict[str, Any]] = []
    yolo_calls: list[dict[str, Any]] = []
    features_by_frame: dict[int, np.ndarray] = {}
    started = time.perf_counter()
    try:
        for frame_record, frame in iterator:
            frame_payload = frame_record.to_dict()
            processed_frame_index = int(frame_payload["processed_frame_index"])
            detector_ran = detector_should_run(
                processed_frame_index=processed_frame_index,
                processing_fps=config.processing_fps,
                detector_fps=config.detector_fps,
            )
            if detector_ran:
                predictor._feats = None
                results = model.predict(
                    source=frame,
                    conf=config.confidence,
                    iou=config.iou,
                    device=resolved_device,
                    verbose=False,
                )
                result = results[0]
                boxes = getattr(result, "boxes", None)
                detections: list[dict[str, Any]] = []
                feature_rows: list[np.ndarray] = []
                if boxes is not None:
                    xyxy_list = boxes.xyxy.tolist()
                    cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
                    conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
                    feats = getattr(result, "feats", None)
                    for index, bbox_xyxy in enumerate(xyxy_list):
                        class_id = int(cls_list[index]) if index < len(cls_list) else -1
                        class_name = str(model.names.get(class_id, class_id)).lower()
                        if class_name not in SUPPORTED_CLASSES:
                            continue
                        detection_id = f"det_{processed_frame_index:06d}_{len(detections) + 1:03d}"
                        detections.append(
                            {
                                "detection_id": detection_id,
                                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
                                "confidence": round(float(conf_list[index]) if index < len(conf_list) else 0.0, 6),
                                "class_id": class_id,
                                "class_name": class_name,
                                "object_family": object_family_for_class(class_name),
                            }
                        )
                        if feats is not None and len(feats) > index:
                            feature_rows.append(np.asarray(feats[index].detach().cpu().numpy(), dtype=np.float32))
                feature_array = np.stack(feature_rows, axis=0) if feature_rows else np.zeros((0, 0), dtype=np.float32)
                if feature_array.size:
                    features_by_frame[processed_frame_index] = feature_array
                cached_detector_frames.append(
                    {
                        "source_frame_index": int(frame_payload["source_frame_index"]),
                        "processed_frame_index": processed_frame_index,
                        "timestamp_seconds": round(float(frame_payload["timestamp_seconds"]), 6),
                        "detector_ran": True,
                        "image_width": int(frame.shape[1]),
                        "image_height": int(frame.shape[0]),
                        "detections": detections,
                        "feature_count": int(feature_array.shape[0]) if feature_array.ndim == 2 else 0,
                        "feature_dimension": int(feature_array.shape[1]) if feature_array.ndim == 2 and feature_array.shape[0] > 0 else 0,
                    }
                )
                yolo_calls.append(
                    {
                        "processed_frame_index": processed_frame_index,
                        "timestamp_seconds": round(float(frame_payload["timestamp_seconds"]), 6),
                        "detector_ran": True,
                        "detection_count": len(detections),
                        "feature_count": int(feature_array.shape[0]) if feature_array.ndim == 2 else 0,
                    }
                )
            schedule.append(
                {
                    "source_frame_index": int(frame_payload["source_frame_index"]),
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": round(float(frame_payload["timestamp_seconds"]), 6),
                    "detector_ran": detector_ran,
                }
            )
    finally:
        hook.remove()
    runtime_seconds = round(time.perf_counter() - started, 6)
    feature_npz_path = shared_dir / "reid_features.npz"
    metadata = save_feature_cache(output_path=feature_npz_path, features_by_frame=features_by_frame)
    metadata.update(
        {
            "status": "success",
            "requested_model": "auto",
            "native_detector_features_used": True,
            "external_encoder_used": False,
            "gpu_device": resolved_device,
            "peak_gpu_vram_allocated_mb": cuda_memory_allocated_mb() or "not_available",
            "peak_gpu_vram_reserved_mb": cuda_memory_reserved_mb() or "not_available",
        }
    )
    write_feature_cache_metadata(metadata_path=shared_dir / "reid_feature_cache_metadata.json", payload=metadata)
    write_json(shared_dir / "video_info.json", {"status": "success", "video_path": str(config.video_path), "source_fps": video_info.source_fps, "source_frame_count": video_info.source_frame_count, "duration_seconds": video_info.duration_seconds, "width": video_info.width, "height": video_info.height})
    write_json(shared_dir / "resolved_detection_config.json", {"status": "success", "processing_fps": config.processing_fps, "detector_fps": config.detector_fps, "yolo_confidence": config.confidence, "yolo_iou": config.iou, "device": config.device, "resolved_device": resolved_device, "adaptive_scheduler": "disabled", "native_feature_capture": True})
    write_json(shared_dir / "frame_schedule.json", {"status": "success", "frames": schedule})
    write_json(shared_dir / "cached_yolo_detections.json", {"status": "success", "frames": cached_detector_frames})
    write_json(shared_dir / "yolo_call_report.json", {"status": "success", "calls": yolo_calls, "runtime_seconds": runtime_seconds})
    canonical = json.dumps(cached_detector_frames, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(canonical).hexdigest()
    checksum_payload = {"status": "success", "sha256": checksum, "detector_frame_count": len(cached_detector_frames)}
    write_json(shared_dir / "detection_cache_checksum.json", checksum_payload)
    return {
        "video_info": {
            "source_fps": video_info.source_fps,
            "source_frame_count": video_info.source_frame_count,
            "duration_seconds": video_info.duration_seconds,
            "width": video_info.width,
            "height": video_info.height,
        },
        "frame_schedule": schedule,
        "cached_yolo_detections": cached_detector_frames,
        "detection_cache_checksum": checksum_payload,
        "yolo_call_report": yolo_calls,
        "feature_cache_path": feature_npz_path,
        "feature_cache_metadata": metadata,
    }
