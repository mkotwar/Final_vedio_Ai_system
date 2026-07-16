from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    DEFAULT_TRACKING_MIN_TRACK_LENGTH,
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_IOU_THRESHOLD,
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_YOLO_DEVICE,
    ENV_YOLO_MODEL_PATH,
    resolve_case_path,
)
from device_manager import resolve_device
from experiments.step04b_bytetrack.bytetrack_adapter import DetectionSet, load_bytetrack_types
from experiments.step04b_bytetrack.fragment_merger import merge_track_fragments
from step_03b_yolo_detection import _load_yolo_class


SUPPORTED_TRACKING_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}
DEFAULT_PERSON_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\object\Person_detection (1)\Person_detection.pt")
DEFAULT_OBJECT_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\object\vehical_detection")


@dataclass(frozen=True)
class PipelineConfig:
    model_specs: list[dict[str, Any]]
    conf_threshold: float
    iou_threshold: float
    device: str
    device_reason: str
    save_annotated: bool
    save_crops: bool
    track_buffer_seconds: float
    high_confidence: float
    low_confidence: float
    match_threshold: float
    min_track_length: int


@dataclass
class YoloFrameProcessor:
    experiment_dir: Path
    config: PipelineConfig
    annotated_dir_name: str
    crop_dir_name: str

    def __post_init__(self) -> None:
        YOLO = _load_yolo_class()
        self.loaded_models = {item["model_role"]: YOLO(str(item["model_path"])) for item in self.config.model_specs}
        self.class_lookups: dict[str, dict[str, str]] = {}
        for role, model in self.loaded_models.items():
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                self.class_lookups[role] = {str(key): str(value) for key, value in names.items()}
            else:
                self.class_lookups[role] = {str(index): str(value) for index, value in enumerate(names)}
        self.annotated_dir = self.experiment_dir / self.annotated_dir_name
        self.crops_dir = self.experiment_dir / self.crop_dir_name
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(self, *, frame_item: dict[str, Any], original_image: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        annotated_image = original_image.copy()
        frame_height, frame_width = original_image.shape[:2]
        frame_payload = {
            "frame_id": frame_item["frame_id"],
            "frame_idx": frame_item["frame_idx"],
            "timestamp_seconds": frame_item["timestamp_seconds"],
            "timestamp_text": frame_item["timestamp_text"],
            "image_path": frame_item["image_path"],
            "tracking_state": frame_item["tracking_state"],
            "target_fps": frame_item["target_fps"],
            "selection_reason": list(frame_item.get("selection_reason", [])),
            "detections": [],
        }
        started = time.perf_counter()
        for model_spec in self.config.model_specs:
            role = str(model_spec["model_role"])
            results = self.loaded_models[role].predict(
                source=original_image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                device=self.config.device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            annotated_image = result.plot()
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_list = boxes.xyxy.tolist()
            xywh_list = boxes.xywh.tolist() if getattr(boxes, "xywh", None) is not None else []
            cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            for index, bbox_xyxy in enumerate(xyxy_list):
                x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = self.class_lookups[role].get(str(class_id), str(class_id)).lower()
                if class_name not in SUPPORTED_TRACKING_CLASSES:
                    continue
                confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
                bbox_width = max(0.0, x2 - x1)
                bbox_height = max(0.0, y2 - y1)
                crop_padding_ratio = 0.05 if class_name != "person" else 0.0
                pad_x = bbox_width * crop_padding_ratio
                pad_y = bbox_height * crop_padding_ratio
                ix1 = max(0, int(round(x1 - pad_x)))
                iy1 = max(0, int(round(y1 - pad_y)))
                ix2 = min(frame_width, int(round(x2 + pad_x)))
                iy2 = min(frame_height, int(round(y2 + pad_y)))
                crop_path = ""
                if self.config.save_crops and ix2 > ix1 and iy2 > iy1:
                    crop = original_image[iy1:iy2, ix1:ix2]
                    crop_name = f"{frame_item['frame_id']}_{role}_{index + 1:03d}_{class_name}.jpg"
                    crop_output_path = self.crops_dir / crop_name
                    if cv2.imwrite(str(crop_output_path), crop):
                        crop_path = str(Path(self.crop_dir_name) / crop_name).replace("\\", "/")
                bbox_area = bbox_width * bbox_height
                bbox_area_ratio = bbox_area / float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                bbox_xywh = [float(value) for value in xywh_list[index]] if index < len(xywh_list) else [0.0, 0.0, 0.0, 0.0]
                detection_payload = {
                    "detection_id": f"{frame_item['frame_id']}_{role}_{index + 1:03d}",
                    "model_role": role,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [round(value, 3) for value in [x1, y1, x2, y2]],
                    "bbox_xywh": [round(float(value), 3) for value in bbox_xywh],
                    "bbox_area_ratio": round(bbox_area_ratio, 6),
                    "frame_id": frame_item["frame_id"],
                    "frame_idx": frame_item["frame_idx"],
                    "timestamp_seconds": frame_item["timestamp_seconds"],
                    "timestamp_text": frame_item["timestamp_text"],
                    "image_path": frame_item["image_path"],
                    "crop_path": crop_path,
                    "crop_exists": bool(crop_path),
                    "tracking_state": frame_item["tracking_state"],
                    "target_fps": frame_item["target_fps"],
                }
                frame_payload["detections"].append(detection_payload)
        elapsed = time.perf_counter() - started
        if self.config.save_annotated:
            annotated_name = f"{frame_item['frame_id']}.jpg"
            cv2.imwrite(str(self.annotated_dir / annotated_name), annotated_image)
        return frame_payload, list(frame_payload["detections"]), elapsed


def resolve_pipeline_config(
    *,
    conf_threshold: float = DEFAULT_YOLO_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_YOLO_IOU_THRESHOLD,
    track_buffer_seconds: float = 2.0,
    high_confidence: float = 0.25,
    low_confidence: float = 0.10,
    match_threshold: float = 0.80,
    min_track_length: int = DEFAULT_TRACKING_MIN_TRACK_LENGTH,
    save_annotated: bool = True,
    save_crops: bool = True,
) -> PipelineConfig:
    import os

    def _resolve_model_path(raw_value: str | None, default_path: Path | None) -> Path | None:
        if raw_value and raw_value.strip():
            candidate = Path(raw_value.strip()).expanduser()
        elif default_path is not None and default_path.exists():
            candidate = default_path
        else:
            return None
        if not candidate.is_absolute():
            candidate = resolve_case_path(str(candidate))
        return candidate

    person_model_path = _resolve_model_path(os.environ.get(ENV_PERSON_YOLO_MODEL_PATH), DEFAULT_PERSON_MODEL_PATH)
    object_model_path = _resolve_model_path(os.environ.get(ENV_OBJECT_YOLO_MODEL_PATH), DEFAULT_OBJECT_MODEL_PATH)
    fallback_model_path = _resolve_model_path(os.environ.get(ENV_YOLO_MODEL_PATH), None)

    model_specs: list[dict[str, Any]] = []
    if person_model_path is not None:
        model_specs.append({"model_role": "person", "model_path": str(person_model_path)})
    if object_model_path is not None:
        model_specs.append({"model_role": "object_vehicle", "model_path": str(object_model_path)})
    if not model_specs and fallback_model_path is not None:
        model_specs.append({"model_role": "combined", "model_path": str(fallback_model_path)})
    if not model_specs:
        raise ValueError(
            f"At least one YOLO model path must be available through {ENV_PERSON_YOLO_MODEL_PATH}, "
            f"{ENV_OBJECT_YOLO_MODEL_PATH}, or {ENV_YOLO_MODEL_PATH}."
        )

    device_decision = resolve_device(component_name="Dynamic YOLO Tracking", override_env_names=(ENV_YOLO_DEVICE,))
    return PipelineConfig(
        model_specs=model_specs,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
        device=device_decision.ultralytics_device,
        device_reason=device_decision.reason,
        save_annotated=bool(save_annotated),
        save_crops=bool(save_crops),
        track_buffer_seconds=float(track_buffer_seconds),
        high_confidence=float(high_confidence),
        low_confidence=float(low_confidence),
        match_threshold=float(match_threshold),
        min_track_length=int(min_track_length),
    )


def class_group(class_name: str) -> str | None:
    normalized = class_name.lower()
    if normalized == "person":
        return "person"
    if normalized in {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}:
        return "vehicle"
    return None


def detect_on_selected_frames(
    *,
    experiment_dir: Path,
    frame_records: list[dict[str, Any]],
    config: PipelineConfig,
    annotated_dir_name: str,
    crop_dir_name: str,
    report_name: str,
    output_name: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], float]:
    YOLO = _load_yolo_class()
    loaded_models = {item["model_role"]: YOLO(str(item["model_path"])) for item in config.model_specs}
    class_lookups: dict[str, dict[str, str]] = {}
    for role, model in loaded_models.items():
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            class_lookups[role] = {str(key): str(value) for key, value in names.items()}
        else:
            class_lookups[role] = {str(index): str(value) for index, value in enumerate(names)}

    annotated_dir = experiment_dir / annotated_dir_name
    crops_dir = experiment_dir / crop_dir_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    inference_times: list[float] = []
    total_detections = 0
    started = time.perf_counter()

    for frame_item in frame_records:
        relative_image_path = Path(str(frame_item["image_path"]))
        absolute_image_path = (experiment_dir / relative_image_path).resolve()
        original_image = cv2.imread(str(absolute_image_path))
        if original_image is None:
            raise RuntimeError(f"Failed to load scheduled frame image: {absolute_image_path}")
        annotated_image = original_image.copy()
        frame_height, frame_width = original_image.shape[:2]
        frame_payload = {
            "frame_id": frame_item["frame_id"],
            "frame_idx": frame_item["frame_idx"],
            "timestamp_seconds": frame_item["timestamp_seconds"],
            "timestamp_text": frame_item["timestamp_text"],
            "image_path": frame_item["image_path"],
            "tracking_state": frame_item["tracking_state"],
            "target_fps": frame_item["target_fps"],
            "selection_reason": list(frame_item.get("selection_reason", [])),
            "detections": [],
        }
        state_counts[str(frame_item["tracking_state"])] += 1
        frame_inference_started = time.perf_counter()
        for model_spec in config.model_specs:
            role = str(model_spec["model_role"])
            results = loaded_models[role].predict(
                source=str(absolute_image_path),
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                device=config.device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            annotated_image = result.plot()
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_list = boxes.xyxy.tolist()
            xywh_list = boxes.xywh.tolist() if getattr(boxes, "xywh", None) is not None else []
            cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            for index, bbox_xyxy in enumerate(xyxy_list):
                x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = class_lookups[role].get(str(class_id), str(class_id)).lower()
                if class_name not in SUPPORTED_TRACKING_CLASSES:
                    continue
                confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
                bbox_width = max(0.0, x2 - x1)
                bbox_height = max(0.0, y2 - y1)
                crop_padding_ratio = 0.05 if class_name != "person" else 0.0
                pad_x = bbox_width * crop_padding_ratio
                pad_y = bbox_height * crop_padding_ratio
                ix1 = max(0, int(round(x1 - pad_x)))
                iy1 = max(0, int(round(y1 - pad_y)))
                ix2 = min(frame_width, int(round(x2 + pad_x)))
                iy2 = min(frame_height, int(round(y2 + pad_y)))
                crop_path = ""
                if config.save_crops and ix2 > ix1 and iy2 > iy1:
                    crop = original_image[iy1:iy2, ix1:ix2]
                    crop_name = f"{frame_item['frame_id']}_{role}_{index + 1:03d}_{class_name}.jpg"
                    crop_output_path = crops_dir / crop_name
                    if cv2.imwrite(str(crop_output_path), crop):
                        crop_path = str(Path(crop_dir_name) / crop_name).replace("\\", "/")
                bbox_area = bbox_width * bbox_height
                bbox_area_ratio = bbox_area / float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                bbox_xywh = [float(value) for value in xywh_list[index]] if index < len(xywh_list) else [0.0, 0.0, 0.0, 0.0]
                detection_payload = {
                    "detection_id": f"{frame_item['frame_id']}_{role}_{index + 1:03d}",
                    "model_role": role,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [round(value, 3) for value in [x1, y1, x2, y2]],
                    "bbox_xywh": [round(float(value), 3) for value in bbox_xywh],
                    "bbox_area_ratio": round(bbox_area_ratio, 6),
                    "frame_id": frame_item["frame_id"],
                    "frame_idx": frame_item["frame_idx"],
                    "timestamp_seconds": frame_item["timestamp_seconds"],
                    "timestamp_text": frame_item["timestamp_text"],
                    "image_path": frame_item["image_path"],
                    "crop_path": crop_path,
                    "crop_exists": bool(crop_path),
                    "tracking_state": frame_item["tracking_state"],
                    "target_fps": frame_item["target_fps"],
                }
                frame_payload["detections"].append(detection_payload)
                detection_rows.append(detection_payload)
                class_counts[class_name] += 1
                total_detections += 1
        inference_times.append(time.perf_counter() - frame_inference_started)
        if config.save_annotated:
            annotated_name = f"{frame_item['frame_id']}.jpg"
            cv2.imwrite(str(annotated_dir / annotated_name), annotated_image)
        frame_results.append(frame_payload)

    elapsed = time.perf_counter() - started
    output_payload = {
        "status": "success",
        "models_used": config.model_specs,
        "yolo_conf_threshold": config.conf_threshold,
        "yolo_iou_threshold": config.iou_threshold,
        "device_used": config.device,
        "frames_processed": len(frame_results),
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "detections": frame_results,
    }
    report_payload = {
        "status": "success",
        "total_video_frames": max((int(item["frame_idx"]) for item in frame_records), default=-1) + 1,
        "frames_processed_by_yolo": len(frame_results),
        "processed_frame_percentage": 0.0,
        "detections_by_class": dict(sorted(class_counts.items())),
        "yolo_inference_time_seconds": round(elapsed, 3),
        "average_yolo_time_per_processed_frame": round(sum(inference_times) / len(inference_times), 6) if inference_times else 0.0,
        "frame_counts_per_controller_state": dict(sorted(state_counts.items())),
        "per_frame_inference_time_seconds": [round(value, 6) for value in inference_times],
    }
    return output_payload, report_payload, detection_rows, elapsed


def run_variable_gap_bytetrack(
    *,
    frame_records: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    reference_fps: float,
    track_buffer_seconds: float,
    high_confidence: float,
    low_confidence: float,
    match_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    BYTETracker, TrackState = load_bytetrack_types()

    class InstrumentedBYTETracker(BYTETracker):
        def __init__(self, args: Any):
            super().__init__(args)
            self.refind_count = 0

        def _apply_match(self, track: Any, det: Any, activated: list[Any], refind: list[Any]) -> None:
            if track.state != TrackState.Tracked:
                self.refind_count += 1
            super()._apply_match(track, det, activated, refind)

    tracker_args = type(
        "TrackerArgs",
        (),
        {
            "track_high_thresh": float(high_confidence),
            "track_low_thresh": float(low_confidence),
            "new_track_thresh": float(high_confidence),
            "match_thresh": float(match_threshold),
            "track_buffer": max(1, int(round(track_buffer_seconds * reference_fps))),
            "fuse_score": False,
            "model": "manual",
        },
    )()
    trackers = {
        "vehicle": InstrumentedBYTETracker(tracker_args),
        "person": InstrumentedBYTETracker(tracker_args),
    }
    grouped_by_frame = defaultdict(list)
    for row in detection_rows:
        grouped_by_frame[str(row["frame_id"])].append(row)

    frame_records = sorted(frame_records, key=lambda item: int(item["frame_idx"]))
    track_histories: dict[str, dict[str, Any]] = {}
    virtual_empty_updates = 0

    def _empty_detection_set() -> DetectionSet:
        return DetectionSet(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    last_timestamp_seconds: float | None = None
    for frame in frame_records:
        timestamp_seconds = float(frame["timestamp_seconds"])
        if last_timestamp_seconds is not None:
            gap_seconds = max(0.0, timestamp_seconds - last_timestamp_seconds)
            virtual_gap = max(0, int(round(gap_seconds * reference_fps)) - 1)
            for _ in range(virtual_gap):
                virtual_empty_updates += 1
                for tracker in trackers.values():
                    tracker.update(_empty_detection_set())
        frame_rows = grouped_by_frame.get(str(frame["frame_id"]), [])
        grouped = {"vehicle": [], "person": []}
        for row in frame_rows:
            group = class_group(str(row["class_name"]))
            if group in grouped:
                grouped[group].append(row)
        for group_name, group_rows in grouped.items():
            if group_rows:
                xyxy = np.asarray([item["bbox_xyxy"] for item in group_rows], dtype=np.float32)
                conf = np.asarray([item["confidence"] for item in group_rows], dtype=np.float32)
                cls = np.asarray([item["class_id"] for item in group_rows], dtype=np.float32)
                detections = DetectionSet(xyxy, conf, cls)
            else:
                detections = _empty_detection_set()
            tracked = trackers[group_name].update(detections)
            if not group_rows:
                continue
            for row in tracked.tolist():
                x1, y1, x2, y2, numeric_track_id, score, _cls_value, detection_index = row
                source_detection = group_rows[int(detection_index)]
                previous_processed_timestamp = timestamp_seconds if last_timestamp_seconds is None else float(last_timestamp_seconds)
                delta_seconds = max(0.0, float(timestamp_seconds) - previous_processed_timestamp)
                track_id = f"{group_name}_track_{int(numeric_track_id):04d}"
                history = track_histories.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "track_type": group_name,
                        "source": "dynamic_bytetrack_raw",
                        "detections": [],
                    },
                )
                history["detections"].append(
                    {
                        **source_detection,
                        "bbox_xyxy": [
                            round(max(0.0, min(float(x1), float(image_width))), 3),
                            round(max(0.0, min(float(y1), float(image_height))), 3),
                            round(max(0.0, min(float(x2), float(image_width))), 3),
                            round(max(0.0, min(float(y2), float(image_height))), 3),
                        ],
                        "match_score": round(float(score), 6),
                        "previous_processed_timestamp": round(previous_processed_timestamp, 6),
                        "current_timestamp": round(float(timestamp_seconds), 6),
                        "delta_seconds": round(delta_seconds, 6),
                    }
                )
        last_timestamp_seconds = timestamp_seconds

    raw_tracks = sorted(track_histories.values(), key=lambda item: float(item["detections"][0]["timestamp_seconds"]))
    return raw_tracks, {
        "track_buffer_frames": int(tracker_args.track_buffer),
        "track_high_thresh": float(tracker_args.track_high_thresh),
        "track_low_thresh": float(tracker_args.track_low_thresh),
        "match_thresh": float(tracker_args.match_thresh),
        "vehicle_refind_count": trackers["vehicle"].refind_count,
        "person_refind_count": trackers["person"].refind_count,
        "virtual_empty_updates": virtual_empty_updates,
        "variable_gap_handling": "Inserted empty tracker updates for skipped virtual frames using the selected frame timestamp gap and the reference FPS.",
    }, {
        "active_vehicle_tracks": len([item for item in raw_tracks if item["track_type"] == "vehicle"]),
        "active_person_tracks": len([item for item in raw_tracks if item["track_type"] == "person"]),
    }


def build_preview_video(
    *,
    experiment_dir: Path,
    preview_name: str,
    frame_records: list[dict[str, Any]],
    merged_tracks_payload: dict[str, Any],
    burst_timestamps: set[float],
) -> None:
    track_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    active_count_by_frame: Counter[str] = Counter()
    for track in list(merged_tracks_payload.get("tracks", [])):
        for detection in list(track.get("detections", [])):
            frame_id = str(detection["frame_id"])
            active_count_by_frame[frame_id] += 1
            track_lookup[(frame_id, str(detection["detection_id"]))] = {
                "track_id": track["track_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox_xyxy": detection["bbox_xyxy"],
                "tracking_state": detection.get("tracking_state", frame_records[0]["tracking_state"] if frame_records else "NORMAL"),
                "target_fps": detection.get("target_fps", 0.0),
                "timestamp_seconds": detection["timestamp_seconds"],
            }

    if not frame_records:
        return
    first_image = cv2.imread(str((experiment_dir / frame_records[0]["image_path"]).resolve()))
    if first_image is None:
        return
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(str(experiment_dir / preview_name), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for frame in frame_records:
        image = cv2.imread(str((experiment_dir / frame["image_path"]).resolve()))
        if image is None:
            continue
        frame_id = str(frame["frame_id"])
        detections = [item for key, item in track_lookup.items() if key[0] == frame_id]
        for item in detections:
            x1, y1, x2, y2 = [int(round(float(value))) for value in item["bbox_xyxy"]]
            color = (40, 200, 40) if item["class_name"] != "person" else (200, 120, 20)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{item['track_id']} {item['class_name']} {float(item['confidence']):.2f}"
            cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        overlay_lines = [
            f"t={float(frame['timestamp_seconds']):.2f}s",
            f"state={frame['tracking_state']}",
            f"target_fps={float(frame['target_fps']):.1f}",
            f"active_tracks={int(active_count_by_frame.get(frame_id, 0))}",
        ]
        if round(float(frame["timestamp_seconds"]), 6) in burst_timestamps:
            overlay_lines.append("BURST_TRIGGER")
        for index, line in enumerate(overlay_lines):
            cv2.putText(image, line, (16, 28 + (index * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()
