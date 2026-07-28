from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import cv2

from ..detection.detection_config import load_detection_config
from ..detection.detection_models import DetectionPacket, VehicleDetection
from ..detection.vehicle_detector import SharedVehicleDetector, normalize_vehicle_class
from ..enrichment.anpr_config import load_anpr_config
from ..enrichment.best_plate_selector import select_best_plate_candidates
from ..enrichment.florence_config import load_florence_config
from ..enrichment.florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from ..enrichment.plate_candidate_collector import PlateCandidateCollector
from ..enrichment.plate_ocr_aggregation import aggregate_ocr_attempts
from ..enrichment.vehicle_evidence_selector import select_vehicle_evidence_candidates
from ..evidence.evidence_config import load_evidence_config
from ..evidence.track_evidence_collector import TrackEvidenceCollector
from ..ingestion.camera_config import apply_file_source_timestamp_policy, load_camera_configs
from ..ingestion.frame_packet import FramePacket
from ..models.florence_runtime_factory import FlorenceRuntimeFactory
from ..models.plate_detector_runtime_factory import PlateDetectorRuntimeFactory
from ..persistence.persistence_config import load_persistence_config
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.track_lifecycle import build_track_uuid
from ..tracking.tracking_config import load_tracking_config
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation


STRONG_OVERLAP_IOU = 0.70


@dataclass(frozen=True, slots=True)
class AspectRatioClassRange:
    min_ratio: float | None = None
    max_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class AspectRatioValidationConfig:
    enabled: bool = True
    classes: dict[str, AspectRatioClassRange] | None = None
    edge_tolerance_multiplier: float = 1.50
    partial_visibility_tolerance_multiplier: float = 1.50
    action: str = "report_only"

    def range_for(self, class_name: str) -> AspectRatioClassRange | None:
        if not self.classes:
            return None
        return self.classes.get(str(class_name).strip().lower())


@dataclass(frozen=True, slots=True)
class AspectRatioEvaluation:
    aspect_ratio: float | None
    minimum: float | None
    maximum: float | None
    deviation: float | None
    status: str


def calculate_aspect_ratio(width: float, height: float) -> float:
    if float(height) <= 0.0:
        raise ValueError("bbox height must be positive for aspect-ratio calculation.")
    return float(width) / float(height)


def evaluate_aspect_ratio(
    *,
    class_name: str,
    width: float,
    height: float,
    touches_edge: bool,
    partial_visibility_ratio: float,
    config: AspectRatioValidationConfig,
) -> AspectRatioEvaluation:
    ratio = calculate_aspect_ratio(width, height)
    class_range = config.range_for(class_name)
    if class_range is None or (class_range.min_ratio is None and class_range.max_ratio is None):
        return AspectRatioEvaluation(ratio, None, None, None, "RANGE_NOT_CONFIGURED")
    multiplier = 1.0
    if touches_edge:
        multiplier = max(multiplier, float(config.edge_tolerance_multiplier))
    if partial_visibility_ratio < 1.0:
        multiplier = max(multiplier, float(config.partial_visibility_tolerance_multiplier))
    minimum = None if class_range.min_ratio is None else float(class_range.min_ratio) / multiplier
    maximum = None if class_range.max_ratio is None else float(class_range.max_ratio) * multiplier
    if minimum is not None and ratio < minimum:
        status = "EDGE_TOLERATED" if touches_edge or partial_visibility_ratio < 1.0 else "BELOW_MINIMUM"
        deviation = minimum - ratio
        return AspectRatioEvaluation(ratio, minimum, maximum, deviation, status)
    if maximum is not None and ratio > maximum:
        status = "EDGE_TOLERATED" if touches_edge or partial_visibility_ratio < 1.0 else "ABOVE_MAXIMUM"
        deviation = ratio - maximum
        return AspectRatioEvaluation(ratio, minimum, maximum, deviation, status)
    return AspectRatioEvaluation(ratio, minimum, maximum, 0.0, "WITHIN_RANGE")


def should_gate_new_track(
    *,
    action: str,
    detection_confidence: float,
    aspect_ratio_status: str,
    is_new_track: bool,
    strong_continuation: bool,
) -> bool:
    if action != "new_track_gate":
        return False
    if strong_continuation:
        return False
    if not is_new_track:
        return False
    return detection_confidence < 0.50 and aspect_ratio_status in {"BELOW_MINIMUM", "ABOVE_MAXIMUM"}


@dataclass(slots=True)
class ExperimentRunConfig:
    label: str
    confidence_threshold: float
    match_threshold: float
    minimum_consecutive_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAM_002 tracking confidence experiment with diagnostics.")
    parser.add_argument("--camera-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\cameras.yaml")
    parser.add_argument("--detection-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\detection.yaml")
    parser.add_argument("--tracking-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\tracking.yaml")
    parser.add_argument("--worker-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\workers.yaml")
    parser.add_argument("--persistence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\persistence.yaml")
    parser.add_argument("--evidence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\evidence.yaml")
    parser.add_argument("--anpr-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\anpr.yaml")
    parser.add_argument("--florence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\florence.yaml")
    parser.add_argument("--camera-code", default="CAM_002")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--confidence-values", nargs="+", type=float, default=[0.25, 0.35, 0.40, 0.50, 0.60])
    parser.add_argument("--output-root", default="debug_runs\\multicamera_vehicle_tracking_pipeline\\confidence_sweep")
    parser.add_argument("--skip-anpr", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None)
    return parser.parse_args()


def resolve_experiment_video(*, camera_config_path: Path, camera_code: str, video_path: str | None) -> tuple[Path, str]:
    if video_path:
        return Path(video_path).expanduser().resolve(), camera_code
    cameras = load_camera_configs(camera_config_path, include_disabled=True, validate_paths=False)
    for camera in cameras:
        if camera.camera_code == camera_code:
            return camera.source_path, camera.camera_code
    raise ValueError(f"Camera code not found in camera config: {camera_code}")


def probe_video(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration = frame_count / fps if fps > 0 else None
    return {
        "video_path": str(video_path),
        "source_fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "frame_width": width,
        "frame_height": height,
    }


def _camera_timestamp_for_frame(frame_number: int, fps: float, started_at: datetime) -> datetime:
    return started_at + timedelta(seconds=(frame_number / max(fps, 1.0)))


def _build_frame_packet(
    *,
    camera_code: str,
    camera_name: str,
    source_path: Path,
    frame_number: int,
    source_fps: float,
    source_frame_count: int,
    frame: Any,
    started_at: datetime,
) -> FramePacket:
    return FramePacket(
        camera_code=camera_code,
        camera_name=camera_name,
        source_path=source_path,
        frame_number=frame_number,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        video_time_seconds=frame_number / max(source_fps, 1.0),
        camera_timestamp=_camera_timestamp_for_frame(frame_number, source_fps, started_at),
        frame=frame,
    )


def _raw_predictions_for_frame(detector: SharedVehicleDetector, frame: Any) -> list[dict[str, Any]]:
    predictions = detector.model.predict(
        source=frame,
        conf=detector.config.confidence_threshold,
        iou=detector.config.iou_threshold,
        imgsz=detector.config.image_size,
        device=detector.device,
        verbose=False,
    )
    if not predictions:
        return []
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    xyxy_values = boxes.xyxy.tolist()
    cls_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
    conf_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    rows: list[dict[str, Any]] = []
    for index, raw_box in enumerate(xyxy_values):
        class_id = int(cls_values[index]) if index < len(cls_values) else 0
        confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
        raw_class_name = detector._class_names.get(class_id, str(class_id))  # noqa: SLF001
        rows.append(
            {
                "detection_index": index,
                "raw_class_id": class_id,
                "raw_class_name": raw_class_name,
                "normalized_class_name": normalize_vehicle_class(raw_class_name),
                "confidence": confidence,
                "bbox_xyxy": [float(value) for value in raw_box[:4]],
            }
        )
    return rows


def _build_detection_packet(
    *,
    detector: SharedVehicleDetector,
    frame_packet: FramePacket,
) -> tuple[DetectionPacket, list[dict[str, Any]], int]:
    started = time.perf_counter()
    predictions = detector.model.predict(
        source=frame_packet.frame,
        conf=detector.inference_confidence_floor,
        iou=detector.config.iou_threshold,
        imgsz=detector.config.image_size,
        device=detector.device,
        verbose=False,
    )
    raw_rows = _raw_predictions_from_predictions(detector=detector, predictions=predictions)
    detections = detector._convert_predictions(predictions, frame_packet)  # noqa: SLF001
    inference_time_ms = (time.perf_counter() - started) * 1000.0
    frame_height, frame_width = frame_packet.frame.shape[:2]
    return (
        DetectionPacket(
            camera_code=frame_packet.camera_code,
            camera_name=frame_packet.camera_name,
            source_path=frame_packet.source_path,
            frame_number=frame_packet.frame_number,
            video_time_seconds=frame_packet.video_time_seconds,
            camera_timestamp=frame_packet.camera_timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            detections=detections,
            inference_time_ms=inference_time_ms,
            detector_model=detector.loaded_model_name or detector.config.model_path,
            detector_device=detector.device,
            source_fps=frame_packet.source_fps,
            frame=frame_packet.frame,
        ),
        raw_rows,
        len(raw_rows) - len(detections),
    )


def _raw_predictions_from_predictions(*, detector: SharedVehicleDetector, predictions: Any) -> list[dict[str, Any]]:
    if not predictions:
        return []
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    xyxy_values = boxes.xyxy.tolist()
    cls_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
    conf_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    rows: list[dict[str, Any]] = []
    for index, raw_box in enumerate(xyxy_values):
        class_id = int(cls_values[index]) if index < len(cls_values) else 0
        confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
        raw_class_name = detector._class_names.get(class_id, str(class_id))  # noqa: SLF001
        normalized_class_name = normalize_vehicle_class(raw_class_name)
        accepted_by_class_threshold, class_threshold, rejection_reason = detector.evaluate_class_threshold(
            normalized_class_name=normalized_class_name,
            confidence=confidence,
        )
        rows.append(
            {
                "detection_index": index,
                "raw_class_id": class_id,
                "raw_class_name": raw_class_name,
                "normalized_class_name": normalized_class_name,
                "confidence": confidence,
                "global_inference_floor": detector.inference_confidence_floor,
                "class_threshold": class_threshold,
                "accepted_by_class_threshold": accepted_by_class_threshold,
                "rejection_reason": rejection_reason,
                "bbox_xyxy": [float(value) for value in raw_box[:4]],
            }
        )
    return rows


def _bbox_metrics(bbox_xyxy: Sequence[float], *, frame_width: int, frame_height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)
    touches_left_edge = x1 <= 0.5
    touches_right_edge = x2 >= float(frame_width) - 0.5
    touches_top_edge = y1 <= 0.5
    touches_bottom_edge = y2 >= float(frame_height) - 0.5
    outside_left = max(0.0, -x1)
    outside_top = max(0.0, -y1)
    outside_right = max(0.0, x2 - float(frame_width))
    outside_bottom = max(0.0, y2 - float(frame_height))
    outside_area_estimate = max(0.0, outside_left + outside_top + outside_right + outside_bottom)
    partial_visibility_ratio = 1.0
    if area > 0.0:
        clipped_width = max(0.0, min(x2, float(frame_width)) - max(x1, 0.0))
        clipped_height = max(0.0, min(y2, float(frame_height)) - max(y1, 0.0))
        partial_visibility_ratio = max(0.0, min(1.0, (clipped_width * clipped_height) / area))
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_center_x": center_x,
        "bbox_center_y": center_y,
        "touches_left_edge": touches_left_edge,
        "touches_right_edge": touches_right_edge,
        "touches_top_edge": touches_top_edge,
        "touches_bottom_edge": touches_bottom_edge,
        "touches_edge": touches_left_edge or touches_right_edge or touches_top_edge or touches_bottom_edge,
        "outside_frame_linear_estimate": outside_area_estimate,
        "partial_visibility_ratio": partial_visibility_ratio,
    }


def calculate_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def associate_detection(
    *,
    detection_bbox: Sequence[float],
    observations: Sequence[TrackObservation],
) -> TrackObservation | None:
    best: tuple[float, TrackObservation] | None = None
    for observation in observations:
        score = calculate_iou(detection_bbox, observation.bbox_xyxy)
        if best is None or score > best[0]:
            best = (score, observation)
    if best is None or best[0] <= 0.0:
        return None
    return best[1]


def ensure_unique_run_directory(root: Path, label: str) -> Path:
    candidate = root / label
    index = 1
    while candidate.exists():
        index += 1
        candidate = root / f"{label}_{index:02d}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _format_label_timestamp(timestamp_seconds: float) -> str:
    return f"{float(timestamp_seconds):.2f}s"


def _build_annotation_label(
    *,
    camera_code: str,
    frame_number: int,
    timestamp_seconds: float,
    class_name: str,
    confidence: float,
    native_tracker_id: int | None = None,
    logical_track_id: str | None = None,
) -> str:
    parts = [
        camera_code,
        f"frame {frame_number}",
        _format_label_timestamp(timestamp_seconds),
        f"{class_name} {float(confidence):.2f}",
    ]
    if native_tracker_id is not None:
        parts.append(f"native {native_tracker_id}")
    if logical_track_id:
        parts.append(f"logical {logical_track_id}")
    return " | ".join(parts)


def _annotate_detections(
    frame: Any,
    *,
    rows: Sequence[dict[str, Any]],
    color: tuple[int, int, int],
    label_builder,
) -> Any:
    annotated = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = [int(round(float(value))) for value in row["bbox_xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = label_builder(row)
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return annotated


def _save_frame_variants(
    *,
    frame_packet: FramePacket,
    raw_rows: Sequence[dict[str, Any]],
    native_rows: Sequence[dict[str, Any]],
    logical_rows: Sequence[dict[str, Any]],
    frame_dir: Path,
    sample_output_root: Path | None = None,
    sample_index: int | None = None,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_full = frame_dir / "raw_full_frame.jpg"
    raw_yolo = frame_dir / "raw_yolo_detections.jpg"
    native_frame = frame_dir / "native_tracker_output.jpg"
    logical_frame = frame_dir / "logical_track_output.jpg"
    cv2.imwrite(str(raw_full), frame_packet.frame)
    raw_yolo_frame = _annotate_detections(
        frame_packet.frame,
        rows=raw_rows,
        color=(0, 255, 0),
        label_builder=lambda row: _build_annotation_label(
            camera_code=frame_packet.camera_code,
            frame_number=frame_packet.frame_number,
            timestamp_seconds=frame_packet.video_time_seconds,
            class_name=str(row["normalized_class_name"] or row["raw_class_name"]).upper(),
            confidence=float(row["confidence"]),
        ),
    )
    cv2.imwrite(str(raw_yolo), raw_yolo_frame)
    native_tracker_frame = _annotate_detections(
        frame_packet.frame,
        rows=native_rows,
        color=(0, 255, 255),
        label_builder=lambda row: _build_annotation_label(
            camera_code=frame_packet.camera_code,
            frame_number=frame_packet.frame_number,
            timestamp_seconds=frame_packet.video_time_seconds,
            class_name=str(row["class_name"]).upper(),
            confidence=float(row["confidence"]),
            native_tracker_id=row["native_tracker_id"],
        ),
    )
    cv2.imwrite(str(native_frame), native_tracker_frame)
    logical_tracker_frame = _annotate_detections(
        frame_packet.frame,
        rows=logical_rows,
        color=(0, 0, 255),
        label_builder=lambda row: _build_annotation_label(
            camera_code=frame_packet.camera_code,
            frame_number=frame_packet.frame_number,
            timestamp_seconds=frame_packet.video_time_seconds,
            class_name=str(row["class_name"]).upper(),
            confidence=float(row["confidence"]),
            native_tracker_id=row["native_tracker_id"],
            logical_track_id=str(row["logical_track_id"]),
        ),
    )
    cv2.imwrite(str(logical_frame), logical_tracker_frame)
    if sample_output_root is None or sample_index is None:
        return
    camera_dir = sample_output_root / frame_packet.camera_code
    detection_dir = camera_dir / "yolo_detections"
    native_dir = camera_dir / "native_tracker_output"
    logical_dir = camera_dir / "logical_track_output"
    raw_dir = camera_dir / "raw_full_frames"
    detection_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)
    logical_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_name = f"sample_{int(sample_index):06d}"
    cv2.imwrite(str(raw_dir / f"{sample_name}.jpg"), frame_packet.frame)
    cv2.imwrite(str(detection_dir / f"{sample_name}.jpg"), raw_yolo_frame)
    cv2.imwrite(str(native_dir / f"{sample_name}.jpg"), native_tracker_frame)
    cv2.imwrite(str(logical_dir / f"{sample_name}.jpg"), logical_tracker_frame)
    sample_payload = {
        "camera_code": frame_packet.camera_code,
        "frame_number": frame_packet.frame_number,
        "video_time_seconds": frame_packet.video_time_seconds,
        "camera_timestamp": frame_packet.camera_timestamp.isoformat() if frame_packet.camera_timestamp is not None else None,
        "raw_rows": list(raw_rows),
        "native_rows": list(native_rows),
        "logical_rows": list(logical_rows),
    }
    (detection_dir / f"{sample_name}.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_track_json(track: LocalVehicleTrack) -> dict[str, Any]:
    class_counts = dict(track.class_observation_counts)
    class_summary = [
        {
            "class_name": class_name,
            "count": count,
            "of_total": track.class_observation_count,
            "summary": f"{class_name.upper()} observed in {count}/{track.class_observation_count} frames",
        }
        for class_name, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    observed_frames = [item.frame_number for item in track.observations]
    gaps = []
    consecutive_runs: list[int] = []
    current_run = 0
    for index in range(1, len(observed_frames)):
        gap = max(0, observed_frames[index] - observed_frames[index - 1] - 1)
        gaps.append(gap)
    for index, frame_number in enumerate(observed_frames):
        if index == 0 or frame_number == observed_frames[index - 1] + 1:
            current_run += 1
        else:
            consecutive_runs.append(current_run)
            current_run = 1
    if current_run > 0:
        consecutive_runs.append(current_run)
    return {
        "track_uuid": track.track_uuid,
        "logical_track_id": track.local_track_id,
        "native_tracker_ids_seen": list(track.native_tracker_ids_seen),
        "number_of_native_tracker_ids": len(track.native_tracker_ids_seen),
        "first_frame": track.first_frame_number,
        "last_frame": track.last_frame_number,
        "observation_count": track.observation_count,
        "raw_class_history": [
            {
                "frame_number": item.frame_number,
                "video_time_seconds": item.video_time_seconds,
                "class_name": item.class_name,
                "raw_class_name": item.raw_class_name,
                "confidence": item.confidence,
                "bbox_xyxy": list(item.bbox_xyxy),
            }
            for item in track.raw_class_history
        ],
        "class_observation_counts": dict(track.class_observation_counts),
        "class_scores": dict(track.class_scores),
        "class_ratios": dict(track.class_ratios),
        "class_max_confidences": dict(track.class_max_confidences),
        "provisional_class": track.provisional_class_name,
        "stable_class": track.stable_class_name,
        "final_class": track.stable_class_name or "unknown",
        "class_status": track.class_status,
        "final_class_reason": track.final_class_reason,
        "class_locked": track.class_is_locked,
        "winner_margin": track.class_winner_margin,
        "total_class_observations": track.class_observation_count,
        "winning_class": track.winning_class_name,
        "winning_class_count": track.winning_class_count,
        "winning_class_ratio": track.winning_class_ratio,
        "winner_confidence_sum": track.winner_confidence_sum,
        "runner_up_class": track.runner_up_class_name,
        "runner_up_count": track.runner_up_class_count,
        "runner_up_ratio": track.runner_up_ratio,
        "winner_count_margin": track.winner_count_margin,
        "winner_score_margin": track.class_winner_margin,
        "count_winner": track.count_winner_class_name,
        "score_winner": track.score_winner_class_name,
        "winner_agreement": track.winners_agree,
        "maximum_consecutive_winner_count": track.maximum_consecutive_winner_count,
        "recent_consecutive_winner_count": track.recent_consecutive_winner_count,
        "recent_class_counts": dict(track.recent_class_counts),
        "recent_winning_class": track.recent_winning_class_name,
        "recent_winning_ratio": track.recent_winning_ratio,
        "class_transition_count": track.class_transition_count,
        "incompatible_transition_count": track.incompatible_class_transition_count,
        "strong_conflict_detected": track.strong_conflict_detected,
        "split_recommended": track.split_recommended,
        "mixed_identity_detected": track.mixed_identity_detected,
        "mixed_identity_classes": list(track.mixed_identity_classes),
        "mixed_identity_start_frame": track.mixed_identity_start_frame,
        "mixed_identity_confidence": track.mixed_identity_confidence,
        "final_class_blocked_due_to_mixed_identity": track.final_class_blocked_due_to_mixed_identity,
        "possible_identity_switch": track.possible_identity_switch,
        "split_executed": bool(track.completion_reason == "identity_split" or track.split_from_track_uuid),
        "split_frame": track.split_frame,
        "split_reason_codes": list(track.split_reason_codes),
        "source_logical_track_id": track.source_logical_track_id,
        "new_logical_track_id": track.new_logical_track_id,
        "split_native_tracker_id": track.split_native_tracker_id,
        "pending_conflict_observation_count": track.pending_conflict_observation_count,
        "stable_class_before_split": track.stable_class_before_split,
        "conflicting_class": track.conflicting_class,
        "average_conflict_confidence": track.average_conflict_confidence,
        "bbox_iou_at_split": track.bbox_iou_at_split,
        "center_distance_at_split": track.center_distance_at_split,
        "normalized_center_distance_at_split": track.normalized_center_distance_at_split,
        "area_ratio_at_split": track.area_ratio_at_split,
        "width_ratio_at_split": track.width_ratio_at_split,
        "height_ratio_at_split": track.height_ratio_at_split,
        "spatial_score_at_split": track.spatial_score_at_split,
        "class_persistence_summary": class_summary,
        "duration_frames": (track.last_frame_number - track.first_frame_number + 1) if track.last_frame_number >= track.first_frame_number else 0,
        "duration_seconds": max(0.0, track.last_video_time_seconds - track.first_video_time_seconds),
        "consecutive_observation_runs": consecutive_runs,
        "longest_consecutive_observation_run": max(consecutive_runs, default=0),
        "number_of_missed_frames": sum(gaps),
        "maximum_consecutive_missing_frames": max([0, *gaps, track.maximum_consecutive_missing_frames]),
        "number_of_reactivations": track.reactivation_count,
        "number_of_fragment_relinks": track.fragment_relink_count,
        "number_of_logical_splits": 1 if track.split_from_track_uuid else 0,
        "split_from_track_uuid": track.split_from_track_uuid,
        "completion_reason": track.completion_reason,
        "state": track.state,
        "evidence": track.evidence_package.to_dict() if track.evidence_package is not None else None,
    }


def _calculate_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "p05", "p10", "median", "p90", "p95", "maximum")}
    sorted_values = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_values[lower]
        fraction = position - lower
        return sorted_values[lower] + ((sorted_values[upper] - sorted_values[lower]) * fraction)

    return {
        "minimum": sorted_values[0],
        "p05": percentile(0.05),
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "maximum": sorted_values[-1],
    }


def _group_aspect_ratio_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_class: dict[str, list[float]] = {}
    for row in rows:
        class_name = str(row.get("raw_class_name") or row.get("class_name") or "unknown").lower()
        aspect_ratio = row.get("aspect_ratio")
        if aspect_ratio in (None, ""):
            continue
        per_class.setdefault(class_name, []).append(float(aspect_ratio))
    return {
        class_name: {"count": len(values), **_calculate_percentiles(values)}
        for class_name, values in sorted(per_class.items())
    }


def _selected_track_rows(rows: Sequence[dict[str, Any]], *, minimum_confidence: float | None = None, non_edge_only: bool = False, confirmed_only: bool = False) -> list[dict[str, Any]]:
    selected = list(rows)
    if minimum_confidence is not None:
        selected = [row for row in selected if float(row.get("detection_confidence") or 0.0) >= float(minimum_confidence)]
    if non_edge_only:
        selected = [row for row in selected if not bool(row.get("touches_left_edge") or row.get("touches_right_edge") or row.get("touches_top_edge") or row.get("touches_bottom_edge"))]
    if confirmed_only:
        selected = [row for row in selected if str(row.get("lifecycle_state", "")) == "active"]
    return selected


def _count_class_occurrences(rows: Sequence[dict[str, Any]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "").strip().lower()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _average_confidence_by_class(rows: Sequence[dict[str, Any]], *, key: str = "normalized_class_name") -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        class_name = str(row.get(key) or "").strip().lower()
        if not class_name:
            continue
        sums[class_name] = sums.get(class_name, 0.0) + float(row.get("confidence") or row.get("detection_confidence") or 0.0)
        counts[class_name] = counts.get(class_name, 0) + 1
    return {class_name: (sums[class_name] / counts[class_name]) for class_name in sorted(counts)}


def _minimum_confidence_by_class(rows: Sequence[dict[str, Any]], *, key: str = "normalized_class_name") -> dict[str, float]:
    minimums: dict[str, float] = {}
    for row in rows:
        class_name = str(row.get(key) or "").strip().lower()
        if not class_name:
            continue
        confidence = float(row.get("confidence") or row.get("detection_confidence") or 0.0)
        minimums[class_name] = min(minimums.get(class_name, confidence), confidence)
    return dict(sorted(minimums.items()))


def _build_frame_level_report(
    *,
    frame_number: int,
    timestamp_seconds: float,
    raw_rows: Sequence[dict[str, Any]],
    filtered_detection_count: int,
    native_tracker_count: int,
    logical_tracker_count: int,
) -> dict[str, Any]:
    return {
        "frame_number": frame_number,
        "timestamp_seconds": timestamp_seconds,
        "raw_detection_count": len(raw_rows),
        "filtered_detection_count": filtered_detection_count,
        "native_tracker_rows": native_tracker_count,
        "logical_rows": logical_tracker_count,
        "detections_below_035": sum(1 for row in raw_rows if float(row.get("confidence") or 0.0) < 0.35),
        "detections_below_050": sum(1 for row in raw_rows if float(row.get("confidence") or 0.0) < 0.50),
    }


def _build_detection_override(confidence_threshold: float, *, class_confidence_thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"confidence_threshold": float(confidence_threshold)}
    if class_confidence_thresholds is not None:
        payload["class_confidence_thresholds"] = class_confidence_thresholds
    return payload


def _build_evidence_index(run_dir: Path, tracks: Sequence[LocalVehicleTrack]) -> None:
    rows: list[str] = []
    for track in sorted(tracks, key=lambda item: (item.first_frame_number, item.local_track_id)):
        candidate_html = []
        if track.evidence_package is not None:
            for key in ("best_overall", "largest", "highest_confidence"):
                candidate = track.evidence_package.candidates.get(key)
                if candidate is None or candidate.file_path is None:
                    continue
                candidate_path = Path(candidate.file_path)
                relative = candidate_path.resolve().relative_to(run_dir.resolve()) if candidate_path.exists() and candidate_path.resolve().is_relative_to(run_dir.resolve()) else None
                href = relative.as_posix() if relative is not None else candidate.file_path.replace("\\", "/")
                candidate_html.append(f"<div><strong>{html.escape(key)}</strong><br><img src=\"{html.escape(href)}\" width=\"240\"></div>")
        rows.append(
            "<section>"
            f"<h2>{html.escape(track.track_uuid)}</h2>"
            f"<p>stable class: {html.escape(str(track.stable_class_name or track.class_name))} | "
            f"first frame: {track.first_frame_number} | last frame: {track.last_frame_number} | "
            f"observations: {track.observation_count}</p>"
            f"<p>raw class history: {html.escape(', '.join(f'{item.class_name}@{item.frame_number}' for item in track.raw_class_history[:20]))}</p>"
            f"{''.join(candidate_html)}"
            "</section>"
        )
    html_payload = (
        "<html><head><meta charset='utf-8'><title>Tracking evidence index</title></head><body>"
        f"<h1>{html.escape(run_dir.name)}</h1>{''.join(rows)}</body></html>"
    )
    (run_dir / "evidence_index.html").write_text(html_payload, encoding="utf-8")


def _track_class_conflict_counts(tracks_json: Sequence[dict[str, Any]]) -> dict[str, int]:
    car_bus = 0
    car_truck = 0
    mixed = 0
    for item in tracks_json:
        classes = set((item.get("class_observation_counts") or {}).keys())
        if len(classes) > 1:
            mixed += 1
        if {"car", "bus"}.issubset(classes):
            car_bus += 1
        if {"car", "truck"}.issubset(classes):
            car_truck += 1
    return {
        "mixed_tracks": mixed,
        "tracks_with_car_and_bus": car_bus,
        "tracks_with_car_and_truck": car_truck,
    }


def _build_class_consistency_payload(tracks_json: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "track_uuid": item["track_uuid"],
            "observations": item["observation_count"],
            "winning_class": item.get("winning_class"),
            "winning_class_ratio": item.get("winning_class_ratio"),
            "maximum_consecutive_winner_count": item.get("maximum_consecutive_winner_count"),
            "runner_up_class": item.get("runner_up_class"),
            "stable_class": item.get("stable_class"),
            "class_status": item.get("class_status"),
            "strong_conflict_detected": item.get("strong_conflict_detected"),
            "split_recommended": item.get("split_recommended"),
            "split_executed": item.get("split_executed"),
            "mixed_identity_detected": item.get("mixed_identity_detected"),
            "mixed_identity_classes": item.get("mixed_identity_classes"),
            "count_winner": item.get("count_winner"),
            "score_winner": item.get("score_winner"),
            "winner_agreement": item.get("winner_agreement"),
            "class_sequence": [
                {
                    "frame_number": row["frame_number"],
                    "class_name": row["class_name"],
                    "confidence": row["confidence"],
                    "bbox_xyxy": row["bbox_xyxy"],
                }
                for row in item.get("raw_class_history", [])
            ],
        }
        for item in tracks_json
    ]
    return {
        "tracks": rows,
        "consistent_tracks": [row for row in rows if row["class_status"] in {"CONSISTENT", "LOCKED"}],
        "ambiguous_tracks": [row for row in rows if row["class_status"] == "AMBIGUOUS"],
        "short_tracks": [row for row in rows if row["class_status"] == "INSUFFICIENT_OBSERVATIONS"],
        "mixed_car_truck_tracks": [
            row for row, item in zip(rows, tracks_json)
            if {"car", "truck"}.issubset(set((item.get("class_observation_counts") or {}).keys()))
        ],
        "mixed_car_bus_tracks": [
            row for row, item in zip(rows, tracks_json)
            if {"car", "bus"}.issubset(set((item.get("class_observation_counts") or {}).keys()))
        ],
        "count_score_disagreement_tracks": [row for row in rows if not bool(row["winner_agreement"])],
        "split_recommended_tracks": [row for row in rows if bool(row["split_recommended"])],
        "mixed_identity_tracks": [row for row in rows if bool(row["mixed_identity_detected"])],
    }


def _class_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| Track | Observations | Winning class | Ratio | Max consecutive | Runner-up | Stable class | Status | Conflict |",
        "| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["tracks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["track_uuid"]),
                    str(row["observations"]),
                    str(row["winning_class"]),
                    f"{float(row['winning_class_ratio'] or 0.0):.3f}" if row.get("winning_class_ratio") is not None else "",
                    str(row["maximum_consecutive_winner_count"]),
                    str(row["runner_up_class"]),
                    str(row["stable_class"]),
                    str(row["class_status"]),
                    "yes" if row["strong_conflict_detected"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _manual_evaluation_sheet(run_dir: Path, tracks: Sequence[LocalVehicleTrack]) -> None:
    rows = [
        {
            "Object": "motorcycle",
            "Expected real objects": 1,
            "Logical tracks created": "",
            "Final classes": "",
            "Best track duration": "",
            "Duplicate IDs": "",
            "Mixed-object track": "",
            "Wrong plate association": "n/a",
            "Candidate artifact paths": "",
        },
        {
            "Object": "white WagonR",
            "Expected real objects": 1,
            "Logical tracks created": "",
            "Final classes": "",
            "Best track duration": "",
            "Duplicate IDs": "",
            "Mixed-object track": "",
            "Wrong plate association": "",
            "Candidate artifact paths": "",
        },
        {
            "Object": "white Honda",
            "Expected real objects": 1,
            "Logical tracks created": "",
            "Final classes": "",
            "Best track duration": "",
            "Duplicate IDs": "",
            "Mixed-object track": "",
            "Wrong plate association": "",
            "Candidate artifact paths": "",
        },
        {
            "Object": "truck",
            "Expected real objects": 1,
            "Logical tracks created": "",
            "Final classes": "",
            "Best track duration": "",
            "Duplicate IDs": "",
            "Mixed-object track": "",
            "Wrong plate association": "",
            "Candidate artifact paths": "",
        },
    ]
    track_paths = []
    for track in tracks:
        if track.evidence_package is not None and track.evidence_package.output_directory is not None:
            track_paths.append(track.evidence_package.output_directory)
    for row in rows:
        row["Candidate artifact paths"] = " | ".join(track_paths[:8])
    _write_csv(run_dir / "manual_object_evaluation.csv", rows)


def _anpr_audit_for_tracks(
    *,
    tracks: Sequence[LocalVehicleTrack],
    artifact_root: Path,
    anpr_config_path: Path,
    florence_config_path: Path,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {"status": "skipped", "reason": "skip_anpr_requested", "results": [], "conflicts": []}
    try:
        anpr_config = load_anpr_config(anpr_config_path, overrides={"enabled": True, "persist_result": False})
        florence_config = load_florence_config(florence_config_path, overrides={"enabled": True})
        runtime_factory = FlorenceRuntimeFactory(project_root=Path.cwd())
        plate_factory = PlateDetectorRuntimeFactory(project_root=Path.cwd())
        plate_runtime = plate_factory.get_runtime(config=anpr_config, model_path_cli=None, device_override=None)
        florence_runtime = runtime_factory.get_runtime(config=florence_config, model_path_cli=None, adapter_path_cli=None, processor_path_cli=None, device_override=None)
        if plate_runtime is None or florence_runtime is None:
            return {"status": "unavailable", "reason": "plate_or_florence_runtime_unavailable", "results": [], "conflicts": []}
        collector = PlateCandidateCollector(detector_runtime=plate_runtime, config=anpr_config, artifact_root=artifact_root)
        ocr = FlorencePlateOcrExtractor(runtime=florence_runtime, ocr_config=anpr_config.ocr, validation_config=anpr_config.validation)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "results": [], "conflicts": []}

    results: list[dict[str, Any]] = []
    for track in tracks:
        if track.evidence_package is None:
            continue
        evidence_inputs = select_vehicle_evidence_candidates(
            completed_track=track,
            configured_roles=anpr_config.vehicle_evidence_roles,
            maximum_candidates=anpr_config.maximum_vehicle_crops_per_track,
            artifact_root=artifact_root,
        )
        candidates = collector.collect(evidence_inputs)
        selected = select_best_plate_candidates(candidates, maximum_for_ocr=anpr_config.ocr.maximum_plate_candidates_for_ocr, config=anpr_config.plate_selection)
        attempts_by_candidate_uri = {}
        for candidate in selected:
            attempts_by_candidate_uri[candidate.relative_storage_uri] = list(ocr.extract_attempts(candidate))
        aggregated = aggregate_ocr_attempts(candidates=selected, attempts_by_candidate_uri=attempts_by_candidate_uri, config=anpr_config)
        selected_candidate = aggregated.selected_candidate
        vehicle_candidate = None
        if selected_candidate is not None and track.evidence_package is not None:
            vehicle_candidate = track.evidence_package.candidates.get(selected_candidate.source_vehicle_role.lower())
        inside_ratio = None
        if selected_candidate is not None and vehicle_candidate is not None:
            plate_area = max(1.0, (selected_candidate.plate_bbox_xyxy[2] - selected_candidate.plate_bbox_xyxy[0]) * (selected_candidate.plate_bbox_xyxy[3] - selected_candidate.plate_bbox_xyxy[1]))
            vehicle_area = max(1.0, vehicle_candidate.area)
            inside_ratio = min(1.0, plate_area / vehicle_area)
        results.append(
            {
                "track_uuid": track.track_uuid,
                "vehicle_stabilized_class": track.stable_class_name or track.class_name,
                "plate_text": aggregated.normalized_text,
                "plate_status": aggregated.status,
                "plate_confidence": aggregated.confidence,
                "source_frame": selected_candidate.frame_number if selected_candidate is not None else None,
                "vehicle_frame": vehicle_candidate.frame_number if vehicle_candidate is not None else None,
                "evidence_role": selected_candidate.source_vehicle_role if selected_candidate is not None else None,
                "plate_bbox": list(selected_candidate.plate_bbox_xyxy) if selected_candidate is not None else None,
                "vehicle_bbox": list(vehicle_candidate.original_bbox_xyxy) if vehicle_candidate is not None else None,
                "plate_inside_vehicle_ratio": inside_ratio,
            }
        )
    conflicts = []
    by_plate: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        plate_text = item.get("plate_text")
        if not plate_text or item.get("plate_status") != "VERIFIED":
            continue
        by_plate.setdefault(str(plate_text), []).append(item)
    for plate_text, items in sorted(by_plate.items()):
        classes = {str(item.get("vehicle_stabilized_class")) for item in items if item.get("vehicle_stabilized_class")}
        if len(items) > 1:
            conflicts.append(
                {
                    "plate_text": plate_text,
                    "track_uuids": [item["track_uuid"] for item in items],
                    "vehicle_classes": sorted(classes),
                    "conflicting_classes": len(classes) > 1,
                }
            )
    return {"status": "ok", "results": results, "conflicts": conflicts}


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    headers = [
        "Confidence",
        "Total detections",
        "Logical tracks",
        "Short tracks",
        "Motorcycle tracks",
        "WagonR tracks",
        "Honda tracks",
        "Wrong BUS labels",
        "Mixed tracks",
        "Empty-space boxes",
        "Plate conflicts",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- " for _ in headers]) + "|"]
    for row in comparison["confidence_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["confidence"]),
                    str(row["total_detections"]),
                    str(row["logical_tracks"]),
                    str(row["short_tracks"]),
                    str(row["motorcycle_tracks"]),
                    str(row["wagonr_tracks"]),
                    str(row["honda_tracks"]),
                    str(row["wrong_bus_labels"]),
                    str(row["mixed_tracks"]),
                    str(row["empty_space_boxes"]),
                    str(row["plate_conflicts"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _heuristic_best_run(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        results,
        key=lambda item: (
            int(item["summary"]["mixed_tracks"]),
            int(item["summary"]["empty_space_boxes"]),
            int(item["summary"]["plate_conflicts"]),
            int(item["summary"]["logical_tracks"]),
            int(item["summary"]["short_tracks"]),
            -int(item["summary"]["total_detections"]),
        ),
    )
    return ranked[0]


def _run_single_configuration(
    *,
    run_config: ExperimentRunConfig,
    camera_code: str,
    camera_name: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    detection_config_path: Path,
    tracking_config_path: Path,
    persistence_config_path: Path,
    evidence_config_path: Path,
    run_dir: Path,
    base_run_id: str,
    aspect_ratio_config: AspectRatioValidationConfig,
    skip_anpr: bool,
    anpr_config_path: Path,
    florence_config_path: Path,
    tracking_overrides: dict[str, Any] | None = None,
    detection_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = f"{base_run_id}:{run_config.label}"
    artifact_run_id = run_id.replace(":", "_")
    artifact_root = (run_dir / "artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_detection_overrides = _build_detection_override(run_config.confidence_threshold)
    if detection_overrides:
        resolved_detection_overrides.update(detection_overrides)
    detection_config = load_detection_config(detection_config_path, overrides=resolved_detection_overrides)
    resolved_tracking_overrides = {
        "minimum_consecutive_frames": run_config.minimum_consecutive_frames,
        "minimum_matching_threshold": run_config.match_threshold,
        "match_thresh": run_config.match_threshold,
    }
    if tracking_overrides:
        resolved_tracking_overrides.update(tracking_overrides)
    tracking_config = load_tracking_config(tracking_config_path, overrides=resolved_tracking_overrides)
    load_persistence_config(persistence_config_path)
    evidence_config = load_evidence_config(evidence_config_path, overrides={"output_root": str(artifact_root)})
    detector = SharedVehicleDetector(detection_config)
    router = CameraDetectionRouter(tracking_config, run_id=run_id, allowed_camera_codes=(camera_code,))
    evidence_collector = TrackEvidenceCollector(evidence_config, run_id=artifact_run_id)
    video_info = probe_video(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video for experiment: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    started_at = datetime(2026, 7, 28, 0, 0, 0)
    frame_rows: list[dict[str, Any]] = []
    raw_detection_rows: list[dict[str, Any]] = []
    accepted_detection_rows: list[dict[str, Any]] = []
    rejected_detection_rows: list[dict[str, Any]] = []
    frame_level_reports: list[dict[str, Any]] = []
    completed_tracks: list[LocalVehicleTrack] = []
    invalid_bbox_count = 0
    duplicate_overlaps = 0
    total_yolo_detections = 0
    for frame_number in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame_packet = _build_frame_packet(
            camera_code=camera_code,
            camera_name=camera_name,
            source_path=video_path,
            frame_number=frame_number,
            source_fps=float(video_info["source_fps"]),
            source_frame_count=int(video_info["frame_count"]),
            frame=frame,
            started_at=started_at,
        )
        detection_packet, raw_rows, raw_minus_filtered = _build_detection_packet(detector=detector, frame_packet=frame_packet)
        invalid_bbox_count += max(0, raw_minus_filtered)
        total_yolo_detections += len(raw_rows)
        tracking_result = router.route(detection_packet)
        evidence_collector.update(detection_packet, tracking_result.observations)
        completed_tracks.extend(tracking_result.completed_tracks)
        logical_rows = []
        native_rows = []
        for native in tracking_result.native_observations or []:
            native_rows.append(
                {
                    "bbox_xyxy": list(native.bbox_xyxy),
                    "class_name": native.class_name,
                    "confidence": native.confidence,
                    "native_tracker_id": native.native_tracker_id or native.local_track_id,
                }
            )
        for logical in tracking_result.observations:
            logical_rows.append(
                {
                    "bbox_xyxy": list(logical.bbox_xyxy),
                    "class_name": logical.class_name,
                    "confidence": logical.confidence,
                    "native_tracker_id": logical.native_tracker_id,
                    "logical_track_id": logical.local_track_id,
                }
            )
        frame_dir = run_dir / "diagnostic_frames" / f"frame_{frame_number:06d}"
        _save_frame_variants(
            frame_packet=frame_packet,
            raw_rows=raw_rows,
            native_rows=native_rows,
            logical_rows=logical_rows,
            frame_dir=frame_dir,
            sample_output_root=run_dir / "sample_frames",
            sample_index=(frame_number - start_frame + 1),
        )
        for left_index, left in enumerate(raw_rows):
            for right in raw_rows[left_index + 1 :]:
                if calculate_iou(left["bbox_xyxy"], right["bbox_xyxy"]) >= STRONG_OVERLAP_IOU:
                    duplicate_overlaps += 1
        for row in raw_rows:
            metrics = _bbox_metrics(row["bbox_xyxy"], frame_width=detection_packet.frame_width, frame_height=detection_packet.frame_height)
            normalized_class = row["normalized_class_name"] or row["raw_class_name"]
            aspect = evaluate_aspect_ratio(
                class_name=str(normalized_class),
                width=float(metrics["bbox_width"]),
                height=float(metrics["bbox_height"]) if float(metrics["bbox_height"]) > 0 else 1.0,
                touches_edge=bool(metrics["touches_edge"]),
                partial_visibility_ratio=float(metrics["partial_visibility_ratio"]),
                config=aspect_ratio_config,
            )
            native_match = associate_detection(detection_bbox=row["bbox_xyxy"], observations=tracking_result.native_observations or [])
            logical_match = associate_detection(detection_bbox=row["bbox_xyxy"], observations=tracking_result.observations)
            stabilized_class = logical_match.class_name if logical_match is not None else None
            frame_rows.append(
                {
                    "run_label": run_config.label,
                    "confidence_threshold": run_config.confidence_threshold,
                    "camera_code": camera_code,
                    "frame_number": frame_number,
                    "timestamp_seconds": frame_packet.video_time_seconds,
                    "detection_index": row["detection_index"],
                    "raw_class_id": row["raw_class_id"],
                    "raw_class_name": row["raw_class_name"],
                    "detection_confidence": row["confidence"],
                    "global_inference_floor": row["global_inference_floor"],
                    "class_threshold": row["class_threshold"],
                    "accepted_by_class_threshold": row["accepted_by_class_threshold"],
                    "rejection_reason": row["rejection_reason"],
                    "bbox_xyxy": json.dumps([round(float(value), 3) for value in row["bbox_xyxy"]]),
                    "bbox_width": round(float(metrics["bbox_width"]), 3),
                    "bbox_height": round(float(metrics["bbox_height"]), 3),
                    "bbox_area": round(float(metrics["bbox_area"]), 3),
                    "bbox_center_x": round(float(metrics["bbox_center_x"]), 3),
                    "bbox_center_y": round(float(metrics["bbox_center_y"]), 3),
                    "touches_left_edge": metrics["touches_left_edge"],
                    "touches_right_edge": metrics["touches_right_edge"],
                    "touches_top_edge": metrics["touches_top_edge"],
                    "touches_bottom_edge": metrics["touches_bottom_edge"],
                    "native_tracker_id": native_match.native_tracker_id if native_match is not None else None,
                    "logical_track_id": logical_match.track_uuid if logical_match is not None else None,
                    "lifecycle_state": logical_match.state if logical_match is not None else None,
                    "stabilized_class": stabilized_class,
                    "aspect_ratio": round(aspect.aspect_ratio, 6) if aspect.aspect_ratio is not None else None,
                    "aspect_ratio_min": round(aspect.minimum, 6) if aspect.minimum is not None else None,
                    "aspect_ratio_max": round(aspect.maximum, 6) if aspect.maximum is not None else None,
                    "aspect_ratio_deviation": round(aspect.deviation, 6) if aspect.deviation is not None else None,
                    "aspect_ratio_status": aspect.status,
                    "partial_visibility_ratio": round(float(metrics["partial_visibility_ratio"]), 6),
                }
            )
            raw_detection_rows.append(
                {
                    "frame_number": frame_number,
                    "class_name": normalized_class,
                    "raw_class_name": row["raw_class_name"],
                    "confidence": row["confidence"],
                    "aspect_ratio": aspect.aspect_ratio,
                    "touches_edge": metrics["touches_edge"],
                    "native_tracker_id": native_match.native_tracker_id if native_match is not None else None,
                    "logical_track_id": logical_match.track_uuid if logical_match is not None else None,
                    "lifecycle_state": logical_match.state if logical_match is not None else None,
                }
            )
            if bool(row["accepted_by_class_threshold"]) and logical_match is not None:
                accepted_detection_rows.append(
                    {
                        "frame_number": frame_number,
                        "raw_class_name": row["raw_class_name"],
                        "normalized_class_name": normalized_class,
                        "confidence": row["confidence"],
                        "global_inference_floor": row["global_inference_floor"],
                        "class_threshold": row["class_threshold"],
                        "bbox_xyxy": json.dumps([round(float(value), 3) for value in row["bbox_xyxy"]]),
                        "native_tracker_id": native_match.native_tracker_id if native_match is not None else None,
                        "logical_track_id": logical_match.track_uuid,
                    }
                )
            elif not bool(row["accepted_by_class_threshold"]):
                rejected_detection_rows.append(
                    {
                        "frame_number": frame_number,
                        "raw_class_name": row["raw_class_name"],
                        "normalized_class_name": normalized_class,
                        "confidence": row["confidence"],
                        "global_inference_floor": row["global_inference_floor"],
                        "class_threshold": row["class_threshold"],
                        "accepted_by_class_threshold": row["accepted_by_class_threshold"],
                        "rejection_reason": row["rejection_reason"],
                        "bbox_xyxy": json.dumps([round(float(value), 3) for value in row["bbox_xyxy"]]),
                    }
                )
        frame_level_reports.append(
            _build_frame_level_report(
                frame_number=frame_number,
                timestamp_seconds=frame_packet.video_time_seconds,
                raw_rows=raw_rows,
                filtered_detection_count=len(detection_packet.detections),
                native_tracker_count=len(tracking_result.native_observations or []),
                logical_tracker_count=len(tracking_result.observations),
            )
        )
    capture.release()
    flush_result = router.flush_all()
    completed_tracks.extend(flush_result.completed_tracks)
    deduped_tracks: dict[str, LocalVehicleTrack] = {}
    for track in completed_tracks:
        track.evidence_package = evidence_collector.finalize_track(track)
        deduped_tracks[track.track_uuid] = track
    tracks = list(deduped_tracks.values())
    tracks_json = [_build_track_json(track) for track in tracks]
    aspect_stats = {
        "all_detections": _group_aspect_ratio_stats(frame_rows),
        "high_confidence_detections": _group_aspect_ratio_stats(_selected_track_rows(frame_rows, minimum_confidence=0.50)),
        "non_edge_detections": _group_aspect_ratio_stats(_selected_track_rows(frame_rows, non_edge_only=True)),
        "confirmed_logical_tracks": _group_aspect_ratio_stats(_selected_track_rows(frame_rows, confirmed_only=True)),
        "good_evidence_frames": _group_aspect_ratio_stats([row for row in frame_rows if row.get("stabilized_class")]),
    }
    plate_audit = _anpr_audit_for_tracks(
        tracks=tracks,
        artifact_root=artifact_root,
        anpr_config_path=anpr_config_path,
        florence_config_path=florence_config_path,
        skip=skip_anpr,
    )
    _write_csv(run_dir / "detections.csv", frame_rows)
    _write_csv(run_dir / "raw_detections.csv", frame_rows)
    _write_csv(run_dir / "accepted_detections.csv", accepted_detection_rows)
    _write_csv(run_dir / "rejected_detections.csv", rejected_detection_rows)
    _json_dump(run_dir / "frame_reports.json", frame_level_reports)
    _json_dump(run_dir / "tracks.json", tracks_json)
    _json_dump(run_dir / "class_history.json", tracks_json)
    _json_dump(run_dir / "plate_audit.json", plate_audit)
    _json_dump(run_dir / "aspect_ratio_stats.json", aspect_stats)
    class_consistency_payload = _build_class_consistency_payload(tracks_json)
    _json_dump(run_dir / "class_consistency_report.json", class_consistency_payload)
    (run_dir / "class_consistency_report.md").write_text(_class_consistency_markdown(class_consistency_payload), encoding="utf-8")
    _build_evidence_index(run_dir, tracks)
    _manual_evaluation_sheet(run_dir, tracks)
    conflict_counts = _track_class_conflict_counts(tracks_json)
    short_tracks = sum(1 for track in tracks_json if int(track["observation_count"]) < 3)
    short_tracks_under_one_second = sum(1 for track in tracks_json if float(track["duration_seconds"]) < 1.0)
    class_counts = _count_class_occurrences(frame_rows, key="raw_class_name")
    empty_space_boxes = sum(
        1
        for row in frame_rows
        if row["aspect_ratio_status"] in {"BELOW_MINIMUM", "ABOVE_MAXIMUM"} and float(row["detection_confidence"]) < 0.50
    )
    outside_class_ratio = sum(1 for row in frame_rows if row["aspect_ratio_status"] in {"BELOW_MINIMUM", "ABOVE_MAXIMUM"})
    edge_tolerated = sum(1 for row in frame_rows if row["aspect_ratio_status"] == "EDGE_TOLERATED")
    abnormal_new_tracks = sum(
        1
        for row in frame_rows
        if row["aspect_ratio_status"] in {"BELOW_MINIMUM", "ABOVE_MAXIMUM"} and not row["logical_track_id"]
    )
    abnormal_confirmed_tracks = sum(
        1
        for row in frame_rows
        if row["aspect_ratio_status"] in {"BELOW_MINIMUM", "ABOVE_MAXIMUM"} and str(row.get("lifecycle_state") or "") == "active"
    )
    wrong_bus_labels = sum(1 for track in tracks_json if (track.get("stable_class") or track.get("class_name")) == "bus" and "car" in (track.get("class_observation_counts") or {}))
    summary = {
        "confidence_threshold": run_config.confidence_threshold,
        "raw_yolo_detections": len(raw_detection_rows),
        "accepted_detections": len(accepted_detection_rows),
        "rejected_by_class_threshold": len(rejected_detection_rows),
        "accepted_by_class": _count_class_occurrences(accepted_detection_rows, key="normalized_class_name"),
        "rejected_by_class": _count_class_occurrences(rejected_detection_rows, key="normalized_class_name"),
        "average_confidence_by_class": _average_confidence_by_class(raw_detection_rows),
        "minimum_accepted_confidence_by_class": _minimum_confidence_by_class(accepted_detection_rows),
        "total_yolo_detections": total_yolo_detections,
        "total_detections": len(frame_rows),
        "detections_below_035": sum(1 for row in frame_rows if float(row["detection_confidence"]) < 0.35),
        "detections_below_050": sum(1 for row in frame_rows if float(row["detection_confidence"]) < 0.50),
        "detections_touching_image_edge": sum(
            1
            for row in frame_rows
            if bool(row["touches_left_edge"] or row["touches_right_edge"] or row["touches_top_edge"] or row["touches_bottom_edge"])
        ),
        "class_detection_counts": class_counts,
        "car_detections": class_counts.get("car", 0),
        "bus_detections": class_counts.get("bus", 0),
        "truck_detections": class_counts.get("truck", 0),
        "motorcycle_detections": class_counts.get("motorcycle", 0),
        "logical_tracks": len(tracks_json),
        "native_tracks_created": len({native_id for track in tracks_json for native_id in track["native_tracker_ids_seen"]}),
        "completed_tracks": sum(1 for track in tracks_json if track["state"] == "completed"),
        "discarded_tracks": sum(1 for track in tracks_json if track["state"] == "discarded"),
        "short_tracks": short_tracks,
        "short_tracks_under_one_second": short_tracks_under_one_second,
        "tracks_with_multiple_native_ids": sum(1 for track in tracks_json if len(track["native_tracker_ids_seen"]) > 1),
        "tracks_split_by_identity_continuity": sum(1 for track in tracks_json if track.get("completion_reason") == "identity_split"),
        "fragment_relinks": sum(int(track["number_of_fragment_relinks"]) for track in tracks_json),
        "tracks_with_more_than_one_raw_class": sum(1 for track in tracks_json if len(track["class_observation_counts"]) > 1),
        "average_observations_per_track": statistics.mean([track["observation_count"] for track in tracks_json]) if tracks_json else 0.0,
        "median_observations_per_track": statistics.median([track["observation_count"] for track in tracks_json]) if tracks_json else 0.0,
        "maximum_track_duration": max([track["duration_seconds"] for track in tracks_json], default=0.0),
        "invalid_bboxes": invalid_bbox_count,
        "duplicate_overlapping_detections": duplicate_overlaps,
        "outside_class_ratio": outside_class_ratio,
        "edge_tolerated": edge_tolerated,
        "abnormal_new_tracks": abnormal_new_tracks,
        "abnormal_confirmed_tracks": abnormal_confirmed_tracks,
        "empty_space_boxes": empty_space_boxes,
        "wrong_bus_labels": wrong_bus_labels,
        "plate_conflicts": len(plate_audit.get("conflicts", [])),
        **conflict_counts,
    }
    run_report = {
        "run_label": run_config.label,
        "run_id": run_id,
        "video": video_info,
        "frame_range": {"start_frame": start_frame, "end_frame": end_frame},
        "detection_config": {
            "model_path": detection_config.model_path,
            "fallback_model_path": detection_config.fallback_model_path,
            "device": detection_config.device,
            "confidence_threshold": detection_config.confidence_threshold,
            "effective_inference_confidence_floor": detector.inference_confidence_floor,
            "iou_threshold": detection_config.iou_threshold,
            "image_size": detection_config.image_size,
            "allowed_classes": list(detection_config.allowed_classes),
            "class_confidence_thresholds": {
                "enabled": detection_config.class_confidence_thresholds.enabled,
                "default": detection_config.class_confidence_thresholds.default,
                "classes": dict(detection_config.class_confidence_thresholds.classes),
            },
        },
        "tracking_config": asdict(tracking_config),
        "summary": summary,
        "artifact_paths": {
            "run_report": str((run_dir / "run_report.json").resolve()),
            "detections_csv": str((run_dir / "detections.csv").resolve()),
            "raw_detections_csv": str((run_dir / "raw_detections.csv").resolve()),
            "accepted_detections_csv": str((run_dir / "accepted_detections.csv").resolve()),
            "rejected_detections_csv": str((run_dir / "rejected_detections.csv").resolve()),
            "tracks_json": str((run_dir / "tracks.json").resolve()),
            "class_history_json": str((run_dir / "class_history.json").resolve()),
            "class_consistency_report_json": str((run_dir / "class_consistency_report.json").resolve()),
            "class_consistency_report_md": str((run_dir / "class_consistency_report.md").resolve()),
            "plate_audit_json": str((run_dir / "plate_audit.json").resolve()),
            "evidence_index_html": str((run_dir / "evidence_index.html").resolve()),
            "diagnostic_frames": str((run_dir / "diagnostic_frames").resolve()),
            "artifacts": str(artifact_root),
        },
    }
    _json_dump(run_dir / "run_report.json", run_report)
    return {
        "run_dir": str(run_dir),
        "run_report": run_report,
        "tracks": tracks_json,
        "frame_rows": frame_rows,
        "plate_audit": plate_audit,
        "summary": summary,
    }


def _build_confidence_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        tracks = item["tracks"]
        rows.append(
            {
                "confidence": item["summary"]["confidence_threshold"],
                "total_detections": item["summary"]["total_detections"],
                "logical_tracks": item["summary"]["logical_tracks"],
                "short_tracks": item["summary"]["short_tracks"],
                "motorcycle_tracks": sum(1 for track in tracks if (track.get("stable_class") or track.get("class_name")) == "motorcycle"),
                "wagonr_tracks": "",
                "honda_tracks": "",
                "wrong_bus_labels": item["summary"]["wrong_bus_labels"],
                "mixed_tracks": item["summary"]["mixed_tracks"],
                "empty_space_boxes": item["summary"]["empty_space_boxes"],
                "plate_conflicts": item["summary"]["plate_conflicts"],
            }
        )
    return rows


def _build_runtime_baseline(
    *,
    camera_config_path: Path,
    detection_config_path: Path,
    tracking_config_path: Path,
    worker_config_path: Path,
    persistence_config_path: Path,
    evidence_config_path: Path,
    video_path: Path,
    camera_code: str,
) -> dict[str, Any]:
    from .report_tracking_configuration import generate_report

    args = SimpleNamespace(
        camera_config=str(camera_config_path),
        detection_config=str(detection_config_path),
        tracking_config=str(tracking_config_path),
        worker_config=str(worker_config_path),
        persistence_config=str(persistence_config_path),
        evidence_config=str(evidence_config_path),
        anpr_config=None,
        camera_code=camera_code,
        camera_codes=None,
        camera_limit=None,
        persist_to_supabase=False,
        dry_run_persistence=True,
        json_output=None,
    )
    report = generate_report(args)
    report["video_probe"] = probe_video(video_path)
    return report


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    camera_config_path = Path(args.camera_config).expanduser().resolve()
    detection_config_path = Path(args.detection_config).expanduser().resolve()
    tracking_config_path = Path(args.tracking_config).expanduser().resolve()
    worker_config_path = Path(args.worker_config).expanduser().resolve()
    persistence_config_path = Path(args.persistence_config).expanduser().resolve()
    evidence_config_path = Path(args.evidence_config).expanduser().resolve()
    anpr_config_path = Path(args.anpr_config).expanduser().resolve()
    florence_config_path = Path(args.florence_config).expanduser().resolve()
    video_path, camera_code = resolve_experiment_video(camera_config_path=camera_config_path, camera_code=args.camera_code, video_path=args.video_path)
    camera_name = camera_code
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = _build_runtime_baseline(
        camera_config_path=camera_config_path,
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        worker_config_path=worker_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        video_path=video_path,
        camera_code=camera_code,
    )
    base_run_id = f"CONFIDENCE_SWEEP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _json_dump(output_root / "baseline_runtime_configuration.json", baseline)
    video_probe = baseline["video_probe"]
    source_fps = float(video_probe["source_fps"] or 30.0)
    frame_budget = int(round(float(args.duration_seconds) * source_fps))
    if args.limit_frames is not None:
        frame_budget = min(frame_budget, int(args.limit_frames))
    end_frame = min(int(video_probe["frame_count"]) - 1, int(args.start_frame) + max(frame_budget - 1, 0))
    confidence_runs: list[dict[str, Any]] = []
    aspect_ratio_config = AspectRatioValidationConfig(
        enabled=True,
        classes={name: AspectRatioClassRange() for name in ("car", "bus", "truck", "motorcycle", "3wheeler")},
        action="report_only",
    )
    for confidence_value in args.confidence_values:
        label = f"confidence_{int(round(confidence_value * 100)):03d}"
        run_dir = ensure_unique_run_directory(output_root, label)
        result = _run_single_configuration(
            run_config=ExperimentRunConfig(label=label, confidence_threshold=float(confidence_value), match_threshold=0.80, minimum_consecutive_frames=1),
            camera_code=camera_code,
            camera_name=camera_name,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=end_frame,
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            run_dir=run_dir,
            base_run_id=base_run_id,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
        )
        confidence_runs.append(result)
    best_confidence_run = _heuristic_best_run(confidence_runs)
    second_stage_runs: list[dict[str, Any]] = []
    second_stage_matrix = [
        ExperimentRunConfig(label="tracking_baseline", confidence_threshold=float(best_confidence_run["summary"]["confidence_threshold"]), match_threshold=0.80, minimum_consecutive_frames=1),
        ExperimentRunConfig(label="tracking_A", confidence_threshold=float(best_confidence_run["summary"]["confidence_threshold"]), match_threshold=0.75, minimum_consecutive_frames=1),
        ExperimentRunConfig(label="tracking_B", confidence_threshold=float(best_confidence_run["summary"]["confidence_threshold"]), match_threshold=0.80, minimum_consecutive_frames=2),
        ExperimentRunConfig(label="tracking_C", confidence_threshold=float(best_confidence_run["summary"]["confidence_threshold"]), match_threshold=0.75, minimum_consecutive_frames=2),
    ]
    for run_config in second_stage_matrix:
        run_dir = ensure_unique_run_directory(output_root, run_config.label)
        result = _run_single_configuration(
            run_config=run_config,
            camera_code=camera_code,
            camera_name=camera_name,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=end_frame,
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            run_dir=run_dir,
            base_run_id=base_run_id,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
        )
        second_stage_runs.append(result)
    comparison = {
        "video": video_probe,
        "camera_code": camera_code,
        "frame_range": {"start_frame": int(args.start_frame), "end_frame": end_frame},
        "baseline_runtime_configuration": str((output_root / "baseline_runtime_configuration.json").resolve()),
        "confidence_rows": _build_confidence_rows(confidence_runs),
        "best_confidence_run": {
            "run_label": best_confidence_run["run_report"]["run_label"],
            "confidence_threshold": best_confidence_run["summary"]["confidence_threshold"],
            "reason": "lowest mixed-track, empty-space, plate-conflict, and logical-track counts under the current automatic heuristic",
            "artifact_path": best_confidence_run["run_report"]["artifact_paths"]["run_report"],
        },
        "second_stage_runs": [
            {
                "run_label": item["run_report"]["run_label"],
                "match_threshold": item["run_report"]["tracking_config"]["minimum_matching_threshold"],
                "minimum_consecutive_frames": item["run_report"]["tracking_config"]["minimum_consecutive_frames"],
                "summary": item["summary"],
                "artifact_path": item["run_report"]["artifact_paths"]["run_report"],
            }
            for item in second_stage_runs
        ],
    }
    _json_dump(output_root / "comparison.json", comparison)
    (output_root / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def main() -> None:
    args = parse_args()
    comparison = run_experiment(args)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
