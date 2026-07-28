from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from ..detection.annotation import annotate_detection_frame
from ..detection.detection_config import DetectionConfig, detection_overrides_from_env, load_detection_config
from ..detection.detection_models import DetectionPacket
from ..detection.vehicle_detector import SharedVehicleDetector
from ..ingestion.frame_packet import FramePacket
from ..orchestration.multi_camera_orchestrator import MultiCameraOrchestrator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionRunResult:
    report: dict[str, object]
    report_path: Path


class MultiCameraDetectionOrchestrator:
    def __init__(
        self,
        camera_config_path: str | Path,
        detection_config_path: str | Path,
        *,
        mode: str = "round_robin",
        max_frames_per_camera: int | None = None,
        detection_overrides: dict[str, object] | None = None,
        detector: SharedVehicleDetector | None = None,
    ) -> None:
        self.camera_orchestrator = MultiCameraOrchestrator(camera_config_path, mode=mode, max_frames_per_camera=max_frames_per_camera)
        self.detection_config_path = Path(detection_config_path).expanduser().resolve()
        merged_overrides = detection_overrides_from_env()
        if detection_overrides:
            merged_overrides.update(detection_overrides)
        self.detection_config = load_detection_config(self.detection_config_path, overrides=merged_overrides)
        self.detector = detector or SharedVehicleDetector(self.detection_config)

    def run(
        self,
        *,
        preview: bool = False,
        preview_scale: float = 1.0,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        output_report: str | Path | None = None,
    ) -> DetectionRunResult:
        camera_configs = self.camera_orchestrator.load_enabled_cameras()
        reader = self.camera_orchestrator._build_reader(camera_configs) if hasattr(self.camera_orchestrator, "_build_reader") else None
        if reader is None:
            from ..ingestion.multi_camera_reader import MultiCameraReader

            reader = MultiCameraReader(camera_configs, mode=self.camera_orchestrator.mode, max_frames_per_camera=self.camera_orchestrator.max_frames_per_camera)
        output_dir = Path(output_report).resolve().parent if output_report else self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_counts: dict[str, int] = {config.camera_code: 0 for config in camera_configs}
        camera_stats: dict[str, dict[str, object]] = {
            config.camera_code: {
                "camera_name": config.camera_name,
                "frames_processed": 0,
                "frames_with_detections": 0,
                "frames_without_detections": 0,
                "total_vehicle_detections": 0,
                "class_counts": {},
                "average_detections_per_frame": 0.0,
                "average_inference_time_ms": 0.0,
                "maximum_inference_time_ms": 0.0,
                "first_frame_number": None,
                "last_frame_number": None,
                "first_video_time_seconds": None,
                "last_video_time_seconds": None,
                "errors": [],
            }
            for config in camera_configs
        }
        combined_class_counts: Counter[str] = Counter()
        total_frames_processed = 0
        total_vehicle_detections = 0
        inference_times: list[float] = []
        wall_started = time.perf_counter()

        try:
            reader.open()
            for frame_packet in reader:
                detection_packet = self.detector.detect(frame_packet)
                total_frames_processed += 1
                total_vehicle_detections += len(detection_packet.detections)
                inference_times.append(detection_packet.inference_time_ms)
                stats = camera_stats[frame_packet.camera_code]
                stats["frames_processed"] = int(stats["frames_processed"]) + 1
                if stats["first_frame_number"] is None:
                    stats["first_frame_number"] = frame_packet.frame_number
                    stats["first_video_time_seconds"] = frame_packet.video_time_seconds
                stats["last_frame_number"] = frame_packet.frame_number
                stats["last_video_time_seconds"] = frame_packet.video_time_seconds
                stats["maximum_inference_time_ms"] = max(float(stats["maximum_inference_time_ms"]), detection_packet.inference_time_ms)
                if detection_packet.detections:
                    stats["frames_with_detections"] = int(stats["frames_with_detections"]) + 1
                else:
                    stats["frames_without_detections"] = int(stats["frames_without_detections"]) + 1
                stats["total_vehicle_detections"] = int(stats["total_vehicle_detections"]) + len(detection_packet.detections)
                class_counts: Counter[str] = Counter(stats["class_counts"])
                for detection in detection_packet.detections:
                    class_counts[detection.class_name] += 1
                    combined_class_counts[detection.class_name] += 1
                stats["class_counts"] = dict(sorted(class_counts.items()))
                stats["average_inference_time_ms"] = (
                    (float(stats["average_inference_time_ms"]) * (int(stats["frames_processed"]) - 1) + detection_packet.inference_time_ms)
                    / int(stats["frames_processed"])
                )
                stats["average_detections_per_frame"] = int(stats["total_vehicle_detections"]) / int(stats["frames_processed"])

                annotated_frame = None
                if preview or save_sample_frames:
                    annotated_frame = annotate_detection_frame(frame_packet, detection_packet)
                if preview and annotated_frame is not None:
                    shown_frame = annotated_frame
                    if preview_scale != 1.0:
                        shown_frame = cv2.resize(shown_frame, None, fx=preview_scale, fy=preview_scale)
                    cv2.imshow("multicamera_detection_preview", shown_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                if save_sample_frames and annotated_frame is not None and sample_counts[frame_packet.camera_code] < sample_frame_limit_per_camera:
                    sample_counts[frame_packet.camera_code] += 1
                    camera_dir = output_dir / frame_packet.camera_code
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    sample_path = camera_dir / f"sample_{sample_counts[frame_packet.camera_code]:06d}.jpg"
                    cv2.imwrite(str(sample_path), annotated_frame)
        finally:
            reader.close()
            if preview:
                cv2.destroyAllWindows()

        wall_runtime = time.perf_counter() - wall_started
        average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        report = {
            "mode": self.camera_orchestrator.mode,
            "camera_count": len(camera_configs),
            "detector": {
                "requested_model": self.detection_config.model_path,
                "actual_model": self.detector.loaded_model_name or self.detection_config.model_path,
                "device": self.detector.device,
                "effective_inference_confidence_floor": self.detection_config.confidence_threshold,
                "class_confidence_thresholds": dict(self.detection_config.class_confidence_thresholds.classes),
                "iou_threshold": self.detection_config.iou_threshold,
                "image_size": self.detection_config.image_size,
                "allow_fallback": self.detection_config.allow_fallback,
                "fallback_model_path": self.detection_config.fallback_model_path,
            },
            "total_frames_processed": total_frames_processed,
            "total_vehicle_detections": total_vehicle_detections,
            "combined_class_counts": dict(sorted(combined_class_counts.items())),
            "average_inference_time_ms": average_inference_time,
            "wall_clock_runtime_seconds": wall_runtime,
            "processing_fps": (total_frames_processed / wall_runtime) if wall_runtime > 0 else 0.0,
            "cameras": camera_stats,
            "generated_at": datetime.now().isoformat(),
        }
        report_path = Path(output_report).resolve() if output_report else output_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info(
            "Detection validation complete cameras=%s frames=%s detections=%s model=%s device=%s report=%s",
            len(camera_configs),
            total_frames_processed,
            total_vehicle_detections,
            self.detector.loaded_model_name or self.detection_config.model_path,
            self.detector.device,
            report_path,
        )
        return DetectionRunResult(report=report, report_path=report_path)

    @staticmethod
    def _default_output_dir() -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("debug_runs") / "multicamera_vehicle_tracking_pipeline" / f"detection_validation_{timestamp}"
