from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from ..detection.detection_config import load_detection_config
from ..detection.detection_models import DetectionPacket, VehicleDetection
from ..detection.vehicle_detector import SharedVehicleDetector, normalize_vehicle_class
from ..ingestion.frame_packet import FramePacket
from ..tracking.annotation import annotate_tracking_frame
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.class_recalculation import evaluate_identity_continuity
from ..tracking.tracker_factory import TrackerFactory
from ..tracking.tracking_config import load_tracking_config
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a camera window to diagnose likely identity switches.")
    parser.add_argument("--run-code", required=True)
    parser.add_argument("--camera-code", required=True)
    parser.add_argument("--track-uuid", default=None)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--tracking-config", required=True)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _camera_timestamp_for_frame(frame_number: int, fps: float) -> datetime:
    base = datetime(2026, 7, 27, 0, 0, 0)
    return base + timedelta(seconds=(frame_number / max(fps, 1.0)))


def _raw_detections_for_frame(detector: SharedVehicleDetector, frame: Any) -> list[dict[str, Any]]:
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


def _filtered_detections_for_frame(detector: SharedVehicleDetector, packet: FramePacket) -> list[VehicleDetection]:
    detection_packet = detector.detect(packet)
    return detection_packet.detections


def _build_frame_packet(
    *,
    camera_code: str,
    camera_name: str,
    source_path: Path,
    frame_number: int,
    source_fps: float,
    source_frame_count: int | None,
    frame: Any,
) -> FramePacket:
    return FramePacket(
        camera_code=camera_code,
        camera_name=camera_name,
        source_path=source_path,
        frame_number=frame_number,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        video_time_seconds=frame_number / max(source_fps, 1.0),
        camera_timestamp=_camera_timestamp_for_frame(frame_number, source_fps),
        frame=frame,
    )


def _build_detection_packet(frame_packet: FramePacket, detections: list[VehicleDetection], detector: SharedVehicleDetector) -> DetectionPacket:
    frame_height, frame_width = frame_packet.frame.shape[:2]
    return DetectionPacket(
        camera_code=frame_packet.camera_code,
        camera_name=frame_packet.camera_name,
        source_path=frame_packet.source_path,
        frame_number=frame_packet.frame_number,
        video_time_seconds=frame_packet.video_time_seconds,
        camera_timestamp=frame_packet.camera_timestamp,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        detections=detections,
        inference_time_ms=0.0,
        detector_model=detector.loaded_model_name or detector.config.model_path,
        detector_device=detector.device,
        source_fps=frame_packet.source_fps,
        frame=frame_packet.frame,
    )


def _active_track_lookup(router: CameraDetectionRouter, camera_code: str) -> dict[str, LocalVehicleTrack]:
    return {track.track_uuid: track for track in router.lifecycle.get_active_tracks(camera_code)}


def _find_likely_switches(
    observations: list[TrackObservation],
    previous_active_tracks: dict[str, LocalVehicleTrack],
    current_active_tracks: dict[str, LocalVehicleTrack],
    router: CameraDetectionRouter,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for observation in observations:
        previous_track = previous_active_tracks.get(observation.track_uuid)
        current_track = current_active_tracks.get(observation.track_uuid)
        if previous_track is None or current_track is None or not previous_track.observations or not current_track.observations:
            continue
        candidate_track = LocalVehicleTrack(
            track_uuid=current_track.track_uuid,
            camera_code=current_track.camera_code,
            local_track_id=current_track.local_track_id,
            class_name=observation.class_name,
            first_frame_number=observation.frame_number,
            last_frame_number=observation.frame_number,
            first_seen_at=observation.camera_timestamp,
            last_seen_at=observation.camera_timestamp,
            first_video_time_seconds=observation.video_time_seconds,
            last_video_time_seconds=observation.video_time_seconds,
            observation_count=1,
            best_confidence=float(observation.confidence),
            state=observation.state,
            observations=[observation],
            camera_name=current_track.camera_name,
            source_path=current_track.source_path,
            stable_class_name=current_track.stable_class_name,
        )
        evaluation = evaluate_identity_continuity(previous_track, candidate_track, router.tracking_config)
        if evaluation.eligible:
            continue
        findings.append(
            {
                "track_uuid": observation.track_uuid,
                "frame_number": observation.frame_number,
                "timestamp": observation.video_time_seconds,
                "reasons": list(evaluation.reasons),
                "spatial_score": evaluation.spatial_score,
                "class_compatibility": evaluation.class_compatibility,
                "area_ratio": evaluation.area_ratio,
                "previous_bbox": list(previous_track.observations[-1].bbox_xyxy),
                "new_bbox": list(observation.bbox_xyxy),
                "previous_class": previous_track.class_name,
                "new_class": observation.class_name,
            }
        )
    return findings


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_report = Path(args.output_report).expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)

    tracking_config = load_tracking_config(args.tracking_config)
    detection_config = load_detection_config(args.detection_config)
    detector = SharedVehicleDetector(detection_config)
    router = CameraDetectionRouter(
        tracking_config,
        run_id=args.run_code,
        tracker_factory=TrackerFactory(tracking_config),
        allowed_camera_codes=(args.camera_code,),
    )

    video_path = Path(args.video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    annotated_path = output_dir / f"{args.run_code}_{args.camera_code}_{args.start_frame}_{args.end_frame}_annotated.mp4"
    writer: cv2.VideoWriter | None = None
    frame_reports: list[dict[str, Any]] = []
    likely_switches: list[dict[str, Any]] = []

    for frame_number in range(args.start_frame, args.end_frame + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame_packet = _build_frame_packet(
            camera_code=args.camera_code,
            camera_name=args.camera_code,
            source_path=video_path,
            frame_number=frame_number,
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            frame=frame,
        )
        raw_detections = _raw_detections_for_frame(detector, frame)
        filtered_detections = _filtered_detections_for_frame(detector, frame_packet)
        detection_packet = _build_detection_packet(frame_packet, filtered_detections, detector)
        previous_active_tracks = _active_track_lookup(router, args.camera_code)
        tracking_result = router.route(detection_packet)
        current_active_tracks = _active_track_lookup(router, args.camera_code)
        findings = _find_likely_switches(tracking_result.observations, previous_active_tracks, current_active_tracks, router)
        likely_switches.extend(findings)

        annotated_frame = annotate_tracking_frame(
            frame_packet,
            tracking_result.observations,
            active_track_count=len(tracking_result.active_tracks),
        )
        for item in raw_detections:
            x1, y1, x2, y2 = (int(round(value)) for value in item["bbox_xyxy"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if writer is None:
            height, width = annotated_frame.shape[:2]
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, min(source_fps, 30.0)),
                (width, height),
            )
        writer.write(annotated_frame)

        active_lookup_by_uuid = _active_track_lookup(router, args.camera_code)
        rows: list[dict[str, Any]] = []
        for detection_index, item in enumerate(raw_detections):
            matched_observation = next(
                (
                    observation
                    for observation in tracking_result.observations
                    if detection_index < len(filtered_detections)
                    and observation.bbox_xyxy == filtered_detections[detection_index].bbox_xyxy
                ),
                None,
            )
            active_track = active_lookup_by_uuid.get(matched_observation.track_uuid) if matched_observation is not None else None
            rows.append(
                {
                    "camera_code": args.camera_code,
                    "frame_number": frame_number,
                    "timestamp": frame_packet.video_time_seconds,
                    "queue_sequence_number": frame_number,
                    "detection_index": detection_index,
                    "detected_class": item["normalized_class_name"],
                    "raw_detected_class": item["raw_class_name"],
                    "detection_confidence": item["confidence"],
                    "detection_bbox": item["bbox_xyxy"],
                    "tracker_id": matched_observation.local_track_id if matched_observation is not None else None,
                    "track_uuid": matched_observation.track_uuid if matched_observation is not None else None,
                    "tracker_state": matched_observation.state if matched_observation is not None else None,
                    "matched_previous_track_id": None,
                    "IoU": None,
                    "association_score": None,
                    "lost_frames": active_track.lost_frame_count if active_track is not None else None,
                    "track_age": active_track.observation_count if active_track is not None else None,
                    "is_activated": bool(active_track is not None and active_track.state != "tentative"),
                }
            )
        frame_reports.append(
            {
                "camera_code": args.camera_code,
                "frame_number": frame_number,
                "timestamp": frame_packet.video_time_seconds,
                "raw_detections": raw_detections,
                "filtered_detections": [
                    {
                        "detection_index": index,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox_xyxy": list(detection.bbox_xyxy),
                    }
                    for index, detection in enumerate(filtered_detections)
                ],
                "tracker_outputs": [
                    {
                        "track_uuid": observation.track_uuid,
                        "tracker_id": observation.local_track_id,
                        "class_name": observation.class_name,
                        "confidence": observation.confidence,
                        "bbox_xyxy": list(observation.bbox_xyxy),
                        "state": observation.state,
                    }
                    for observation in tracking_result.observations
                ],
                "diagnostic_rows": rows,
                "likely_switches": findings,
            }
        )

    capture.release()
    if writer is not None:
        writer.release()

    report = {
        "run_code": args.run_code,
        "camera_code": args.camera_code,
        "track_uuid": args.track_uuid,
        "video_path": str(video_path),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "effective_tracking_config": asdict(tracking_config),
        "effective_detection_config": {
            "model_path": detection_config.model_path,
            "fallback_model_path": detection_config.fallback_model_path,
            "allow_fallback": detection_config.allow_fallback,
            "device": detector.device,
            "confidence_threshold": detection_config.confidence_threshold,
            "iou_threshold": detection_config.iou_threshold,
            "image_size": detection_config.image_size,
            "allowed_classes": list(detection_config.allowed_classes),
            "source_fps": source_fps,
        },
        "annotated_video_path": str(annotated_path),
        "frame_reports": frame_reports,
        "first_likely_switch": likely_switches[0] if likely_switches else None,
        "limitations": [
            "The current ByteTrack backends do not expose internal match IoU or association score per accepted update.",
            "matched_previous_track_id, IoU, and association_score remain null unless backend metadata becomes available.",
        ],
    }
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_report": str(output_report), "annotated_video_path": str(annotated_path), "first_likely_switch": report["first_likely_switch"]}, indent=2))


if __name__ == "__main__":
    main()
