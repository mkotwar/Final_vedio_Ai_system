from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (case_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from continuous_mot_hybrid.bytetrack_backend import ByteTrackBackend
    from continuous_mot_hybrid.detection_track_adapter import detections_from_rows
    from continuous_mot_hybrid.fixed_5fps_validation_core import (
        ACTIVE,
        LOST,
        RECOVERABLE,
        TERMINATED,
        TrackLifecycleRecord,
        build_validation_checks,
        build_validation_metrics,
        classify_zone,
        compute_new_id_reason,
        detector_should_run,
        recovery_window_seconds,
        update_confirmation,
    )
    from continuous_mot_hybrid.report_writer import write_html_from_markdown, write_json, write_markdown
    from continuous_mot_hybrid.video_frame_stream import read_video_info, stream_processed_frames
    from continuous_mot_hybrid.yolo_detector import YoloDetector, resolve_model_specs
    from tests.td_case2.config import ENV_OBJECT_YOLO_MODEL_PATH, ENV_PERSON_YOLO_MODEL_PATH, ENV_YOLO_MODEL_PATH, repo_root as td_repo_root, resolve_case_path
else:
    from .bytetrack_backend import ByteTrackBackend
    from .detection_track_adapter import detections_from_rows
    from .fixed_5fps_validation_core import (
        ACTIVE,
        LOST,
        RECOVERABLE,
        TERMINATED,
        TrackLifecycleRecord,
        build_validation_checks,
        build_validation_metrics,
        classify_zone,
        compute_new_id_reason,
        detector_should_run,
        recovery_window_seconds,
        update_confirmation,
    )
    from .report_writer import write_html_from_markdown, write_json, write_markdown
    from .video_frame_stream import read_video_info, stream_processed_frames
    from .yolo_detector import YoloDetector, resolve_model_specs
    from tests.td_case2.config import ENV_OBJECT_YOLO_MODEL_PATH, ENV_PERSON_YOLO_MODEL_PATH, ENV_YOLO_MODEL_PATH, repo_root as td_repo_root, resolve_case_path


@dataclass(frozen=True)
class FixedValidationConfig:
    video_path: Path
    run_dir: Path
    camera_id: str
    camera_group: str
    camera_timezone: str
    processing_fps: float
    detector_fps: float
    device: str
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    match_thresh: float = 0.80
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed 5 FPS ByteTrack validation.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--camera-id", default="test_cam_01")
    parser.add_argument("--camera-group", default="single_camera_comparison")
    parser.add_argument("--camera-timezone", default="Asia/Kolkata")
    parser.add_argument("--processing-fps", type=float, default=10.0)
    parser.add_argument("--detector-fps", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir")
    return parser


def _resolve_models() -> tuple[Path | None, Path | None, Path | None]:
    def _path(env_name: str) -> Path | None:
        raw = str(__import__("os").environ.get(env_name, "")).strip()
        return resolve_case_path(raw) if raw else None
    return _path(ENV_PERSON_YOLO_MODEL_PATH), _path(ENV_OBJECT_YOLO_MODEL_PATH), _path(ENV_YOLO_MODEL_PATH)


def _make_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = td_repo_root() / "debug_runs" / f"fixed_5fps_bytetrack_{video_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("01_video", "02_detections", "03_tracking", "04_debug_frames", "05_reports", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_config(args: argparse.Namespace) -> FixedValidationConfig:
    video_path = Path(args.video_path).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _make_run_dir(video_path)
    if args.run_dir:
        for child in ("01_video", "02_detections", "03_tracking", "04_debug_frames", "05_reports", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
    return FixedValidationConfig(
        video_path=video_path,
        run_dir=run_dir,
        camera_id=args.camera_id,
        camera_group=args.camera_group,
        camera_timezone=args.camera_timezone,
        processing_fps=float(args.processing_fps),
        detector_fps=float(args.detector_fps),
        device=str(args.device),
    )


def _color_for_state(state: str) -> tuple[int, int, int]:
    return {
        ACTIVE: (40, 200, 40),
        "tentative": (40, 180, 220),
        RECOVERABLE: (0, 180, 255),
        "reactivated": (180, 40, 220),
        TERMINATED: (120, 120, 120),
    }.get(state, (200, 200, 200))


def _annotate_frame(*, frame: Any, timestamp_seconds: float, detector_ran: bool, tracks: list[dict[str, Any]]) -> Any:
    image = frame.copy()
    cv2.putText(
        image,
        f"t={timestamp_seconds:.2f}s {'DETECTOR' if detector_ran else 'SKIPPED'}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for track in tracks:
        x1, y1, x2, y2 = [int(round(float(value))) for value in track["bbox_xyxy"]]
        state = str(track["state"])
        color = _color_for_state(state)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{track['track_id']} {track['class_name']} {state}"
        cv2.putText(image, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return image


def main() -> None:
    args = _arg_parser().parse_args()
    config = _build_config(args)
    video_info = read_video_info(config.video_path)
    person_model_path, object_model_path, combined_model_path = _resolve_models()
    detector = YoloDetector(
        model_specs=resolve_model_specs(
            person_model_path=person_model_path,
            object_model_path=object_model_path,
            combined_model_path=combined_model_path,
        ),
        confidence=config.yolo_confidence,
        iou=config.yolo_iou,
        device=config.device,
    )
    backend = ByteTrackBackend(
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        match_thresh=config.match_thresh,
        track_buffer_frames=max(1, int(round(config.detector_fps * 1.0))),
    )
    write_json(
        config.run_dir / "01_video" / "video_info.json",
        {
            "status": "success",
            "video_path": str(config.video_path),
            "source_fps": video_info.source_fps,
            "source_frame_count": video_info.source_frame_count,
            "duration_seconds": video_info.duration_seconds,
            "width": video_info.width,
            "height": video_info.height,
        },
    )
    write_json(
        config.run_dir / "01_video" / "resolved_config.json",
        {
            "video_path": str(config.video_path),
            "run_dir": str(config.run_dir),
            "camera_id": config.camera_id,
            "camera_group": config.camera_group,
            "camera_timezone": config.camera_timezone,
            "processing_fps": config.processing_fps,
            "detector_fps": config.detector_fps,
            "mot_backend": "bytetrack",
            "visual_tracker": "disabled",
            "adaptive_scheduler": "disabled",
            "reconciliation": "disabled",
            "device": config.device,
        },
    )

    _, _, _, frame_iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    writer = cv2.VideoWriter(
        str(config.run_dir / "04_debug_frames" / "annotated_tracking.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.processing_fps,
        (video_info.width, video_info.height),
    )

    per_frame_events: list[dict[str, Any]] = []
    yolo_detections: list[dict[str, Any]] = []
    yolo_call_report: list[dict[str, Any]] = []
    raw_tracks: dict[str, TrackLifecycleRecord] = {}
    state_transitions: list[dict[str, Any]] = []
    new_id_events: list[dict[str, Any]] = []
    reactivation_events: list[dict[str, Any]] = []
    termination_events: list[dict[str, Any]] = []
    skipped_detector_events: list[dict[str, Any]] = []
    tracking_started = time.perf_counter()

    previous_states: dict[str, str] = {}
    previous_snapshot: dict[str, dict[str, Any]] = {}

    for frame_record, frame in frame_iterator:
        frame_payload = frame_record.to_dict()
        processed_frame_index = int(frame_payload["processed_frame_index"])
        timestamp_seconds = float(frame_payload["timestamp_seconds"])
        detector_ran = detector_should_run(processed_frame_index=processed_frame_index, processing_fps=config.processing_fps, detector_fps=config.detector_fps)
        active_before = [track_id for track_id, state in previous_states.items() if state in {"active", "tentative"}]
        recoverable_before = [track_id for track_id, state in previous_states.items() if state == RECOVERABLE]
        current_detections: list[dict[str, Any]] = []
        if detector_ran:
            current_detections = detector.detect(
                frame=frame,
                frame_record=frame_payload,
                scheduler_state="fixed_5fps",
                detector_reason="fixed_interval",
            )
            yolo_detections.extend(current_detections)
            yolo_call_report.append(
                {
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "detector_ran": True,
                    "detection_count": len(current_detections),
                    "detections_found": bool(current_detections),
                }
            )
            snapshot = backend.update(detections=detections_from_rows(current_detections))
        else:
            yolo_call_report.append(
                {
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "detector_ran": False,
                    "detection_count": 0,
                    "detections_found": None,
                }
            )
            snapshot = backend.handle_detector_skipped()
            skipped_detector_events.append(
                {
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "active_track_ids_before": active_before,
                    "recoverable_track_ids_before": recoverable_before,
                }
            )

        tracked_rows = [row for row in snapshot if row.backend_state == "tracked"]
        lost_rows = [row for row in snapshot if row.backend_state == "lost"]
        tracked_ids = [row.track_id for row in tracked_rows]
        lost_ids = [row.track_id for row in lost_rows]
        matched_track_ids = [row.track_id for row in tracked_rows if row.matched_detection_id]
        reactivated_ids = [row.track_id for row in tracked_rows if previous_states.get(row.track_id) == RECOVERABLE and row.matched_detection_id]
        if detector_ran:
            for row in tracked_rows:
                zone = classify_zone(list(row.bbox_xyxy), frame_width=video_info.width, frame_height=video_info.height)
                record = raw_tracks.get(row.track_id)
                if record is None:
                    record = TrackLifecycleRecord(
                        track_id=row.track_id,
                        family=row.family,
                        class_name=row.class_name,
                        created_timestamp_seconds=timestamp_seconds,
                        created_zone=zone,
                        last_detector_confirmation_timestamp=timestamp_seconds,
                    )
                    raw_tracks[row.track_id] = record
                    reason = compute_new_id_reason(
                        bbox_xyxy=list(row.bbox_xyxy),
                        class_name=row.class_name,
                        family=row.family,
                        recoverable_tracks=list(previous_snapshot.values()),
                        frame_width=video_info.width,
                        frame_height=video_info.height,
                    )
                    new_id_events.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": reason,
                            "zone": zone,
                            "detector_ran": True,
                        }
                    )
                if row.matched_detection_id:
                    record.detector_hit_timestamps.append(timestamp_seconds)
                    record.class_name = row.class_name
                    record.last_detector_confirmation_timestamp = timestamp_seconds
                    update_confirmation(record, timestamp_seconds=timestamp_seconds)
                current_state = ACTIVE if record.confirmed else "tentative"
                if previous_states.get(row.track_id) == RECOVERABLE and row.matched_detection_id:
                    record.reactivated_count += 1
                    reactivation_events.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "success": True,
                            "detector_ran": True,
                        }
                    )
                    current_state = "reactivated"
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": current_state,
                        "detector_ran": True,
                    }
                )
                if previous_states.get(row.track_id) != current_state:
                    state_transitions.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": previous_states.get(row.track_id),
                            "to_state": current_state,
                        }
                    )
                record.last_state = ACTIVE if record.confirmed else "tentative"
            for row in lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                recoverable_window = recovery_window_seconds(family=record.family, confirmed=record.confirmed)
                elapsed = timestamp_seconds - record.last_detector_confirmation_timestamp
                state = RECOVERABLE if elapsed <= recoverable_window else LOST
                if elapsed > recoverable_window and not record.terminated:
                    record.terminated = True
                    record.termination_reason = "recovery_window_expired"
                    termination_events.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": record.termination_reason,
                        }
                    )
                    state = TERMINATED
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": state,
                        "detector_ran": True,
                    }
                )
                if previous_states.get(row.track_id) != state:
                    state_transitions.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": previous_states.get(row.track_id),
                            "to_state": state,
                        }
                    )
                record.last_state = state
            for track_id, old_state in list(previous_states.items()):
                if track_id in tracked_ids or track_id in lost_ids:
                    continue
                record = raw_tracks.get(track_id)
                if record is not None and not record.terminated and old_state in {RECOVERABLE, LOST, ACTIVE, "tentative"}:
                    record.terminated = True
                    record.termination_reason = "removed_from_tracker"
                    termination_events.append(
                        {
                            "track_id": track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": record.termination_reason,
                        }
                    )
                    state_transitions.append(
                        {
                            "track_id": track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": old_state,
                            "to_state": TERMINATED,
                        }
                    )
                    previous_states[track_id] = TERMINATED
        else:
            for row in tracked_rows + lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                state = previous_states.get(row.track_id, ACTIVE if record.confirmed else "tentative")
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": state,
                        "detector_ran": False,
                    }
                )

        current_states: dict[str, str] = {}
        for track_id, record in raw_tracks.items():
            if record.terminated:
                current_states[track_id] = TERMINATED
            elif track_id in tracked_ids:
                current_states[track_id] = ACTIVE if record.confirmed else "tentative"
            elif track_id in lost_ids:
                current_states[track_id] = RECOVERABLE if not record.terminated else TERMINATED
        lost_ids_event = [track_id for track_id in active_before if track_id in lost_ids]
        terminated_ids_event = [item["track_id"] for item in termination_events if float(item["timestamp_seconds"]) == timestamp_seconds]
        new_ids_event = [item["track_id"] for item in new_id_events if float(item["timestamp_seconds"]) == timestamp_seconds]
        reactivated_ids_event = [item["track_id"] for item in reactivation_events if float(item["timestamp_seconds"]) == timestamp_seconds and item.get("success")]
        frame_event = {
            "source_frame_index": frame_payload["source_frame_index"],
            "processed_frame_index": processed_frame_index,
            "timestamp_seconds": timestamp_seconds,
            "detector_ran": detector_ran,
            "detections_found": (True if current_detections else False) if detector_ran else None,
            "detection_count": len(current_detections),
            "active_track_ids_before": active_before,
            "recoverable_track_ids_before": recoverable_before,
            "matched_track_ids": matched_track_ids,
            "reactivated_track_ids": reactivated_ids_event,
            "new_track_ids": new_ids_event,
            "lost_track_ids": lost_ids_event if detector_ran else [],
            "terminated_track_ids": terminated_ids_event if detector_ran else [],
            "active_track_ids_after": [track_id for track_id, state in current_states.items() if state in {ACTIVE, "tentative"}],
            "recoverable_track_ids_after": [track_id for track_id, state in current_states.items() if state == RECOVERABLE],
        }
        per_frame_events.append(frame_event)
        previous_states = current_states
        previous_snapshot = {row.track_id: {"track_id": row.track_id, "family": row.family, "class_name": row.class_name, "bbox_xyxy": list(row.bbox_xyxy)} for row in lost_rows}

        annotated_tracks = []
        for row in tracked_rows + lost_rows:
            annotated_tracks.append(
                {
                    "track_id": row.track_id,
                    "class_name": row.class_name,
                    "bbox_xyxy": list(row.bbox_xyxy),
                    "state": current_states.get(row.track_id, row.backend_state),
                }
            )
        annotated = _annotate_frame(frame=frame, timestamp_seconds=timestamp_seconds, detector_ran=detector_ran, tracks=annotated_tracks)
        writer.write(annotated)
        if new_ids_event or reactivated_ids_event or terminated_ids_event or (len(annotated_tracks) >= 4):
            debug_path = config.run_dir / "04_debug_frames" / f"debug_{processed_frame_index:06d}.jpg"
            cv2.imwrite(str(debug_path), annotated)

    writer.release()
    tracking_runtime_seconds = time.perf_counter() - tracking_started
    metrics = build_validation_metrics(
        records=raw_tracks,
        per_frame_events=per_frame_events,
        termination_events=termination_events,
        new_id_events=new_id_events,
        reactivation_events=reactivation_events,
    )
    checks = build_validation_checks(
        per_frame_events=per_frame_events,
        new_id_events=new_id_events,
        reactivation_events=reactivation_events,
        records=raw_tracks,
    )
    report = {
        "status": "success" if checks["passed"] and metrics["tracks_lost_due_to_skipped_detector_frame"] == 0 else "warning",
        "source_fps": video_info.source_fps,
        "processing_fps": config.processing_fps,
        "detector_fps": config.detector_fps,
        "processed_frames": len(per_frame_events),
        "detector_frames": len([item for item in per_frame_events if item["detector_ran"]]),
        "skipped_detector_frames": len([item for item in per_frame_events if not item["detector_ran"]]),
        "yolo_calls": len([item for item in per_frame_events if item["detector_ran"]]),
        "tracking_runtime_seconds": round(tracking_runtime_seconds, 6),
        **metrics,
        "validation_checks": checks,
        "separate_skipped_frame_code_path": True,
        "annotated_video_path": str(config.run_dir / "04_debug_frames" / "annotated_tracking.mp4"),
    }

    write_json(config.run_dir / "02_detections" / "detector_frame_schedule.json", {"status": "success", "frames": yolo_call_report})
    write_json(config.run_dir / "02_detections" / "yolo_detections.json", {"status": "success", "detections": yolo_detections})
    write_json(config.run_dir / "02_detections" / "yolo_call_report.json", {"status": "success", "calls": yolo_call_report})
    write_json(config.run_dir / "03_tracking" / "frame_tracking_events.json", {"status": "success", "events": per_frame_events})
    write_json(
        config.run_dir / "03_tracking" / "raw_tracks.json",
        {
            "status": "success",
            "tracks": [
                {
                    "track_id": record.track_id,
                    "family": record.family,
                    "class_name": record.class_name,
                    "created_timestamp_seconds": record.created_timestamp_seconds,
                    "created_zone": record.created_zone,
                    "confirmed": record.confirmed,
                    "reactivated_count": record.reactivated_count,
                    "terminated": record.terminated,
                    "termination_reason": record.termination_reason,
                    "observations": record.observations,
                }
                for record in raw_tracks.values()
            ],
        },
    )
    write_json(config.run_dir / "03_tracking" / "track_state_transitions.json", {"status": "success", "events": state_transitions})
    write_json(config.run_dir / "03_tracking" / "new_id_events.json", {"status": "success", "events": new_id_events})
    write_json(config.run_dir / "03_tracking" / "reactivation_events.json", {"status": "success", "events": reactivation_events})
    write_json(config.run_dir / "03_tracking" / "termination_events.json", {"status": "success", "events": termination_events})
    write_json(config.run_dir / "03_tracking" / "skipped_detector_frame_events.json", {"status": "success", "events": skipped_detector_events})
    write_json(config.run_dir / "05_reports" / "tracking_validation_report.json", report)
    lines = [
        "# Fixed 5 FPS ByteTrack Validation Report",
        "",
        f"- Run directory: {config.run_dir}",
        f"- Separate skipped-frame code path: {report['separate_skipped_frame_code_path']}",
        f"- Processed frames: {report['processed_frames']}",
        f"- Detector frames: {report['detector_frames']}",
        f"- Skipped detector frames: {report['skipped_detector_frames']}",
        f"- YOLO calls: {report['yolo_calls']}",
        f"- Raw tracks: {report['raw_track_ids']}",
        f"- Confirmed tracks: {report['confirmed_tracks']}",
        f"- Tentative tracks: {report['tentative_tracks']}",
        f"- Reactivated tracks: {report['reactivated_tracks']}",
        f"- Tracks lost due to skipped detector frame: {report['tracks_lost_due_to_skipped_detector_frame']}",
        "",
        "## Validation Checks",
    ]
    for warning in checks["warnings"] or ["No validation warnings."]:
        lines.append(f"- {warning}")
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(config.run_dir / "05_reports" / "tracking_validation_report.md", lines)
    write_html_from_markdown(config.run_dir / "05_reports" / "tracking_validation_report.html", markdown_text)
    print(f"run_dir={config.run_dir}")


if __name__ == "__main__":
    main()
