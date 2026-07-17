from __future__ import annotations

import argparse
import time
import sys
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
    from continuous_mot_hybrid.recoverable_track_matcher import match_recoverable_tracks
    from continuous_mot_hybrid.recoverable_track_store import RecoverableTrackSnapshot, RecoverableTrackStore
    from continuous_mot_hybrid.recovery_candidate_index import RecoveryCandidateIndex
    from continuous_mot_hybrid.recovery_scoring import RecoveryScoringConfig, summarize_scores
    from continuous_mot_hybrid.report_writer import write_html_from_markdown, write_json, write_markdown
    from continuous_mot_hybrid.stable_identity_manager import StableIdentityManager
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
    from .recoverable_track_matcher import match_recoverable_tracks
    from .recoverable_track_store import RecoverableTrackSnapshot, RecoverableTrackStore
    from .recovery_candidate_index import RecoveryCandidateIndex
    from .recovery_scoring import RecoveryScoringConfig, summarize_scores
    from .report_writer import write_html_from_markdown, write_json, write_markdown
    from .stable_identity_manager import StableIdentityManager
    from .video_frame_stream import read_video_info, stream_processed_frames
    from .yolo_detector import YoloDetector, resolve_model_specs
    from tests.td_case2.config import ENV_OBJECT_YOLO_MODEL_PATH, ENV_PERSON_YOLO_MODEL_PATH, ENV_YOLO_MODEL_PATH, repo_root as td_repo_root, resolve_case_path


@dataclass(frozen=True)
class FixedReactivationConfig:
    video_path: Path
    run_dir: Path
    camera_id: str
    camera_group: str
    camera_timezone: str
    processing_fps: float
    detector_fps: float
    device: str
    enable_recovery_histogram: bool
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    match_thresh: float = 0.80
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed 5 FPS ByteTrack reactivation validation.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--camera-id", default="test_cam_01")
    parser.add_argument("--camera-group", default="single_camera_comparison")
    parser.add_argument("--camera-timezone", default="Asia/Kolkata")
    parser.add_argument("--processing-fps", type=float, default=10.0)
    parser.add_argument("--detector-fps", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir")
    parser.add_argument("--enable-recovery-histogram", action="store_true")
    return parser


def _resolve_models() -> tuple[Path | None, Path | None, Path | None]:
    def _path(env_name: str) -> Path | None:
        raw = str(__import__("os").environ.get(env_name, "")).strip()
        return resolve_case_path(raw) if raw else None

    return _path(ENV_PERSON_YOLO_MODEL_PATH), _path(ENV_OBJECT_YOLO_MODEL_PATH), _path(ENV_YOLO_MODEL_PATH)


def _make_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = td_repo_root() / "debug_runs" / f"fixed_5fps_reactivation_{video_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("01_video", "02_detections", "03_tracking", "04_recovery", "05_debug_frames", "06_reports", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_config(args: argparse.Namespace) -> FixedReactivationConfig:
    video_path = Path(args.video_path).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _make_run_dir(video_path)
    if args.run_dir:
        for child in ("01_video", "02_detections", "03_tracking", "04_recovery", "05_debug_frames", "06_reports", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
    return FixedReactivationConfig(
        video_path=video_path,
        run_dir=run_dir,
        camera_id=args.camera_id,
        camera_group=args.camera_group,
        camera_timezone=args.camera_timezone,
        processing_fps=float(args.processing_fps),
        detector_fps=float(args.detector_fps),
        device=str(args.device),
        enable_recovery_histogram=bool(args.enable_recovery_histogram),
    )


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return ((float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0, (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0)


def _direction_group(current_bbox: list[float], previous_bbox: list[float] | None) -> str:
    if previous_bbox is None:
        return "unknown"
    current_center = _bbox_center(current_bbox)
    previous_center = _bbox_center(previous_bbox)
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    if abs(dx) >= abs(dy) and dx > 4.0:
        return "left_to_right"
    if abs(dx) >= abs(dy) and dx < -4.0:
        return "right_to_left"
    if abs(dy) > abs(dx) and dy > 4.0:
        return "top_to_bottom"
    if abs(dy) > abs(dx) and dy < -4.0:
        return "bottom_to_top"
    return "unknown"


def _crop_histogram(frame: Any, bbox_xyxy: list[float]) -> list[float] | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(float(bbox_xyxy[0]))))
    y1 = max(0, min(height - 1, int(float(bbox_xyxy[1]))))
    x2 = max(x1 + 1, min(width, int(float(bbox_xyxy[2]))))
    y2 = max(y1 + 1, min(height, int(float(bbox_xyxy[3]))))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return [round(float(value), 6) for value in hist.tolist()]


def _color_for_state(state: str) -> tuple[int, int, int]:
    return {
        ACTIVE: (40, 200, 40),
        "tentative": (40, 180, 220),
        RECOVERABLE: (0, 180, 255),
        "reactivated": (180, 40, 220),
        "possible": (255, 200, 0),
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
        label = f"raw={track['track_id']} local={track.get('local_object_id', '-') } {track['class_name']} {state}"
        cv2.putText(image, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)
    return image


def _recoverable_payload(entries: list[RecoverableTrackSnapshot]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": item.last_tracker_id,
            "family": item.object_family,
            "class_name": item.stable_class,
            "bbox_xyxy": list(item.last_detector_supported_bbox),
        }
        for item in entries
    ]


def _build_snapshot_for_record(
    *,
    local_object_id: int,
    record: TrackLifecycleRecord,
    identity_manager: StableIdentityManager,
    timestamp_seconds: float,
    recovery_reason: str,
    frame_width: int,
    frame_height: int,
    frame: Any,
    enable_histogram: bool,
) -> RecoverableTrackSnapshot | None:
    detector_observations = [item for item in record.observations if bool(item.get("detector_ran"))]
    if not detector_observations:
        return None
    last = detector_observations[-1]
    previous = detector_observations[-2] if len(detector_observations) >= 2 else None
    last_bbox = list(last["bbox_xyxy"])
    previous_bbox = list(previous["bbox_xyxy"]) if previous is not None else None
    last_center = _bbox_center(last_bbox)
    if previous_bbox is None:
        estimated_velocity = (0.0, 0.0)
    else:
        previous_center = _bbox_center(previous_bbox)
        delta_t = max(1e-6, float(last["timestamp_seconds"]) - float(previous["timestamp_seconds"]))
        estimated_velocity = (
            (last_center[0] - previous_center[0]) / delta_t,
            (last_center[1] - previous_center[1]) / delta_t,
        )
    width = max(1.0, float(last_bbox[2]) - float(last_bbox[0]))
    height = max(1.0, float(last_bbox[3]) - float(last_bbox[1]))
    histogram = _crop_histogram(frame, last_bbox) if enable_histogram else None
    identity = identity_manager.identities[local_object_id]
    return RecoverableTrackSnapshot(
        local_object_id=local_object_id,
        last_tracker_id=record.track_id,
        tracker_id_history=list(identity.tracker_id_history),
        object_family=record.family,
        stable_class=identity.stable_class,
        class_votes=dict(identity.class_votes),
        last_detector_supported_bbox=last_bbox,
        previous_detector_supported_bbox=previous_bbox,
        last_center=last_center,
        estimated_velocity=estimated_velocity,
        last_timestamp_seconds=float(record.observations[-1]["timestamp_seconds"]),
        last_detector_timestamp_seconds=float(last["timestamp_seconds"]),
        track_duration_seconds=max(0.0, float(record.observations[-1]["timestamp_seconds"]) - float(record.observations[0]["timestamp_seconds"])),
        detector_hit_count=len(detector_observations),
        entry_zone=record.created_zone,
        likely_exit_zone=classify_zone(last_bbox, frame_width=frame_width, frame_height=frame_height),
        movement_direction=_direction_group(last_bbox, previous_bbox),
        bbox_width=width,
        bbox_height=height,
        bbox_area=width * height,
        aspect_ratio=width / max(height, 1e-6),
        detector_confidence=float(last.get("detector_confidence") or 0.0),
        recovery_expiry_timestamp=float(last["timestamp_seconds"]) + recovery_window_seconds(family=record.family, confirmed=record.confirmed),
        recovery_reason=recovery_reason,
        histogram_descriptor=histogram,
    )


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
    identity_manager = StableIdentityManager()
    recoverable_store = RecoverableTrackStore()
    scoring_config = RecoveryScoringConfig()
    candidate_index = RecoveryCandidateIndex(frame_width=video_info.width, frame_height=video_info.height)
    raw_track_to_local_object_id: dict[str, int] = {}
    possible_recoveries: list[dict[str, Any]] = []

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
            "reactivation_layer": "enabled",
            "enable_recovery_histogram": config.enable_recovery_histogram,
            "device": config.device,
        },
    )

    _, _, _, frame_iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    writer = cv2.VideoWriter(
        str(config.run_dir / "05_debug_frames" / "annotated_tracking.mp4"),
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
    recovery_attempts_all: list[dict[str, Any]] = []
    tracking_started = time.perf_counter()

    previous_states: dict[str, str] = {}
    previous_lost_rows: dict[str, dict[str, Any]] = {}

    for frame_record, frame in frame_iterator:
        frame_payload = frame_record.to_dict()
        processed_frame_index = int(frame_payload["processed_frame_index"])
        timestamp_seconds = float(frame_payload["timestamp_seconds"])
        detector_ran = detector_should_run(processed_frame_index=processed_frame_index, processing_fps=config.processing_fps, detector_fps=config.detector_fps)
        active_before = [track_id for track_id, state in previous_states.items() if state in {ACTIVE, "tentative"}]
        recoverable_before = [track_id for track_id, state in previous_states.items() if state == RECOVERABLE]
        current_detections: list[dict[str, Any]] = []

        expired_entries = recoverable_store.expire(timestamp_seconds=timestamp_seconds)
        for expired in expired_entries:
            identity_manager.mark_terminated(local_object_id=expired.local_object_id, reason="recovery_window_expired")

        if detector_ran:
            current_detections = detector.detect(
                frame=frame,
                frame_record=frame_payload,
                scheduler_state="fixed_5fps",
                detector_reason="fixed_interval",
            )
            for detection in current_detections:
                previous_bbox = None
                for lost_row in previous_lost_rows.values():
                    if lost_row["family"] == detection["family"]:
                        previous_bbox = list(lost_row["bbox_xyxy"])
                        break
                detection["zone"] = classify_zone(list(detection["bbox_xyxy"]), frame_width=video_info.width, frame_height=video_info.height)
                detection["frame_width"] = video_info.width
                detection["frame_height"] = video_info.height
                detection["direction_group"] = _direction_group(list(detection["bbox_xyxy"]), previous_bbox)
                detection["histogram_descriptor"] = _crop_histogram(frame, list(detection["bbox_xyxy"])) if config.enable_recovery_histogram else None
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
        previous_recoverable_entries = recoverable_store.active_entries(timestamp_seconds=timestamp_seconds)
        raw_new_rows = [row for row in tracked_rows if row.track_id not in raw_tracks]

        accepted_by_tracker_id: dict[str, dict[str, Any]] = {}
        if detector_ran and raw_new_rows and previous_recoverable_entries:
            detection_by_tracker_id = {
                row.track_id: next((item for item in current_detections if item["detection_id"] == row.matched_detection_id), None)
                for row in raw_new_rows
                if row.matched_detection_id
            }
            candidate_entries_by_tracker_id: dict[str, list[RecoverableTrackSnapshot]] = {}
            for row in raw_new_rows:
                detection = detection_by_tracker_id.get(row.track_id)
                if detection is None:
                    continue
                candidate_entries_by_tracker_id[row.track_id] = candidate_index.query(
                    unmatched_detection={**detection, "tracker_id": row.track_id},
                    recoverable_entries=previous_recoverable_entries,
                    timestamp_seconds=timestamp_seconds,
                )
            match_result = match_recoverable_tracks(
                unmatched_detections=[
                    {
                        **detection_by_tracker_id[row.track_id],
                        "tracker_id": row.track_id,
                    }
                    for row in raw_new_rows
                    if row.track_id in detection_by_tracker_id
                ],
                candidate_entries_by_tracker_id=candidate_entries_by_tracker_id,
                timestamp_seconds=timestamp_seconds,
                scoring_config=scoring_config,
            )
            recovery_attempts_all.extend(match_result.all_attempts)
            accepted_by_tracker_id = {str(item["new_tracker_id"]): item for item in match_result.accepted_matches}
            for item in match_result.possible_matches:
                possible_recoveries.append(item)
                local_object_id = int(item["proposed_local_object_id"])
                identity_manager.add_possible_recovery_link(local_object_id=local_object_id, payload=item)
            for item in match_result.accepted_matches:
                local_object_id = int(item["proposed_local_object_id"])
                recoverable_store.remove(local_object_id)
                old_tracker_id = str(item["previous_tracker_id"])
                raw_track_to_local_object_id.pop(old_tracker_id, None)
                raw_track_to_local_object_id[str(item["new_tracker_id"])] = local_object_id
                identity_manager.remap_tracker(
                    previous_tracker_id=old_tracker_id,
                    new_tracker_id=str(item["new_tracker_id"]),
                    local_object_id=local_object_id,
                    timestamp_seconds=timestamp_seconds,
                    recovery_score=float(item["total_score"]),
                )
                identity_manager.add_gap_event(
                    local_object_id=local_object_id,
                    gap_start_seconds=float(previous_lost_rows.get(old_tracker_id, {}).get("timestamp_seconds", timestamp_seconds)),
                    gap_end_seconds=timestamp_seconds,
                    recovery_score=float(item["total_score"]),
                )
                reactivation_events.append(
                    {
                        "track_id": str(item["new_tracker_id"]),
                        "timestamp_seconds": timestamp_seconds,
                        "success": True,
                        "detector_ran": True,
                        "local_object_id": local_object_id,
                        "previous_tracker_id": old_tracker_id,
                        "recovery_score": float(item["total_score"]),
                        "final_decision": "accepted",
                    }
                )

        if detector_ran:
            for row in tracked_rows:
                zone = classify_zone(list(row.bbox_xyxy), frame_width=video_info.width, frame_height=video_info.height)
                record = raw_tracks.get(row.track_id)
                local_object_id = raw_track_to_local_object_id.get(row.track_id)
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
                    accepted = accepted_by_tracker_id.get(row.track_id)
                    if accepted is None:
                        local_object_id = identity_manager.create_identity(
                            tracker_id=row.track_id,
                            object_family=row.family,
                            class_name=row.class_name,
                            timestamp_seconds=timestamp_seconds,
                            zone=zone,
                        )
                        raw_track_to_local_object_id[row.track_id] = local_object_id
                    else:
                        local_object_id = int(accepted["proposed_local_object_id"])
                    reason = "reactivated_existing_object" if accepted is not None else compute_new_id_reason(
                        bbox_xyxy=list(row.bbox_xyxy),
                        class_name=row.class_name,
                        family=row.family,
                        recoverable_tracks=_recoverable_payload(previous_recoverable_entries),
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
                            "local_object_id": local_object_id,
                        }
                    )
                if local_object_id is None:
                    local_object_id = raw_track_to_local_object_id.get(row.track_id)
                if local_object_id is None:
                    local_object_id = identity_manager.create_identity(
                        tracker_id=row.track_id,
                        object_family=row.family,
                        class_name=row.class_name,
                        timestamp_seconds=timestamp_seconds,
                        zone=zone,
                    )
                    raw_track_to_local_object_id[row.track_id] = local_object_id
                if row.matched_detection_id:
                    record.detector_hit_timestamps.append(timestamp_seconds)
                    record.class_name = row.class_name
                    record.last_detector_confirmation_timestamp = timestamp_seconds
                    update_confirmation(record, timestamp_seconds=timestamp_seconds)
                current_state = ACTIVE if record.confirmed else "tentative"
                if row.track_id in accepted_by_tracker_id:
                    record.reactivated_count += 1
                    current_state = "reactivated"
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": current_state,
                        "detector_ran": True,
                        "detector_confidence": row.matched_detection_confidence,
                    }
                )
                identity_manager.add_observation(
                    local_object_id=local_object_id,
                    tracker_id=row.track_id,
                    class_name=row.class_name,
                    timestamp_seconds=timestamp_seconds,
                    source_frame_index=int(frame_payload["source_frame_index"]),
                    processed_frame_index=processed_frame_index,
                    bbox_xyxy=list(row.bbox_xyxy),
                    state=current_state,
                    detector_ran=True,
                    detector_confidence=row.matched_detection_confidence,
                )
                if record.confirmed:
                    identity_manager.mark_confirmed(local_object_id=local_object_id)
                if previous_states.get(row.track_id) != current_state:
                    state_transitions.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": previous_states.get(row.track_id),
                            "to_state": current_state,
                            "local_object_id": local_object_id,
                        }
                    )
                record.last_state = ACTIVE if record.confirmed else "tentative"
            for row in lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                local_object_id = raw_track_to_local_object_id.get(row.track_id)
                recoverable_window = recovery_window_seconds(family=record.family, confirmed=record.confirmed)
                elapsed = timestamp_seconds - record.last_detector_confirmation_timestamp
                state = RECOVERABLE if elapsed <= recoverable_window else LOST
                if state == RECOVERABLE and local_object_id is not None and recoverable_store.get(local_object_id) is None:
                    snapshot_entry = _build_snapshot_for_record(
                        local_object_id=local_object_id,
                        record=record,
                        identity_manager=identity_manager,
                        timestamp_seconds=timestamp_seconds,
                        recovery_reason="tracker_marked_lost",
                        frame_width=video_info.width,
                        frame_height=video_info.height,
                        frame=frame,
                        enable_histogram=config.enable_recovery_histogram,
                    )
                    if snapshot_entry is not None:
                        recoverable_store.add(snapshot_entry)
                        identity_manager.release_tracker(row.track_id)
                if elapsed > recoverable_window and not record.terminated:
                    record.terminated = True
                    record.termination_reason = "recovery_window_expired"
                    termination_events.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": record.termination_reason,
                            "local_object_id": local_object_id,
                        }
                    )
                    if local_object_id is not None:
                        identity_manager.mark_terminated(local_object_id=local_object_id, reason=record.termination_reason)
                    state = TERMINATED
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": state,
                        "detector_ran": True,
                        "detector_confidence": row.matched_detection_confidence,
                    }
                )
                if local_object_id is not None:
                    identity_manager.add_observation(
                        local_object_id=local_object_id,
                        tracker_id=row.track_id,
                        class_name=row.class_name,
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=int(frame_payload["source_frame_index"]),
                        processed_frame_index=processed_frame_index,
                        bbox_xyxy=list(row.bbox_xyxy),
                        state=state,
                        detector_ran=True,
                        detector_confidence=row.matched_detection_confidence,
                    )
                if previous_states.get(row.track_id) != state:
                    state_transitions.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": previous_states.get(row.track_id),
                            "to_state": state,
                            "local_object_id": local_object_id,
                        }
                    )
                record.last_state = state
            for track_id, old_state in list(previous_states.items()):
                if track_id in tracked_ids or track_id in lost_ids:
                    continue
                record = raw_tracks.get(track_id)
                if record is not None and not record.terminated and old_state in {RECOVERABLE, LOST, ACTIVE, "tentative"}:
                    local_object_id = raw_track_to_local_object_id.get(track_id)
                    if local_object_id is not None and recoverable_store.get(local_object_id) is None:
                        snapshot_entry = _build_snapshot_for_record(
                            local_object_id=local_object_id,
                            record=record,
                            identity_manager=identity_manager,
                            timestamp_seconds=timestamp_seconds,
                            recovery_reason="removed_from_tracker",
                            frame_width=video_info.width,
                            frame_height=video_info.height,
                            frame=frame,
                            enable_histogram=config.enable_recovery_histogram,
                        )
                        if snapshot_entry is not None:
                            recoverable_store.add(snapshot_entry)
                            identity_manager.release_tracker(track_id)
                    record.terminated = True
                    record.termination_reason = "removed_from_tracker"
                    termination_events.append(
                        {
                            "track_id": track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": record.termination_reason,
                            "local_object_id": local_object_id,
                        }
                    )
                    state_transitions.append(
                        {
                            "track_id": track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "from_state": old_state,
                            "to_state": TERMINATED,
                            "local_object_id": local_object_id,
                        }
                    )
                    previous_states[track_id] = TERMINATED
        else:
            for row in tracked_rows + lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                local_object_id = raw_track_to_local_object_id.get(row.track_id)
                state = previous_states.get(row.track_id, ACTIVE if record.confirmed else "tentative")
                record.observations.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "source_frame_index": frame_payload["source_frame_index"],
                        "processed_frame_index": processed_frame_index,
                        "bbox_xyxy": list(row.bbox_xyxy),
                        "state": state,
                        "detector_ran": False,
                        "detector_confidence": row.matched_detection_confidence,
                    }
                )
                if local_object_id is not None:
                    identity_manager.add_observation(
                        local_object_id=local_object_id,
                        tracker_id=row.track_id,
                        class_name=row.class_name,
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=int(frame_payload["source_frame_index"]),
                        processed_frame_index=processed_frame_index,
                        bbox_xyxy=list(row.bbox_xyxy),
                        state=state,
                        detector_ran=False,
                        detector_confidence=row.matched_detection_confidence,
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
        previous_lost_rows = {
            row.track_id: {
                "track_id": row.track_id,
                "family": row.family,
                "class_name": row.class_name,
                "bbox_xyxy": list(row.bbox_xyxy),
                "timestamp_seconds": timestamp_seconds,
            }
            for row in lost_rows
        }

        annotated_tracks = []
        for row in tracked_rows + lost_rows:
            annotated_tracks.append(
                {
                    "track_id": row.track_id,
                    "local_object_id": raw_track_to_local_object_id.get(row.track_id),
                    "class_name": row.class_name,
                    "bbox_xyxy": list(row.bbox_xyxy),
                    "state": current_states.get(row.track_id, row.backend_state),
                }
            )
        annotated = _annotate_frame(frame=frame, timestamp_seconds=timestamp_seconds, detector_ran=detector_ran, tracks=annotated_tracks)
        writer.write(annotated)
        if new_ids_event or reactivated_ids_event or terminated_ids_event or len(annotated_tracks) >= 4:
            debug_path = config.run_dir / "05_debug_frames" / f"debug_{processed_frame_index:06d}.jpg"
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
    local_id_durations = []
    for identity in identity_manager.identities.values():
        if identity.observations:
            local_id_durations.append(max(0.0, float(identity.observations[-1]["timestamp_seconds"]) - float(identity.observations[0]["timestamp_seconds"])))
    score_summary = summarize_scores(recovery_attempts_all)
    stable_report = {
        "raw_tracker_ids": len(raw_tracks),
        "stable_local_object_ids": len(identity_manager.identities),
        "tracker_id_remap_count": len(identity_manager.tracker_id_remap_events),
        "recoverable_entries_created": recoverable_store.entries_created,
        "recoverable_entries_expired": recoverable_store.expired_entries,
        "possible_recoveries": len(possible_recoveries),
        "local_objects_under_0_5_seconds": len([value for value in local_id_durations if value < 0.5]),
        "local_objects_under_1_0_seconds": len([value for value in local_id_durations if value < 1.0]),
        "removed_tracker_events": len([item for item in termination_events if item["reason"] == "removed_from_tracker"]),
        "application_level_terminations": len([identity for identity in identity_manager.identities.values() if identity.terminated]),
    }
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
        "stable_identity_metrics": stable_report,
        "recovery_candidate_report": candidate_index.build_report(),
        "recovery_score_summary": score_summary,
        "validation_checks": checks,
        "separate_skipped_frame_code_path": True,
        "annotated_video_path": str(config.run_dir / "05_debug_frames" / "annotated_tracking.mp4"),
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
    write_json(config.run_dir / "04_recovery" / "stable_identity_mappings.json", identity_manager.build_mappings_payload())
    write_json(config.run_dir / "04_recovery" / "stable_identity_timelines.json", identity_manager.build_timelines_payload())
    write_json(config.run_dir / "04_recovery" / "stable_identity_gaps.json", identity_manager.build_gap_payload())
    write_json(config.run_dir / "04_recovery" / "recoverable_track_store.json", recoverable_store.build_snapshot_payload())
    write_json(config.run_dir / "04_recovery" / "possible_recoveries.json", {"status": "success", "events": possible_recoveries})
    write_json(config.run_dir / "06_reports" / "tracking_validation_report.json", report)
    lines = [
        "# Fixed 5 FPS Reactivation Validation Report",
        "",
        f"- Run directory: {config.run_dir}",
        f"- Separate skipped-frame code path: {report['separate_skipped_frame_code_path']}",
        f"- Raw tracker IDs: {report['raw_track_ids']}",
        f"- Stable local-object IDs: {report['stable_identity_metrics']['stable_local_object_ids']}",
        f"- Tracker-ID remaps: {report['stable_identity_metrics']['tracker_id_remap_count']}",
        f"- Recoverable entries created: {report['stable_identity_metrics']['recoverable_entries_created']}",
        f"- Reactivation attempts: {report['reactivation_attempts']}",
        f"- Successful reactivations: {report['successful_reactivations']}",
        f"- Failed reactivations: {report['failed_reactivations']}",
        f"- Possible recoveries: {report['stable_identity_metrics']['possible_recoveries']}",
        f"- Tracks lost due to skipped detector frame: {report['tracks_lost_due_to_skipped_detector_frame']}",
        "",
        "## Validation Checks",
    ]
    for warning in checks["warnings"] or ["No validation warnings."]:
        lines.append(f"- {warning}")
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(config.run_dir / "06_reports" / "tracking_validation_report.md", lines)
    write_html_from_markdown(config.run_dir / "06_reports" / "tracking_validation_report.html", markdown_text)
    print(f"run_dir={config.run_dir}")


if __name__ == "__main__":
    main()
