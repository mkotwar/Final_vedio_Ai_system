from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (case_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from continuous_mot_hybrid.bytetrack_backend import ByteTrackBackend
    from continuous_mot_hybrid.compare_verified_reid_results import compare_verified_results
    from continuous_mot_hybrid.detection_track_adapter import detections_from_rows
    from continuous_mot_hybrid.fixed_5fps_validation_core import (
        ACTIVE,
        RECOVERABLE,
        TERMINATED,
        TrackLifecycleRecord,
        build_validation_checks,
        build_validation_metrics,
        classify_zone,
        compute_new_id_reason,
        recovery_window_seconds,
        update_confirmation,
    )
    from continuous_mot_hybrid.live_botsort_reid_runner import LiveReidDetectionConfig, build_live_detection_feature_cache
    from continuous_mot_hybrid.local_reid_model_inventory import write_local_reid_inventory
    from continuous_mot_hybrid.reid_capability_audit import write_reid_capability_audit
    from continuous_mot_hybrid.reid_feature_cache import load_feature_cache
    from continuous_mot_hybrid.report_writer import write_html_from_markdown, write_json, write_markdown
    from continuous_mot_hybrid.tracker_backend_visualizer import save_visual_review_cases, select_visual_timestamps
    from continuous_mot_hybrid.tracker_yaml_builder import write_tracker_yaml
    from continuous_mot_hybrid.verified_botsort_reid_backend import VerifiedBotSortReidBackend
    from continuous_mot_hybrid.video_frame_stream import read_video_info, stream_processed_frames
    from tests.td_case2.config import repo_root as td_repo_root
else:
    from .bytetrack_backend import ByteTrackBackend
    from .compare_verified_reid_results import compare_verified_results
    from .detection_track_adapter import detections_from_rows
    from .fixed_5fps_validation_core import (
        ACTIVE,
        RECOVERABLE,
        TERMINATED,
        TrackLifecycleRecord,
        build_validation_checks,
        build_validation_metrics,
        classify_zone,
        compute_new_id_reason,
        recovery_window_seconds,
        update_confirmation,
    )
    from .live_botsort_reid_runner import LiveReidDetectionConfig, build_live_detection_feature_cache
    from .local_reid_model_inventory import write_local_reid_inventory
    from .reid_capability_audit import write_reid_capability_audit
    from .reid_feature_cache import load_feature_cache
    from .report_writer import write_html_from_markdown, write_json, write_markdown
    from .tracker_backend_visualizer import save_visual_review_cases, select_visual_timestamps
    from .tracker_yaml_builder import write_tracker_yaml
    from .verified_botsort_reid_backend import VerifiedBotSortReidBackend
    from .video_frame_stream import read_video_info, stream_processed_frames
    from tests.td_case2.config import repo_root as td_repo_root


@dataclass(frozen=True)
class VerifiedReidConfig:
    video_path: Path
    run_dir: Path
    camera_id: str
    camera_group: str
    camera_timezone: str
    processing_fps: float
    detector_fps: float
    device: str
    reid_model: str
    no_download: bool
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    match_thresh: float = 0.80
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    yolo_model_path: Path = Path("yolo11m.pt")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verified BoT-SORT ReID comparison against fixed ByteTrack.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--camera-id", default="test_cam_01")
    parser.add_argument("--camera-group", default="single_camera_comparison")
    parser.add_argument("--camera-timezone", default="Asia/Kolkata")
    parser.add_argument("--processing-fps", type=float, default=10.0)
    parser.add_argument("--detector-fps", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reid-model", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--run-dir")
    return parser


def _make_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = td_repo_root() / "debug_runs" / f"verified_botsort_reid_{video_path.stem}_{timestamp}"
    for child in ("00_audit", "01_shared", "02_bytetrack", "03_botsort_reid", "04_comparison", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_config(args: argparse.Namespace) -> VerifiedReidConfig:
    video_path = Path(args.video_path).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _make_run_dir(video_path)
    return VerifiedReidConfig(
        video_path=video_path,
        run_dir=run_dir,
        camera_id=str(args.camera_id),
        camera_group=str(args.camera_group),
        camera_timezone=str(args.camera_timezone),
        processing_fps=float(args.processing_fps),
        detector_fps=float(args.detector_fps),
        device=str(args.device),
        reid_model=str(args.reid_model),
        no_download=bool(args.no_download),
        yolo_model_path=(Path.cwd() / "yolo11m.pt").resolve(),
    )


def _annotate_frame(*, frame: Any, timestamp_seconds: float, detector_ran: bool, backend_label: str, tracks: list[dict[str, Any]]) -> Any:
    image = frame.copy()
    cv2.putText(image, f"{backend_label} t={timestamp_seconds:.2f}s {'DETECTOR' if detector_ran else 'SKIPPED'}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for track in tracks:
        x1, y1, x2, y2 = [int(round(float(value))) for value in track["bbox_xyxy"]]
        color = (40, 200, 40) if track["state"] == ACTIVE else (40, 180, 220)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"TID:{track['track_id']}"
        if track.get("reid_label") is not None:
            label = f"{label} REID:{track['reid_label']} SIM:{track.get('appearance_similarity','NA')}"
        cv2.putText(image, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)
    return image


def _replay_backend(
    *,
    backend_name: str,
    backend: Any,
    config: VerifiedReidConfig,
    cache_payload: dict[str, Any],
    feature_vectors_by_frame: dict[int, np.ndarray] | None,
    output_dir: Path,
    add_reid_labels: bool,
) -> dict[str, Any]:
    video_info = read_video_info(config.video_path)
    cached_by_processed = {int(item["processed_frame_index"]): item for item in cache_payload["cached_yolo_detections"]}
    _, _, _, iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    writer = cv2.VideoWriter(str(output_dir / "annotated_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), config.processing_fps, (video_info.width, video_info.height))
    per_frame_events: list[dict[str, Any]] = []
    raw_tracks: dict[str, TrackLifecycleRecord] = {}
    state_transitions: list[dict[str, Any]] = []
    new_id_events: list[dict[str, Any]] = []
    reactivation_events: list[dict[str, Any]] = []
    termination_events: list[dict[str, Any]] = []
    skipped_detector_events: list[dict[str, Any]] = []
    previous_states: dict[str, str] = {}
    previous_snapshot: dict[str, dict[str, Any]] = {}
    new_ids_from_boundaries = 0
    new_ids_from_interior = 0
    frames_for_visuals: dict[int, Any] = {}
    started = time.perf_counter()
    for frame_record, frame in iterator:
        frame_payload = frame_record.to_dict()
        processed_frame_index = int(frame_payload["processed_frame_index"])
        timestamp_seconds = float(frame_payload["timestamp_seconds"])
        detector_frame = cached_by_processed.get(processed_frame_index)
        detector_ran = detector_frame is not None
        active_before = [track_id for track_id, state in previous_states.items() if state in {ACTIVE, "tentative"}]
        recoverable_before = [track_id for track_id, state in previous_states.items() if state == RECOVERABLE]
        current_detections = detector_frame["detections"] if detector_frame is not None else []
        feature_vectors = feature_vectors_by_frame.get(processed_frame_index) if feature_vectors_by_frame is not None else None
        if detector_ran:
            if backend_name == "botsort_reid":
                snapshot = backend.update(detections=detections_from_rows(current_detections), frame=frame, feature_vectors=feature_vectors)
            else:
                snapshot = backend.update(detections=detections_from_rows(current_detections), frame=frame)
        else:
            snapshot = backend.handle_detector_skipped()
            skipped_detector_events.append({"processed_frame_index": processed_frame_index, "timestamp_seconds": timestamp_seconds, "active_track_ids_before": active_before, "recoverable_track_ids_before": recoverable_before})
        tracked_rows = [row for row in snapshot if row.backend_state == "tracked"]
        lost_rows = [row for row in snapshot if row.backend_state == "lost"]
        tracked_ids = [row.track_id for row in tracked_rows]
        lost_ids = [row.track_id for row in lost_rows]
        matched_track_ids = [row.track_id for row in tracked_rows if row.matched_detection_id]
        if detector_ran:
            for row in tracked_rows:
                zone = classify_zone(list(row.bbox_xyxy), frame_width=video_info.width, frame_height=video_info.height)
                record = raw_tracks.get(row.track_id)
                if record is None:
                    record = TrackLifecycleRecord(track_id=row.track_id, family=row.family, class_name=row.class_name, created_timestamp_seconds=timestamp_seconds, created_zone=zone, last_detector_confirmation_timestamp=timestamp_seconds)
                    raw_tracks[row.track_id] = record
                    reason = compute_new_id_reason(bbox_xyxy=list(row.bbox_xyxy), class_name=row.class_name, family=row.family, recoverable_tracks=list(previous_snapshot.values()), frame_width=video_info.width, frame_height=video_info.height)
                    if zone == "interior":
                        new_ids_from_interior += 1
                    else:
                        new_ids_from_boundaries += 1
                    new_id_events.append({"track_id": row.track_id, "timestamp_seconds": timestamp_seconds, "reason": reason, "zone": zone, "detector_ran": True, "backend": backend_name})
                if row.matched_detection_id:
                    record.detector_hit_timestamps.append(timestamp_seconds)
                    record.class_name = row.class_name
                    record.last_detector_confirmation_timestamp = timestamp_seconds
                    update_confirmation(record, timestamp_seconds=timestamp_seconds)
                current_state = ACTIVE if record.confirmed else "tentative"
                if previous_states.get(row.track_id) == RECOVERABLE and row.matched_detection_id:
                    record.reactivated_count += 1
                    reactivation_events.append({"track_id": row.track_id, "timestamp_seconds": timestamp_seconds, "success": True, "detector_ran": True, "backend": backend_name})
                    current_state = "reactivated"
                record.observations.append({"timestamp_seconds": timestamp_seconds, "source_frame_index": frame_payload["source_frame_index"], "processed_frame_index": processed_frame_index, "bbox_xyxy": list(row.bbox_xyxy), "state": current_state, "detector_ran": True, "matched_detection_id": row.matched_detection_id})
                if previous_states.get(row.track_id) != current_state:
                    state_transitions.append({"track_id": row.track_id, "timestamp_seconds": timestamp_seconds, "from_state": previous_states.get(row.track_id), "to_state": current_state})
                record.last_state = ACTIVE if record.confirmed else "tentative"
            for row in lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                recoverable_window = recovery_window_seconds(family=record.family, confirmed=record.confirmed)
                elapsed = timestamp_seconds - record.last_detector_confirmation_timestamp
                state = RECOVERABLE if elapsed <= recoverable_window else "lost"
                if elapsed > recoverable_window and not record.terminated:
                    record.terminated = True
                    record.termination_reason = "recovery_window_expired"
                    termination_events.append({"track_id": row.track_id, "timestamp_seconds": timestamp_seconds, "reason": record.termination_reason})
                    state = TERMINATED
                record.observations.append({"timestamp_seconds": timestamp_seconds, "source_frame_index": frame_payload["source_frame_index"], "processed_frame_index": processed_frame_index, "bbox_xyxy": list(row.bbox_xyxy), "state": state, "detector_ran": True})
                if previous_states.get(row.track_id) != state:
                    state_transitions.append({"track_id": row.track_id, "timestamp_seconds": timestamp_seconds, "from_state": previous_states.get(row.track_id), "to_state": state})
                record.last_state = state
            for track_id, old_state in list(previous_states.items()):
                if track_id in tracked_ids or track_id in lost_ids:
                    continue
                record = raw_tracks.get(track_id)
                if record is not None and not record.terminated and old_state in {RECOVERABLE, "lost", ACTIVE, "tentative"}:
                    record.terminated = True
                    record.termination_reason = "removed_from_tracker"
                    termination_events.append({"track_id": track_id, "timestamp_seconds": timestamp_seconds, "reason": record.termination_reason})
                    state_transitions.append({"track_id": track_id, "timestamp_seconds": timestamp_seconds, "from_state": old_state, "to_state": TERMINATED})
                    previous_states[track_id] = TERMINATED
        else:
            for row in tracked_rows + lost_rows:
                record = raw_tracks.get(row.track_id)
                if record is None:
                    continue
                state = previous_states.get(row.track_id, ACTIVE if record.confirmed else "tentative")
                record.observations.append({"timestamp_seconds": timestamp_seconds, "source_frame_index": frame_payload["source_frame_index"], "processed_frame_index": processed_frame_index, "bbox_xyxy": list(row.bbox_xyxy), "state": state, "detector_ran": False})
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
        per_frame_events.append({"source_frame_index": frame_payload["source_frame_index"], "processed_frame_index": processed_frame_index, "timestamp_seconds": timestamp_seconds, "detector_ran": detector_ran, "detections_found": (True if current_detections else False) if detector_ran else None, "detection_count": len(current_detections), "active_track_ids_before": active_before, "recoverable_track_ids_before": recoverable_before, "matched_track_ids": matched_track_ids, "reactivated_track_ids": reactivated_ids_event, "new_track_ids": new_ids_event, "lost_track_ids": lost_ids_event if detector_ran else [], "terminated_track_ids": terminated_ids_event if detector_ran else [], "active_track_ids_after": [track_id for track_id, state in current_states.items() if state in {ACTIVE, 'tentative'}], "recoverable_track_ids_after": [track_id for track_id, state in current_states.items() if state == RECOVERABLE]})
        previous_states = current_states
        previous_snapshot = {row.track_id: {"track_id": row.track_id, "family": row.family, "class_name": row.class_name, "bbox_xyxy": list(row.bbox_xyxy)} for row in lost_rows}
        annotated_tracks = []
        for row in tracked_rows + lost_rows:
            annotated_tracks.append({"track_id": row.track_id, "class_name": row.class_name, "bbox_xyxy": list(row.bbox_xyxy), "state": current_states.get(row.track_id, row.backend_state), "reid_label": ("match" if add_reid_labels and row.matched_detection_id else "no-match") if add_reid_labels else None, "appearance_similarity": "not_available"})
        annotated = _annotate_frame(frame=frame, timestamp_seconds=timestamp_seconds, detector_ran=detector_ran, backend_label=backend_name, tracks=annotated_tracks)
        writer.write(annotated)
        if len(frames_for_visuals) < 300:
            frames_for_visuals[processed_frame_index] = annotated.copy()
    writer.release()
    tracking_runtime_seconds = round(time.perf_counter() - started, 6)
    metrics = build_validation_metrics(records=raw_tracks, per_frame_events=per_frame_events, termination_events=termination_events, new_id_events=new_id_events, reactivation_events=reactivation_events)
    checks = build_validation_checks(per_frame_events=per_frame_events, new_id_events=new_id_events, reactivation_events=reactivation_events, records=raw_tracks)
    raw_track_rows = []
    for record in raw_tracks.values():
        start_timestamp = float(record.observations[0]["timestamp_seconds"]) if record.observations else record.created_timestamp_seconds
        end_timestamp = float(record.observations[-1]["timestamp_seconds"]) if record.observations else record.created_timestamp_seconds
        raw_track_rows.append({"tracker_id": record.track_id, "class": record.class_name, "object_family": record.family, "start_timestamp": start_timestamp, "end_timestamp": end_timestamp, "duration": round(max(0.0, end_timestamp - start_timestamp), 6), "detector_hit_count": len(record.detector_hit_timestamps), "state": record.last_state, "confirmed": record.confirmed, "termination_reason": record.termination_reason, "tracker_backend": backend_name, "observations": record.observations})
    report = {"status": "success" if checks["passed"] and metrics["tracks_lost_due_to_skipped_detector_frame"] == 0 else "warning", "processed_frames": len(per_frame_events), "detector_frames": len([item for item in per_frame_events if item["detector_ran"]]), "skipped_detector_frames": len([item for item in per_frame_events if not item["detector_ran"]]), "tracker_runtime_seconds": tracking_runtime_seconds, "end_to_end_replay_runtime_seconds": tracking_runtime_seconds, "average_ms_per_processed_frame": round((tracking_runtime_seconds * 1000.0) / max(len(per_frame_events), 1), 6), "realtime_factor": round((video_info.duration_seconds / tracking_runtime_seconds), 6) if tracking_runtime_seconds > 0 else "not_available", "peak_gpu_memory_mb": getattr(backend, "verification", {}).get("peak_gpu_vram_mb", "not_available"), "peak_system_memory_mb": "not_available", "active_tracks_final": len([item for item in previous_states.values() if item in {ACTIVE, 'tentative'}]), "lost_tracks_final": len([item for item in previous_states.values() if item == RECOVERABLE]), "removed_tracks_final": len([item for item in raw_tracks.values() if item.termination_reason == 'removed_from_tracker']), "interior_new_ids": new_ids_from_interior, "boundary_new_ids": new_ids_from_boundaries, "matched_detections": sum(len(item["matched_track_ids"]) for item in per_frame_events), "appearance_assisted_matches": getattr(backend, 'verification', {}).get('appearance_assisted_accepted_matches', 'not_available'), "appearance_rejections": getattr(backend, 'verification', {}).get('appearance_rejected_matches', 'not_available'), "ambiguous_appearance_matches": getattr(backend, 'verification', {}).get('ambiguous_appearance_matches', 'not_available'), **metrics, "validation_checks": checks, "same_detection_cache_checksum": cache_payload["detection_cache_checksum"]["sha256"]}
    write_json(output_dir / "resolved_tracker_config.json", {"status": "success", "backend": backend_name})
    write_json(output_dir / "frame_tracking_events.json", {"status": "success", "events": per_frame_events})
    write_json(output_dir / "raw_tracks.json", {"status": "success", "tracks": raw_track_rows})
    write_json(output_dir / "track_state_transitions.json", {"status": "success", "events": state_transitions})
    write_json(output_dir / "new_id_events.json", {"status": "success", "events": new_id_events})
    write_json(output_dir / "reactivation_events.json", {"status": "success", "events": reactivation_events})
    write_json(output_dir / "termination_events.json", {"status": "success", "events": termination_events})
    write_json(output_dir / "skipped_detector_frame_events.json", {"status": "success", "events": skipped_detector_events})
    write_json(output_dir / "skipped_frame_behavior_report.json", {"status": "success", "tracks_lost_due_to_skipped_detector_frame": metrics["tracks_lost_due_to_skipped_detector_frame"]})
    write_json(output_dir / "tracking_metrics.json", report)
    lines = [f"# {backend_name} Tracking Report", "", f"- Processed frames: {report['processed_frames']}", f"- Detector frames: {report['detector_frames']}", f"- Raw tracker IDs: {report['raw_track_ids']}", f"- Confirmed tracks: {report['confirmed_tracks']}", f"- Tentative tracks: {report['tentative_tracks']}", f"- Tracks lost due to skipped frames: {report['tracks_lost_due_to_skipped_detector_frame']}"]
    write_markdown(output_dir / "tracking_report.md", lines)
    write_html_from_markdown(output_dir / "tracking_report.html", "\n".join(lines) + "\n")
    return {"report": report, "raw_tracks": raw_track_rows, "frame_tracking_events": per_frame_events, "visual_frames": frames_for_visuals}


def main() -> None:
    config = _build_config(_arg_parser().parse_args())
    site_packages_root = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    audit = write_reid_capability_audit(output_dir=config.run_dir / "00_audit", site_packages_root=site_packages_root, yolo_model_path=config.yolo_model_path)
    inventory = write_local_reid_inventory(config.run_dir / "00_audit" / "local_reid_model_inventory.json")
    if config.reid_model != "auto":
        matches = [item for item in inventory["models"] if item["path"] == str(Path(config.reid_model).resolve()) and item["exists"]]
        if not matches:
            raise FileNotFoundError(f"Requested local ReID model was not found: {config.reid_model}")
    live_cache = build_live_detection_feature_cache(config=LiveReidDetectionConfig(video_path=config.video_path, processing_fps=config.processing_fps, detector_fps=config.detector_fps, confidence=config.yolo_confidence, iou=config.yolo_iou, model_path=config.yolo_model_path, device=config.device), shared_dir=config.run_dir / "01_shared")
    feature_vectors_by_frame = load_feature_cache(live_cache["feature_cache_path"])
    write_tracker_yaml(source_yaml=site_packages_root / "ultralytics" / "cfg" / "trackers" / "botsort.yaml", destination_yaml=config.run_dir / "03_botsort_reid" / "botsort_reid.yaml", overrides={"tracker_type": "botsort", "track_high_thresh": config.track_high_thresh, "track_low_thresh": config.track_low_thresh, "new_track_thresh": config.track_high_thresh, "track_buffer": max(1, int(round(config.detector_fps * 1.0))), "match_thresh": config.match_thresh, "fuse_score": True, "with_reid": True, "model": config.reid_model})
    write_tracker_yaml(source_yaml=site_packages_root / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml", destination_yaml=config.run_dir / "02_bytetrack" / "bytetrack.yaml", overrides={"tracker_type": "bytetrack", "track_high_thresh": config.track_high_thresh, "track_low_thresh": config.track_low_thresh, "new_track_thresh": config.track_high_thresh, "track_buffer": max(1, int(round(config.detector_fps * 1.0))), "match_thresh": config.match_thresh, "fuse_score": False})
    bytetrack_payload = _replay_backend(backend_name="bytetrack", backend=ByteTrackBackend(track_high_thresh=config.track_high_thresh, track_low_thresh=config.track_low_thresh, match_thresh=config.match_thresh, track_buffer_frames=max(1, int(round(config.detector_fps * 1.0)))), config=config, cache_payload=live_cache, feature_vectors_by_frame=None, output_dir=config.run_dir / "02_bytetrack", add_reid_labels=False)
    botsort_backend = VerifiedBotSortReidBackend(track_high_thresh=config.track_high_thresh, track_low_thresh=config.track_low_thresh, match_thresh=config.match_thresh, track_buffer_frames=max(1, int(round(config.detector_fps * 1.0))), gmc_method="sparseOptFlow", model=config.reid_model, device=None if config.device == "auto" else config.device)
    botsort_payload = _replay_backend(backend_name="botsort_reid", backend=botsort_backend, config=config, cache_payload=live_cache, feature_vectors_by_frame=feature_vectors_by_frame, output_dir=config.run_dir / "03_botsort_reid", add_reid_labels=True)
    reid_verification = botsort_backend.write_verification(config.run_dir / "03_botsort_reid" / "reid_runtime_verification.json")
    timestamps = select_visual_timestamps(bytetrack_events=bytetrack_payload["frame_tracking_events"], fallback_duration_seconds=float(live_cache["video_info"]["duration_seconds"]))
    visual_manifest = save_visual_review_cases(video_path=config.video_path, processing_fps=config.processing_fps, output_dir=config.run_dir / "04_comparison", timestamps=timestamps, frames_by_backend={"bytetrack": bytetrack_payload["visual_frames"], "botsort_reid": botsort_payload["visual_frames"]})
    compare_verified_results(output_dir=config.run_dir / "04_comparison", bytetrack_report=bytetrack_payload["report"], botsort_report=botsort_payload["report"], reid_verification=reid_verification, visual_manifest=visual_manifest)
    print(f"run_dir={config.run_dir}")


if __name__ == "__main__":
    main()
