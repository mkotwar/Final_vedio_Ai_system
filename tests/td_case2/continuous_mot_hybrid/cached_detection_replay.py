from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2

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


def _color_for_state(state: str) -> tuple[int, int, int]:
    return {
        ACTIVE: (40, 200, 40),
        "tentative": (40, 180, 220),
        RECOVERABLE: (0, 180, 255),
        "reactivated": (180, 40, 220),
        TERMINATED: (120, 120, 120),
    }.get(state, (200, 200, 200))


def annotate_frame(*, frame: Any, timestamp_seconds: float, detector_ran: bool, backend_label: str, tracks: list[dict[str, Any]]) -> Any:
    image = frame.copy()
    cv2.putText(
        image,
        f"{backend_label} t={timestamp_seconds:.2f}s {'DETECTOR' if detector_ran else 'SKIPPED'}",
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
        cv2.putText(
            image,
            f"{track['track_id']} {track['class_name']} {state}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def build_detection_cache(*, config: Any, detector: Any, shared_dir: Path) -> dict[str, Any]:
    video_info = read_video_info(config.video_path)
    write_json(
        shared_dir / "video_info.json",
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
        shared_dir / "resolved_detection_config.json",
        {
            "status": "success",
            "processing_fps": config.processing_fps,
            "detector_fps": config.detector_fps,
            "yolo_confidence": config.yolo_confidence,
            "yolo_iou": config.yolo_iou,
            "device": config.device,
            "adaptive_scheduler": "disabled",
        },
    )
    _, _, _, iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    schedule: list[dict[str, Any]] = []
    cached_detector_frames: list[dict[str, Any]] = []
    yolo_calls: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame_record, frame in iterator:
        frame_payload = frame_record.to_dict()
        processed_frame_index = int(frame_payload["processed_frame_index"])
        detector_ran = detector_should_run(
            processed_frame_index=processed_frame_index,
            processing_fps=config.processing_fps,
            detector_fps=config.detector_fps,
        )
        if detector_ran:
            detections = detector.detect(
                frame=frame,
                frame_record=frame_payload,
                scheduler_state="fixed_5fps",
                detector_reason="shared_cache",
            )
            cached_detector_frames.append(
                {
                    "source_frame_index": int(frame_payload["source_frame_index"]),
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": round(float(frame_payload["timestamp_seconds"]), 6),
                    "detector_ran": True,
                    "image_width": int(frame.shape[1]),
                    "image_height": int(frame.shape[0]),
                    "detections": [
                        {
                            "detection_id": item["detection_id"],
                            "bbox_xyxy": [float(value) for value in item["bbox_xyxy"]],
                            "confidence": float(item["confidence"]),
                            "class_id": int(item["class_id"]),
                            "class_name": str(item["class_name"]),
                            "object_family": str(item["family"]),
                        }
                        for item in detections
                    ],
                }
            )
            yolo_calls.append(
                {
                    "processed_frame_index": processed_frame_index,
                    "timestamp_seconds": round(float(frame_payload["timestamp_seconds"]), 6),
                    "detector_ran": True,
                    "detection_count": len(detections),
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
    runtime_seconds = round(time.perf_counter() - started, 6)
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
    }


def replay_backend(
    *,
    backend_name: str,
    backend: Any,
    config: Any,
    cache_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    video_info = read_video_info(config.video_path)
    cached_by_processed = {
        int(item["processed_frame_index"]): item
        for item in cache_payload["cached_yolo_detections"]
    }
    _, _, _, iterator = stream_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps, debug_frames_dir=None)
    writer = cv2.VideoWriter(
        str(output_dir / "annotated_tracking.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.processing_fps,
        (video_info.width, video_info.height),
    )
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
        if detector_ran:
            snapshot = backend.update(detections=detections_from_rows(current_detections), frame=frame)
        else:
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
                    if zone == "interior":
                        new_ids_from_interior += 1
                    else:
                        new_ids_from_boundaries += 1
                    new_id_events.append(
                        {
                            "track_id": row.track_id,
                            "timestamp_seconds": timestamp_seconds,
                            "reason": reason,
                            "zone": zone,
                            "detector_ran": True,
                            "backend": backend_name,
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
                            "backend": backend_name,
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
                        "matched_detection_id": row.matched_detection_id,
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
        per_frame_events.append(
            {
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
        )
        previous_states = current_states
        previous_snapshot = {
            row.track_id: {"track_id": row.track_id, "family": row.family, "class_name": row.class_name, "bbox_xyxy": list(row.bbox_xyxy)}
            for row in lost_rows
        }
        annotated_tracks = [
            {
                "track_id": row.track_id,
                "class_name": row.class_name,
                "bbox_xyxy": list(row.bbox_xyxy),
                "state": current_states.get(row.track_id, row.backend_state),
            }
            for row in tracked_rows + lost_rows
        ]
        annotated = annotate_frame(
            frame=frame,
            timestamp_seconds=timestamp_seconds,
            detector_ran=detector_ran,
            backend_label=backend_name,
            tracks=annotated_tracks,
        )
        writer.write(annotated)
        if new_ids_event or reactivated_ids_event or terminated_ids_event or len(annotated_tracks) >= 4:
            debug_path = output_dir / f"debug_{processed_frame_index:06d}.jpg"
            cv2.imwrite(str(debug_path), annotated)
        if len(frames_for_visuals) < 300:
            frames_for_visuals[processed_frame_index] = annotated.copy()
    writer.release()
    tracking_runtime_seconds = round(time.perf_counter() - started, 6)
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
    raw_track_rows = []
    for record in raw_tracks.values():
        start_timestamp = float(record.observations[0]["timestamp_seconds"]) if record.observations else record.created_timestamp_seconds
        end_timestamp = float(record.observations[-1]["timestamp_seconds"]) if record.observations else record.created_timestamp_seconds
        raw_track_rows.append(
            {
                "tracker_id": record.track_id,
                "class": record.class_name,
                "object_family": record.family,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "duration": round(max(0.0, end_timestamp - start_timestamp), 6),
                "detector_hit_count": len(record.detector_hit_timestamps),
                "state": record.last_state,
                "confirmed": record.confirmed,
                "lost_or_reactivated_state": "reactivated" if record.reactivated_count else ("lost" if record.last_state == RECOVERABLE else record.last_state),
                "termination_reason": record.termination_reason,
                "source_detector_frame_ids": sorted(
                    {
                        int(obs["processed_frame_index"])
                        for obs in record.observations
                        if bool(obs.get("detector_ran"))
                    }
                ),
                "tracker_backend": backend_name,
                "reid_enabled": getattr(backend, "verification", {}).get("requested_with_reid", False),
                "appearance_match_evidence": "not_available",
                "observations": record.observations,
            }
        )
    report = {
        "status": "success" if checks["passed"] and metrics["tracks_lost_due_to_skipped_detector_frame"] == 0 else "warning",
        "processed_frames": len(per_frame_events),
        "detector_frames": len([item for item in per_frame_events if item["detector_ran"]]),
        "skipped_detector_frames": len([item for item in per_frame_events if not item["detector_ran"]]),
        "tracker_runtime_seconds": tracking_runtime_seconds,
        "end_to_end_replay_runtime_seconds": tracking_runtime_seconds,
        "average_ms_per_processed_frame": round((tracking_runtime_seconds * 1000.0) / max(len(per_frame_events), 1), 6),
        "realtime_factor": round((video_info.duration_seconds / tracking_runtime_seconds), 6) if tracking_runtime_seconds > 0 else "not_available",
        "peak_gpu_memory_mb": getattr(backend, "verification", {}).get("peak_allocated_vram_mb", "not_available"),
        "peak_system_memory_mb": "not_available",
        "active_tracks_final": len([item for item in previous_states.values() if item in {ACTIVE, "tentative"}]),
        "lost_tracks_final": len([item for item in previous_states.values() if item == RECOVERABLE]),
        "removed_tracks_final": len([item for item in raw_tracks.values() if item.termination_reason == "removed_from_tracker"]),
        "interior_new_ids": new_ids_from_interior,
        "boundary_new_ids": new_ids_from_boundaries,
        "average_detector_hits_per_confirmed_track": round(
            sum(len(item.detector_hit_timestamps) for item in raw_tracks.values() if item.confirmed) / max(len([item for item in raw_tracks.values() if item.confirmed]), 1),
            6,
        ) if raw_tracks else "not_available",
        "mean_tracker_lifespan": round(
            sum(
                max(0.0, float(item["duration"]))
                for item in raw_track_rows
            ) / max(len(raw_track_rows), 1),
            6,
        ) if raw_track_rows else "not_available",
        "matched_detections": sum(len(item["matched_track_ids"]) for item in per_frame_events),
        "unmatched_detections": sum(max(0, int(item["detection_count"]) - len(item["matched_track_ids"]) - len(item["new_track_ids"])) for item in per_frame_events if item["detector_ran"]),
        "unmatched_active_tracks": sum(len(item["recoverable_track_ids_after"]) for item in per_frame_events if item["detector_ran"]),
        "association_failures": len([item for item in new_id_events if item["reason"] in {"reactivation_failed", "spatial_conflict"}]),
        "class_family_conflicts": len([item for item in new_id_events if item["reason"] == "class_family_conflict"]),
        "spatial_conflicts": len([item for item in new_id_events if item["reason"] == "spatial_conflict"]),
        "appearance_assisted_matches": getattr(backend, "verification", {}).get("accepted_appearance_matches", "not_available"),
        "appearance_rejections": getattr(backend, "verification", {}).get("rejected_appearance_matches", "not_available"),
        "ambiguous_appearance_matches": "not_available",
        **metrics,
        "validation_checks": checks,
        "same_detection_cache_checksum": cache_payload["detection_cache_checksum"]["sha256"],
    }
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
    lines = [
        f"# {backend_name} Tracking Report",
        "",
        f"- Processed frames: {report['processed_frames']}",
        f"- Detector frames: {report['detector_frames']}",
        f"- Raw tracker IDs: {report['raw_track_ids']}",
        f"- Confirmed tracks: {report['confirmed_tracks']}",
        f"- Tentative tracks: {report['tentative_tracks']}",
        f"- Reactivations: {report['successful_reactivations']}",
        f"- Tracker removals: {report['removed_tracks_final']}",
        f"- Tracks lost due to skipped frames: {report['tracks_lost_due_to_skipped_detector_frame']}",
    ]
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(output_dir / "tracking_report.md", lines)
    write_html_from_markdown(output_dir / "tracking_report.html", markdown_text)
    return {
        "report": report,
        "raw_tracks": raw_track_rows,
        "frame_tracking_events": per_frame_events,
        "visual_frames": frames_for_visuals,
    }
