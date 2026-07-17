from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (case_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from continuous_mot_hybrid.adaptive_detector_scheduler import AdaptiveDetectorScheduler, SchedulerObservation
    from continuous_mot_hybrid.botsort_backend import BotSortBackend
    from continuous_mot_hybrid.bytetrack_backend import ByteTrackBackend
    from continuous_mot_hybrid.config import build_arg_parser, resolve_config
    from continuous_mot_hybrid.detection_track_adapter import detections_from_rows, observation_from_backend_track
    from continuous_mot_hybrid.metrics import build_detector_report, build_runtime_report
    from continuous_mot_hybrid.motion_estimator import estimate_scene_motion
    from continuous_mot_hybrid.report_writer import write_json, write_markdown
    from continuous_mot_hybrid.short_gap_visual_tracker import ShortGapVisualTrackerManager
    from continuous_mot_hybrid.track_lifecycle import LifecycleConfig, summarize_track_histories
    from continuous_mot_hybrid.track_state import object_family_for_class
    from continuous_mot_hybrid.track_timeline import build_tracking_timeline_report
    from continuous_mot_hybrid.video_frame_stream import read_video_info, stream_processed_frames
    from continuous_mot_hybrid.yolo_detector import YoloDetector, resolve_model_specs
else:
    from .adaptive_detector_scheduler import AdaptiveDetectorScheduler, SchedulerObservation
    from .botsort_backend import BotSortBackend
    from .bytetrack_backend import ByteTrackBackend
    from .config import build_arg_parser, resolve_config
    from .detection_track_adapter import detections_from_rows, observation_from_backend_track
    from .metrics import build_detector_report, build_runtime_report
    from .motion_estimator import estimate_scene_motion
    from .report_writer import write_json, write_markdown
    from .short_gap_visual_tracker import ShortGapVisualTrackerManager
    from .track_lifecycle import LifecycleConfig, summarize_track_histories
    from .track_state import object_family_for_class
    from .track_timeline import build_tracking_timeline_report
    from .video_frame_stream import read_video_info, stream_processed_frames
    from .yolo_detector import YoloDetector, resolve_model_specs


def _dedupe_backend_tracks(rows):
    deduped = {}
    priority = {"tracked": 0, "lost": 1}
    for row in rows:
        existing = deduped.get(row.track_id)
        if existing is None or priority.get(row.backend_state, 9) < priority.get(existing.backend_state, 9):
            deduped[row.track_id] = row
    return list(deduped.values())


def _write_initial_outputs(config, video_info, frame_records):
    write_json(
        config.output_dirs["video"] / "video_info.json",
        {
            "status": "success",
            "video_path": str(config.video_path),
            "source_fps": round(video_info.source_fps, 6),
            "source_frame_count": video_info.source_frame_count,
            "duration_seconds": round(video_info.duration_seconds, 6),
            "width": video_info.width,
            "height": video_info.height,
        },
    )
    write_json(config.output_dirs["video"] / "resolved_config.json", config.to_dict())
    write_json(config.output_dirs["frames"] / "frame_schedule.json", {"status": "success", "frames": frame_records})
    write_json(
        config.output_dirs["frames"] / "frame_stream_metrics.json",
        {
            "status": "success",
            "processed_frame_count": len(frame_records),
            "processing_fps": config.processing_fps,
            "first_timestamp_seconds": frame_records[0]["timestamp_seconds"] if frame_records else 0.0,
            "last_timestamp_seconds": frame_records[-1]["timestamp_seconds"] if frame_records else 0.0,
        },
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = resolve_config(args)
    video_info = read_video_info(config.video_path)
    frame_debug_dir = config.output_dirs["frames"] / "debug_frames" if config.save_debug_frames else None
    _, _, _, frame_iterator = stream_processed_frames(
        video_path=config.video_path,
        processing_fps=config.processing_fps,
        debug_frames_dir=frame_debug_dir,
    )
    frame_records: list[dict[str, Any]] = []
    loaded_frames: list[Any] = []
    for frame_record, frame in frame_iterator:
        frame_records.append(frame_record.to_dict())
        loaded_frames.append(frame)
    _write_initial_outputs(config, video_info, frame_records)

    model_specs = resolve_model_specs(
        person_model_path=config.person_model_path,
        object_model_path=config.object_model_path,
        combined_model_path=config.combined_model_path,
    )
    detector = YoloDetector(
        model_specs=model_specs,
        confidence=config.yolo_confidence,
        iou=config.yolo_iou,
        device=config.device,
    )
    backend = (
        ByteTrackBackend(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            match_thresh=config.match_thresh,
            track_buffer_frames=max(1, int(round(config.track_buffer_seconds * config.processing_fps))),
        )
        if config.mot_backend == "bytetrack"
        else BotSortBackend(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            match_thresh=config.match_thresh,
            track_buffer_frames=max(1, int(round(config.track_buffer_seconds * config.processing_fps))),
        )
    )
    scheduler = AdaptiveDetectorScheduler(
        normal_interval_seconds=config.detector_normal_interval_seconds,
        sparse_interval_seconds=config.detector_sparse_interval_seconds,
        idle_interval_seconds=config.detector_idle_interval_seconds,
        maximum_gap_seconds=config.detector_max_gap_seconds,
    )
    visual_manager = ShortGapVisualTrackerManager(
        tracker_name=config.short_gap_visual_tracker,
        maximum_bridge_seconds=config.visual_bridge_max_seconds,
        frame_width=video_info.width,
        frame_height=video_info.height,
    )
    lifecycle_config = LifecycleConfig(
        min_person_confirm_hits=config.min_person_confirm_hits,
        min_vehicle_confirm_hits=config.min_vehicle_confirm_hits,
        lost_recovery_seconds=config.lost_recovery_seconds,
    )

    detection_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    detector_call_metrics: list[dict[str, Any]] = []
    previous_frame = None
    previous_detection_count = 0
    previous_active_track_count = 0
    recent_losses = 0
    last_detector_call_timestamp: float | None = None
    tracking_started = time.perf_counter()

    for frame_record, frame in zip(frame_records, loaded_frames):
        motion = estimate_scene_motion(previous_frame, frame)
        observation = SchedulerObservation(
            timestamp_seconds=float(frame_record["timestamp_seconds"]),
            active_tracks=previous_active_track_count,
            low_confidence_tracks=0,
            recent_track_losses=recent_losses,
            recent_unmatched_detections=max(0, previous_detection_count - previous_active_track_count),
            average_assignment_cost=0.0 if previous_active_track_count > 0 else 0.5,
            scene_motion_change=float(motion.motion_score),
            overlap_count=max(0, previous_active_track_count - 1),
            entry_zone_activity=previous_active_track_count == 0 and motion.changed_pixels_ratio > 0.015,
            recent_visual_tracker_failure=False,
            last_detector_call_timestamp=last_detector_call_timestamp,
        )
        decision = scheduler.decide(observation)
        current_detection_rows: list[dict[str, Any]] = []
        detector_reason = "no_call"
        if decision.should_run_detector:
            detector_reason = "scheduled"
            if decision.state == "emergency":
                detector_reason = "emergency"
            current_detection_rows = detector.detect(
                frame=frame,
                frame_record=frame_record,
                scheduler_state=decision.state,
                detector_reason=detector_reason,
            )
            detection_rows.extend(current_detection_rows)
            scheduler.record_detector_call(frame_record=frame_record, decision=decision, reason=detector_reason)
            detector_call_metrics.append(
                {
                    "processed_frame_index": int(frame_record["processed_frame_index"]),
                    "timestamp_seconds": round(float(frame_record["timestamp_seconds"]), 6),
                    "scheduler_state": decision.state,
                    "detector_call_reason": detector_reason,
                    "detection_count": len(current_detection_rows),
                }
            )
            last_detector_call_timestamp = float(frame_record["timestamp_seconds"])
        detections = detections_from_rows(current_detection_rows)
        backend_tracks = _dedupe_backend_tracks(backend.update(detections=detections))
        previous_active_track_count = len([item for item in backend_tracks if item.backend_state == "tracked"])
        recent_losses = len([item for item in backend_tracks if item.backend_state == "lost"])
        previous_detection_count = len(current_detection_rows)

        detector_track_ids = {item.track_id: item for item in backend_tracks if item.matched_detection_id}
        if current_detection_rows and config.enable_short_gap_visual_tracker:
            for track_id, backend_track in detector_track_ids.items():
                visual_manager.start_or_refresh(
                    track_id=track_id,
                    frame=frame,
                    bbox_xyxy=list(backend_track.bbox_xyxy),
                    timestamp_seconds=float(frame_record["timestamp_seconds"]),
                )
                visual_manager.reconcile_with_detector(
                    track_id=track_id,
                    detector_bbox_xyxy=list(backend_track.bbox_xyxy),
                    timestamp_seconds=float(frame_record["timestamp_seconds"]),
                )

        for backend_track in backend_tracks:
            bbox_source = "yolo" if backend_track.matched_detection_id else "mot_predicted"
            observation_validity = "valid"
            if not current_detection_rows and config.enable_short_gap_visual_tracker and backend_track.confirmed:
                bridge_event = visual_manager.update(
                    track_id=backend_track.track_id,
                    frame=frame,
                    timestamp_seconds=float(frame_record["timestamp_seconds"]),
                )
                if bridge_event is not None:
                    bbox_source = str(bridge_event["bbox_source"])
                    observation_validity = "valid" if bbox_source == "visual_bridge_supported" else "invalid"
                    if "bbox_xyxy" in bridge_event:
                        backend_track = type(backend_track)(
                            track_id=backend_track.track_id,
                            family=backend_track.family,
                            class_name=backend_track.class_name,
                            bbox_xyxy=[float(value) for value in bridge_event["bbox_xyxy"]],
                            confirmed=backend_track.confirmed,
                            age_frames=backend_track.age_frames,
                            hits=backend_track.hits,
                            time_since_update_frames=backend_track.time_since_update_frames,
                            backend_state=backend_track.backend_state,
                            matched_detection_id=backend_track.matched_detection_id,
                            matched_detection_confidence=backend_track.matched_detection_confidence,
                            association_cost=backend_track.association_cost,
                        )
            observation_row = observation_from_backend_track(
                backend_track=backend_track,
                frame_record=frame_record,
                delta_seconds=float(frame_record["frame_time_delta"]) if frame_record["frame_time_delta"] else (1.0 / config.processing_fps),
                bbox_source=bbox_source,
                lifecycle_state="confirmed" if backend_track.confirmed else "tentative",
                observation_validity=observation_validity,
            ).to_dict()
            observation_rows.append(observation_row)
        previous_frame = frame

    tracking_runtime_seconds = time.perf_counter() - tracking_started
    track_rows = summarize_track_histories(observation_rows, config=lifecycle_config)
    tracking_report = build_tracking_timeline_report(track_rows)

    write_json(config.output_dirs["detections"] / "detector_schedule.json", {"status": "success", "calls": scheduler.call_history})
    write_json(config.output_dirs["detections"] / "detector_schedule_report.json", scheduler.build_report())
    write_json(config.output_dirs["detections"] / "yolo_detections.json", {"status": "success", "detections": detection_rows})
    write_json(config.output_dirs["detections"] / "yolo_call_metrics.json", {"status": "success", "calls": detector_call_metrics})
    write_json(config.output_dirs["tracking"] / "track_observations.json", {"status": "success", "observations": observation_rows})
    write_json(config.output_dirs["tracking"] / "raw_tracks.json", {"status": "success", "tracks": track_rows})
    write_json(config.output_dirs["tracking"] / "tracking_report.json", tracking_report)
    write_json(config.output_dirs["tracking"] / "visual_bridge_events.json", {"status": "success", "events": visual_manager.events})
    write_json(config.output_dirs["tracking"] / "visual_bridge_failures.json", {"status": "success", "events": visual_manager.failures})

    runtime_report = build_runtime_report(
        video_duration_seconds=video_info.duration_seconds,
        processed_frames=len(frame_records),
        tracking_runtime_seconds=tracking_runtime_seconds,
        cleanup_runtime_seconds=0.0,
        crop_runtime_seconds=0.0,
    )
    detector_report = build_detector_report(
        processed_frames=len(frame_records),
        detector_calls=scheduler.call_history,
        source_duration_seconds=video_info.duration_seconds,
    )
    write_json(config.output_dirs["reports"] / "runtime_report.json", runtime_report)
    write_json(config.output_dirs["reports"] / "detector_report.json", detector_report)
    write_json(config.output_dirs["reports"] / "tracking_report.json", tracking_report)
    write_markdown(
        config.output_dirs["reports"] / "final_report.md",
        [
            "# Continuous MOT Hybrid Tracking Stage",
            "",
            f"- Run directory: {config.run_dir}",
            f"- Processed frames: {len(frame_records)}",
            f"- Detector calls: {len(scheduler.call_history)}",
            f"- Raw tracks: {tracking_report['raw_track_ids']}",
            f"- Confirmed tracks: {tracking_report['confirmed_tracks']}",
        ],
    )
    write_json(
        config.output_dirs["reports"] / "final_report.json",
        {
            "status": "success",
            "run_dir": str(config.run_dir),
            "mot_backend": config.mot_backend,
            "processed_frames": len(frame_records),
            "detector_calls": len(scheduler.call_history),
            "raw_tracks": tracking_report["raw_track_ids"],
            "confirmed_tracks": tracking_report["confirmed_tracks"],
        },
    )
    print(f"run_dir={config.run_dir}")


if __name__ == "__main__":
    main()
