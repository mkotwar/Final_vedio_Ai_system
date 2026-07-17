from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    if str(case_root) not in sys.path:
        sys.path.insert(0, str(case_root))
    from hybrid_tracking_test.box_validation import validate_propagated_bbox
    from hybrid_tracking_test.compare_tracking_results import compare_tracking_results
    from hybrid_tracking_test.config import build_arg_parser, resolve_config
    from hybrid_tracking_test.data_models import DetectionObservation
    from hybrid_tracking_test.detector_adapter import DetectorAdapter, build_detector_config, detector_runtime_metadata
    from hybrid_tracking_test.hybrid_track_manager import HybridTrackManager
    from hybrid_tracking_test.kcf_tracker_wrapper import kcf_api_name
    from hybrid_tracking_test.metrics import build_acceptance_assessment, build_main_report, build_timing_report, build_track_summary
    from hybrid_tracking_test.motion_trigger import MotionTrigger, detect_entry_zone_motion
    from hybrid_tracking_test.video_reader import iter_processed_frames, read_video_metadata
    from hybrid_tracking_test.visualization import build_annotated_video
else:
    from .box_validation import validate_propagated_bbox
    from .compare_tracking_results import compare_tracking_results
    from .config import build_arg_parser, resolve_config
    from .data_models import DetectionObservation
    from .detector_adapter import DetectorAdapter, build_detector_config, detector_runtime_metadata
    from .hybrid_track_manager import HybridTrackManager
    from .kcf_tracker_wrapper import kcf_api_name
    from .metrics import build_acceptance_assessment, build_main_report, build_timing_report, build_track_summary
    from .motion_trigger import MotionTrigger, detect_entry_zone_motion
    from .video_reader import iter_processed_frames, read_video_metadata
    from .visualization import build_annotated_video


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _failure(message: str, *, severity: str = "error", source_frame_index: int | None = None, timestamp_seconds: float | None = None) -> dict[str, Any]:
    return {
        "severity": severity,
        "message": message,
        "source_frame_index": source_frame_index,
        "timestamp_seconds": timestamp_seconds,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    config = resolve_config(args)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        if kcf_api_name() == "unavailable":
            raise RuntimeError(
                "OpenCV KCF tracker is unavailable. Install a compatible opencv-contrib-python package."
            )
        video_metadata = read_video_metadata(config.video_path)
        detector_config = build_detector_config(
            minimum_detection_confidence=config.minimum_detection_confidence,
            class_confidence_thresholds=config.class_confidence_thresholds,
            device_override=config.device,
        )
        detector = DetectorAdapter(detector_config)
        track_manager = HybridTrackManager(config)
        motion_trigger = MotionTrigger(
            motion_min_area_ratio=config.motion_min_area_ratio,
            motion_persistence_frames=config.motion_persistence_frames,
            motion_track_region_expansion=config.motion_track_region_expansion,
        )
        frame_metrics: list[dict[str, Any]] = []
        frame_payloads_by_source_index: dict[int, dict[str, Any]] = {}
        timing_lists: dict[str, list[float]] = {
            "decode_time_ms": [],
            "preprocess_time_ms": [],
            "yolo_inference_ms": [],
            "association_time_ms": [],
            "kcf_initialization_time_ms": [],
            "kcf_update_ms": [],
            "kcf_update_ms_per_track": [],
            "motion_trigger_time_ms": [],
            "visualization_time_ms": [],
            "json_writing_time_ms": [],
        }
        last_yolo_timestamp: float | None = None
        last_overlap_trigger_frame = -999999
        cumulative_yolo_calls = 0
        cumulative_kcf_updates = 0
        last_empty_scene_yolo_timestamp: float | None = None

        for frame_item in iter_processed_frames(video_path=config.video_path, processing_fps=config.processing_fps):
            frame = frame_item["frame"]
            source_frame_index = int(frame_item["source_frame_index"])
            processed_frame_index = int(frame_item["processed_frame_index"])
            timestamp_seconds = float(frame_item["source_timestamp_seconds"])
            motion_started = time.perf_counter()
            motion_payload = motion_trigger.evaluate(
                frame=frame,
                active_boxes=track_manager.active_track_boxes(),
                roi_masks=config.entry_zones,
            ) if config.enable_motion_trigger else {
                "enabled": False,
                "triggered": False,
                "motion_area_pixels": 0,
                "motion_area_ratio": 0.0,
                "uncovered_region_count": 0,
                "largest_uncovered_region": 0,
                "persistence_count": 0,
                "uncovered_regions": [],
            }
            timing_lists["motion_trigger_time_ms"].append((time.perf_counter() - motion_started) * 1000.0)

            entry_triggered = False
            entry_zone_names: list[str] = []
            if config.enable_entry_zone_trigger and motion_payload["uncovered_regions"]:
                entry_triggered, entry_zone_names = detect_entry_zone_motion(
                    motion_regions=list(motion_payload["uncovered_regions"]),
                    entry_zones=config.entry_zones,
                    frame_width=video_metadata.width,
                    frame_height=video_metadata.height,
                )

            update_started = time.perf_counter()
            kcf_track_payloads, kcf_refresh_reasons = track_manager.update_kcf_tracks(
                frame=frame,
                source_frame_index=source_frame_index,
                processed_frame_index=processed_frame_index,
                timestamp_seconds=timestamp_seconds,
                validate_fn=lambda current_bbox_xyxy, previous_bbox_xyxy: validate_propagated_bbox(
                    current_bbox_xyxy=current_bbox_xyxy,
                    previous_bbox_xyxy=previous_bbox_xyxy,
                    frame_width=video_metadata.width,
                    frame_height=video_metadata.height,
                    minimum_area_ratio_change=config.minimum_area_ratio_change,
                    maximum_area_ratio_change=config.maximum_area_ratio_change,
                    minimum_aspect_ratio_change=config.minimum_aspect_ratio_change,
                    maximum_aspect_ratio_change=config.maximum_aspect_ratio_change,
                    maximum_center_jump_diagonals=config.maximum_center_jump_diagonals,
                    minimum_visible_area_ratio=config.minimum_visible_area_ratio,
                ),
            )
            kcf_elapsed_ms = (time.perf_counter() - update_started) * 1000.0
            timing_lists["kcf_update_ms"].append(kcf_elapsed_ms)
            active_tracks = track_manager.active_track_list()
            if active_tracks:
                timing_lists["kcf_update_ms_per_track"].append(kcf_elapsed_ms / max(len(active_tracks), 1))
            cumulative_kcf_updates += len(active_tracks)
            track_manager.counters["kcf_update_count"] += len(active_tracks)

            yolo_trigger_reasons: list[str] = []
            no_active_tracks = not active_tracks
            scheduled_refresh_due = bool(config.enable_scheduled_refresh and processed_frame_index % config.yolo_interval_frames == 0)
            maximum_gap_reached = bool(last_yolo_timestamp is None or (timestamp_seconds - last_yolo_timestamp) >= config.max_yolo_gap_seconds)
            meaningful_motion = bool(config.enable_motion_trigger and motion_payload["triggered"])
            empty_scene_heartbeat_due = bool(
                no_active_tracks
                and (
                    last_empty_scene_yolo_timestamp is None
                    or (timestamp_seconds - last_empty_scene_yolo_timestamp) >= config.empty_scene_yolo_interval_seconds
                )
            )
            if scheduled_refresh_due:
                yolo_trigger_reasons.append("scheduled_refresh")
            if maximum_gap_reached and not no_active_tracks:
                yolo_trigger_reasons.append("maximum_yolo_gap_reached")
            yolo_trigger_reasons.extend(kcf_refresh_reasons)
            if meaningful_motion:
                yolo_trigger_reasons.append("meaningful_uncovered_motion_detected")
            if entry_triggered:
                yolo_trigger_reasons.append("entry_zone_motion_detected")
            if empty_scene_heartbeat_due:
                yolo_trigger_reasons.append("empty_scene_heartbeat")
            overlap_triggered = False
            if config.enable_overlap_trigger and (processed_frame_index - last_overlap_trigger_frame) >= config.overlap_trigger_cooldown_frames:
                for left_index in range(len(active_tracks)):
                    for right_index in range(left_index + 1, len(active_tracks)):
                        left = active_tracks[left_index]
                        right = active_tracks[right_index]
                        iou_value = 0.0
                        try:
                            from hybrid_tracking_test.box_validation import bbox_iou  # type: ignore
                        except Exception:
                            from .box_validation import bbox_iou  # type: ignore
                        iou_value = bbox_iou(left.bbox_xyxy, right.bbox_xyxy)
                        if iou_value >= config.maximum_pairwise_overlap_iou and left.bbox_source == "kcf" and right.bbox_source == "kcf":
                            overlap_triggered = True
                            break
                    if overlap_triggered:
                        break
                if overlap_triggered:
                    yolo_trigger_reasons.append("severe_track_overlap_detected")
                    last_overlap_trigger_frame = processed_frame_index

            yolo_executed = bool(yolo_trigger_reasons)
            detections: list[dict[str, Any]] = []
            associations: list[dict[str, Any]] = []
            yolo_inference_ms = 0.0
            association_ms = 0.0

            if yolo_executed:
                if "scheduled_refresh" in yolo_trigger_reasons:
                    track_manager.counters["scheduled_yolo_call_count"] += 1
                    track_manager._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="scheduled_yolo_refresh",
                        details={"reasons": list(yolo_trigger_reasons)},
                    )
                else:
                    track_manager.counters["emergency_yolo_call_count"] += 1
                    track_manager._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="emergency_yolo_refresh",
                        details={"reasons": list(yolo_trigger_reasons)},
                    )
                if "meaningful_uncovered_motion_detected" in yolo_trigger_reasons:
                    track_manager.counters["motion_triggered_yolo_call_count"] += 1
                if "empty_scene_heartbeat" in yolo_trigger_reasons:
                    track_manager.counters["empty_scene_yolo_call_count"] += 1
                    last_empty_scene_yolo_timestamp = timestamp_seconds
                if any(reason in yolo_trigger_reasons for reason in {"kcf_tracker_failed", "stale_track_requires_refresh"}):
                    track_manager.counters["kcf_failure_yolo_call_count"] += 1
                if "severe_track_overlap_detected" in yolo_trigger_reasons:
                    track_manager.counters["overlap_yolo_call_count"] += 1
                if "invalid_kcf_box_detected" in yolo_trigger_reasons:
                    track_manager.counters["box_validation_yolo_call_count"] += 1
                if meaningful_motion:
                    motion_event_details = {
                        key: value
                        for key, value in motion_payload.items()
                        if key != "mask"
                    }
                    track_manager._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="uncovered_motion_trigger",
                        details=motion_event_details,
                    )
                if entry_triggered:
                    track_manager._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="entry_zone_trigger",
                        details={"zones": entry_zone_names},
                    )
                if overlap_triggered:
                    track_manager._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="overlap_trigger",
                        details={},
                    )
                yolo_started = time.perf_counter()
                raw_detections = detector.detect(frame)
                yolo_inference_ms = (time.perf_counter() - yolo_started) * 1000.0
                timing_lists["yolo_inference_ms"].append(yolo_inference_ms)
                cumulative_yolo_calls += 1
                track_manager.counters["yolo_call_count"] += 1
                last_yolo_timestamp = timestamp_seconds
                detection_models = [
                    DetectionObservation(
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        confidence=float(item["confidence"]),
                        bbox_xyxy=[float(value) for value in item["bbox_xyxy"]],
                        model_source=str(item["model_source"]),
                        detection_index=index,
                        source_frame_index=source_frame_index,
                        processed_frame_index=processed_frame_index,
                        timestamp_seconds=timestamp_seconds,
                    )
                    for index, item in enumerate(raw_detections)
                ]
                association_started = time.perf_counter()
                track_payloads, detections, associations = track_manager.refresh_with_detections(
                    detections=detection_models,
                    frame=frame,
                    source_frame_index=source_frame_index,
                    processed_frame_index=processed_frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
                association_ms = (time.perf_counter() - association_started) * 1000.0
                timing_lists["association_time_ms"].append(association_ms)
                frame_tracks_payload = [
                    {
                        "track_id": track.track_id,
                        "class_id": track.class_id,
                        "class_name": track.class_name,
                        "object_family": track.object_family,
                        "bbox_xyxy": [round(float(value), 3) for value in track.bbox_xyxy],
                        "bbox_source": track.bbox_source,
                        "status": track.status,
                        "kcf_success": track.kcf_success,
                        "frames_since_detection": track.frames_since_detection(processed_frame_index),
                        "seconds_since_detection": round(track.seconds_since_detection(timestamp_seconds), 6),
                        "last_detection_confidence": None if track.last_detection_confidence is None else round(float(track.last_detection_confidence), 6),
                        "reactivation_count": int(track.reactivation_count),
                        "validation": {"valid": True, "reasons": [], "metrics": {}},
                    }
                    for track in track_manager.active_track_list()
                ]
            else:
                frame_tracks_payload = kcf_track_payloads

            frame_payload = {
                "source_frame_index": source_frame_index,
                "processed_frame_index": processed_frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "time_delta_seconds": round(float(frame_item["time_delta_from_previous_processed_frame"]), 6),
                "processing_fps": float(config.processing_fps),
                "yolo_executed": yolo_executed,
                "yolo_trigger_reasons": list(dict.fromkeys(yolo_trigger_reasons)),
                "yolo_inference_ms": round(yolo_inference_ms, 6),
                "kcf_update_ms": round(kcf_elapsed_ms, 6),
                "active_track_count": len(track_manager.active_track_list()),
                "motion_trigger": {
                    "enabled": bool(motion_payload.get("enabled", False)),
                    "triggered": bool(motion_payload.get("triggered", False)),
                    "motion_area_ratio": motion_payload.get("motion_area_ratio", 0.0),
                    "uncovered_region_count": motion_payload.get("uncovered_region_count", 0),
                    "motion_area_pixels": motion_payload.get("motion_area_pixels", 0),
                    "largest_uncovered_region": motion_payload.get("largest_uncovered_region", 0),
                    "persistence_count": motion_payload.get("persistence_count", 0),
                },
                "detections": detections,
                "associations": associations,
                "tracks": frame_tracks_payload,
                "cumulative_yolo_calls": cumulative_yolo_calls,
                "cumulative_kcf_updates": cumulative_kcf_updates,
            }
            frame_metrics.append(frame_payload)
            frame_payloads_by_source_index[source_frame_index] = frame_payload

        final_timestamp = frame_metrics[-1]["timestamp_seconds"] if frame_metrics else 0.0
        final_source_frame = frame_metrics[-1]["source_frame_index"] if frame_metrics else 0
        track_manager.flush_at_video_end(timestamp_seconds=final_timestamp, source_frame_index=final_source_frame)

        video_info_payload = {
            "input_video_path": str(video_metadata.video_path),
            "video_name": video_metadata.video_path.name,
            "fps": video_metadata.fps,
            "frame_count": video_metadata.frame_count,
            "duration_seconds": video_metadata.duration_seconds,
            "width": video_metadata.width,
            "height": video_metadata.height,
        }
        timing_report = build_timing_report(
            video_duration_seconds=video_metadata.duration_seconds,
            total_runtime_seconds=time.perf_counter() - started,
            processed_frame_count=len(frame_metrics),
            yolo_call_count=cumulative_yolo_calls,
            kcf_update_count=cumulative_kcf_updates,
            timing_lists=timing_lists,
        )
        track_summary = build_track_summary(track_manager.all_tracks())
        main_report = build_main_report(
            config={
                **config.to_dict(),
                "opencv_version": cv2.__version__,
                "kcf_api_used": kcf_api_name(),
                "detector_runtime": detector_runtime_metadata(detector_config),
            },
            video_metadata=video_info_payload,
            frame_metrics=frame_metrics,
            track_manager=track_manager,
            event_records=[item.to_dict() for item in track_manager.events],
            failures=failures,
            timing_report=timing_report,
        )

        write_started = time.perf_counter()
        _write_json(output_dir / "04c_hybrid_tracks.json", {"status": "success", "track_summaries": track_summary["tracks"]})
        _write_json(output_dir / "04c_hybrid_tracking_report.json", main_report)
        _write_json(output_dir / "04c_hybrid_tracking_timing.json", timing_report)
        _write_json(output_dir / "04c_hybrid_tracking_events.json", {"status": "success", "events": [item.to_dict() for item in track_manager.events]})
        _write_json(output_dir / "04c_hybrid_track_summary.json", track_summary)
        _write_json(output_dir / "04c_hybrid_config.json", config.to_dict())
        _write_json(output_dir / "04c_hybrid_failures.json", {"status": "success" if not failures else "warnings", "failures": failures})
        _write_json(output_dir / "04c_hybrid_frame_metrics.json", {"status": "success", "frames": frame_metrics})
        timing_lists["json_writing_time_ms"].append((time.perf_counter() - write_started) * 1000.0)

        if config.save_annotated_video:
            viz_started = time.perf_counter()
            build_annotated_video(
                video_path=config.video_path,
                frame_payloads_by_source_index=frame_payloads_by_source_index,
                output_path=output_dir / "04c_hybrid_annotated_video.mp4",
                source_fps=video_metadata.fps,
            )
            timing_lists["visualization_time_ms"].append((time.perf_counter() - viz_started) * 1000.0)

        comparison_payload, comparison_markdown = compare_tracking_results(
            run_dir=config.run_dir,
            hybrid_tracks_path=output_dir / "04c_hybrid_tracks.json",
        )
        comparison_payload["acceptance_assessment"] = build_acceptance_assessment(
            report_payload=main_report,
            comparison_payload=comparison_payload,
        )
        _write_json(output_dir / "04c_hybrid_comparison_report.json", comparison_payload)
        (output_dir / "04c_hybrid_comparison_report.md").write_text(comparison_markdown, encoding="utf-8")

        print("Hybrid tracking experiment completed")
        print(f"Video: {config.video_path.name}")
        print(f"Duration: {video_metadata.duration_seconds:.3f}s")
        print(f"Source FPS: {video_metadata.fps:.3f}")
        print(f"Processing FPS: {config.processing_fps:.3f}")
        print(f"Processed frames: {len(frame_metrics)}")
        print(f"YOLO calls: {cumulative_yolo_calls}")
        print(f"KCF updates: {cumulative_kcf_updates}")
        print(f"YOLO reduction: {main_report['yolo_reduction_percent']:.3f}%")
        print(f"Tracks created: {track_manager.counters.get('tracks_created', 0)}")
        print(f"Tracks confirmed: {track_manager.counters.get('tracks_confirmed', 0)}")
        print(f"KCF failures: {track_manager.counters.get('kcf_failure_count', 0)}")
        print(f"Emergency YOLO calls: {track_manager.counters.get('emergency_yolo_call_count', 0)}")
        print(f"Runtime: {timing_report['total_runtime_seconds']:.3f}s")
        print(f"Realtime factor: {timing_report['realtime_factor']:.3f}")
        print(f"Outputs: {output_dir}")
    except Exception as exc:
        failures.append(_failure(str(exc), severity="error"))
        _write_json(output_dir / "04c_hybrid_failures.json", {"status": "failed", "failures": failures})
        raise


if __name__ == "__main__":
    main()
