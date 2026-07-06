from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _ensure_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_ensure_repo_root()) not in sys.path:
    sys.path.insert(0, str(_ensure_repo_root()))

from run_tender_demo_pipeline import (
    DEFAULT_CLIP_OVERLAP_SECONDS,
    DEFAULT_CONTEXT_AFTER_SECONDS,
    DEFAULT_CONTEXT_BEFORE_SECONDS,
    DEFAULT_MAX_CLIP_SECONDS,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MIN_EXPANDED_CLIP_SECONDS,
    _create_candidate_clips,
    _create_debug_run_dir,
    _expand_candidate_clips,
    _extract_video_info,
    _read_motion_threshold,
    _sample_base_frames,
    _score_motion_on_sampled_frames,
    _select_motion_candidates,
    _write_video_info,
)
from step_11_yolo_object_scoring import run_yolo_object_scoring
from step_11b_object_motion_state import estimate_object_motion_states
from step_11c_plate_ocr_color_enrichment import run_plate_ocr_color_enrichment
from step_13_rank_candidate_clips import rank_candidate_clips
from step_14_select_topk_clips import select_topk_clips_for_qwen
from step_15_create_topk_vlm_inputs import create_topk_vlm_inputs
from step_16_run_topk_qwen import run_qwen_on_topk_vlm_inputs
from step_17_topk_final_summary import create_topk_final_summary
from step_18_export_event_clips import export_event_clips
from step_19_create_demo_report import create_demo_report_html

from tests.object_search_case.run_object_search_case import (
    DEFAULT_LOCAL_PERSON_MODEL,
    DEFAULT_LOCAL_VEHICLE_MODEL,
    PERSON_CLASS_FILTER_IDS,
    PERSON_CLASS_NAME,
    VEHICLE_CLASS_NAMES,
    BenchmarkBoTSORTTracker,
    _build_track_summaries,
    _detect_frames,
    _merge_frame_detections,
    _write_json as _write_object_json,
)


def _set_default_env(name: str, value: str) -> None:
    if name not in os.environ:
        os.environ[name] = value


def _sampled_manifest_to_object_tuples(
    sampled_frames: list[dict[str, object]],
    video_name: str,
) -> list[tuple[str, str, float, Path]]:
    repo_root = _ensure_repo_root()
    tuples: list[tuple[str, str, float, Path]] = []
    for item in sampled_frames:
        frame_path = repo_root / str(item["frame_path"])
        tuples.append(
            (
                str(item["sample_id"]),
                video_name,
                float(item["timestamp_seconds"]),
                frame_path,
            )
        )
    return tuples


def _run_tracking_and_search_index(
    *,
    run_dir: Path,
    video_name: str,
    sampled_frames: list[dict[str, object]],
    person_model: Path,
    vehicle_model: Path,
    person_conf: float,
    vehicle_conf: float,
    person_imgsz: int,
    vehicle_imgsz: int,
) -> dict[str, Any]:
    sampled_tuples = _sampled_manifest_to_object_tuples(sampled_frames, video_name)
    person_detections = _detect_frames(
        sampled_tuples,
        yolo_model_name=str(person_model),
        yolo_conf=person_conf,
        yolo_imgsz=person_imgsz,
        class_filter_ids=PERSON_CLASS_FILTER_IDS,
        allowed_class_names={PERSON_CLASS_NAME},
    )
    vehicle_detections = _detect_frames(
        sampled_tuples,
        yolo_model_name=str(vehicle_model),
        yolo_conf=vehicle_conf,
        yolo_imgsz=vehicle_imgsz,
        allowed_class_names=VEHICLE_CLASS_NAMES,
    )
    merged = _merge_frame_detections([person_detections, vehicle_detections])

    tracking_dir = run_dir / "10b_tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    benchmark_inputs = [
        SimpleNamespace(
            frame_id=item.frame_id,
            video_id=item.video_id,
            timestamp_seconds=item.timestamp_seconds,
            frame_width=item.frame_width,
            frame_height=item.frame_height,
            detections=[
                SimpleNamespace(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
                for detection in item.detections
            ],
        )
        for item in merged
    ]
    BenchmarkBoTSORTTracker.track_frames(
        benchmark_inputs,
        extracted_tuples=sampled_tuples,
        debug_output_dir=tracking_dir,
    )
    search_index, tracking_frames = _build_track_summaries(
        frame_detections=merged,
        tracking_results_path=tracking_dir / "tracking_results.json",
        sampled_frames=sampled_tuples,
        output_dir=run_dir / "10c_object_search_index_assets",
    )
    _write_object_json(run_dir / "10b_tracking_results.json", {"frames": tracking_frames})
    _write_object_json(run_dir / "10c_object_search_index.json", {"items": search_index})
    return {
        "tracking_dir": str(tracking_dir),
        "tracked_object_count": len(search_index),
        "tracking_frame_count": len(tracking_frames),
        "search_index_path": str(run_dir / "10c_object_search_index.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tender demo YOLO-first testcase: sampling -> YOLO/tracking/index -> event candidates -> Top-K -> VLM"
    )
    parser.add_argument("--video", required=True, help="Absolute or relative path to input video")
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    parser.add_argument("--person-model", default=str(DEFAULT_LOCAL_PERSON_MODEL))
    parser.add_argument("--vehicle-model", default=str(DEFAULT_LOCAL_VEHICLE_MODEL))
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--vehicle-conf", type=float, default=0.25)
    parser.add_argument("--person-imgsz", type=int, default=640)
    parser.add_argument("--vehicle-imgsz", type=int, default=640)
    parser.add_argument("--top-k-clips", type=int, default=8)
    parser.add_argument("--top-k-max", type=int, default=25)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    repo_root = _ensure_repo_root()
    video_path = Path(args.video).expanduser()
    if not video_path.is_absolute():
        video_path = (repo_root / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    person_model = Path(args.person_model).expanduser()
    vehicle_model = Path(args.vehicle_model).expanduser()
    if not person_model.is_absolute():
        person_model = (repo_root / person_model).resolve()
    if not vehicle_model.is_absolute():
        vehicle_model = (repo_root / vehicle_model).resolve()
    if not person_model.exists():
        raise FileNotFoundError(f"Person model not found: {person_model}")
    if not vehicle_model.exists():
        raise FileNotFoundError(f"Vehicle model not found: {vehicle_model}")

    _set_default_env("TENDER_DEMO_INPUT_VIDEO", str(video_path))
    _set_default_env("TENDER_DEMO_SAMPLE_EVERY_SECONDS", str(max(0.1, float(args.sample_every_seconds))))
    _set_default_env("TENDER_DEMO_YOLO_INPUT_SCOPE", "sampled_frames")
    _set_default_env("TENDER_DEMO_TOP_K_CLIPS", str(max(1, int(args.top_k_clips))))
    _set_default_env("TENDER_DEMO_TOP_K_MAX_CLIPS", str(max(1, int(args.top_k_max))))
    _set_default_env("TENDER_DEMO_QWEN_MAX_NEW_TOKENS", str(max(64, int(args.qwen_max_new_tokens))))

    print("[tender-demo-yolo-first] Starting isolated YOLO-first testcase")
    print(f"[tender-demo-yolo-first] Video: {video_path}")
    print(f"[tender-demo-yolo-first] Person model: {person_model}")
    print(f"[tender-demo-yolo-first] Vehicle model: {vehicle_model}")

    run_dir = _create_debug_run_dir(video_path)
    video_info = _extract_video_info(video_path)
    _write_video_info(run_dir, video_info)

    _, _, sampled_frames = _sample_base_frames(
        video_path=video_path,
        run_dir=run_dir,
        fps=float(video_info["fps"]),
        total_frames=int(video_info["total_frames"]),
        sample_every_seconds=max(0.1, float(args.sample_every_seconds)),
    )
    print(f"[tender-demo-yolo-first] Sampled frames: {len(sampled_frames)}")

    from step_10_yolo_detection import run_yolo_detection_on_selected_frames

    yolo_items = run_yolo_detection_on_selected_frames(run_dir)
    print(f"[tender-demo-yolo-first] YOLO detections on sampled frames: {len(yolo_items)}")

    tracking_report = _run_tracking_and_search_index(
        run_dir=run_dir,
        video_name=video_path.name,
        sampled_frames=sampled_frames,
        person_model=person_model,
        vehicle_model=vehicle_model,
        person_conf=float(args.person_conf),
        vehicle_conf=float(args.vehicle_conf),
        person_imgsz=int(args.person_imgsz),
        vehicle_imgsz=int(args.vehicle_imgsz),
    )
    print(
        "[tender-demo-yolo-first] Object search index built: "
        f"{tracking_report['tracked_object_count']} tracked objects"
    )

    _, motion_scores = _score_motion_on_sampled_frames(sampled_frames=sampled_frames, run_dir=run_dir)
    _, motion_candidates = _select_motion_candidates(
        motion_scores=motion_scores,
        run_dir=run_dir,
        motion_threshold=_read_motion_threshold(),
    )
    print(f"[tender-demo-yolo-first] Motion candidates: {len(motion_candidates)}")

    _, candidate_clips = _create_candidate_clips(
        motion_candidates=motion_candidates,
        run_dir=run_dir,
        max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
        max_clip_seconds=DEFAULT_MAX_CLIP_SECONDS,
        overlap_seconds=DEFAULT_CLIP_OVERLAP_SECONDS,
    )
    _, expanded_clips = _expand_candidate_clips(
        candidate_clips=candidate_clips,
        video_info=video_info,
        run_dir=run_dir,
        context_before_seconds=DEFAULT_CONTEXT_BEFORE_SECONDS,
        context_after_seconds=DEFAULT_CONTEXT_AFTER_SECONDS,
        min_expanded_clip_seconds=DEFAULT_MIN_EXPANDED_CLIP_SECONDS,
    )
    print(f"[tender-demo-yolo-first] Event candidate clips: {len(expanded_clips)}")

    yolo_scoring_report = run_yolo_object_scoring(run_dir)
    motion_state_report = estimate_object_motion_states(run_dir)
    plate_color_report = run_plate_ocr_color_enrichment(run_dir)
    ranked_report = rank_candidate_clips(run_dir)
    selected_report = select_topk_clips_for_qwen(run_dir)
    vlm_inputs_report = create_topk_vlm_inputs(run_dir)
    vlm_outputs = run_qwen_on_topk_vlm_inputs(run_dir)
    final_summary = create_topk_final_summary(run_dir)
    export_report = export_event_clips(run_dir)
    demo_report = create_demo_report_html(run_dir)

    _write_object_json(
        run_dir / "00_yolo_first_testcase_summary.json",
        {
            "pipeline_name": "tender_demo_yolo_first_test_case",
            "pipeline_steps": [
                "base frame sampling",
                "YOLO/object detection",
                "tracking",
                "object search index",
                "event candidate detection",
                "event-aware Top-K selection",
                "VLM input creation",
                "Qwen/OpenRouter VLM",
                "final event summary/report",
            ],
            "sampled_frame_count": len(sampled_frames),
            "yolo_detection_items": len(yolo_items),
            "tracked_object_count": tracking_report["tracked_object_count"],
            "event_candidate_clips": len(expanded_clips),
            "ranked_clips": len(ranked_report.get("ranked_clips", [])),
            "selected_topk_clips": len(selected_report.get("selected_clips", [])),
            "vlm_inputs": len(vlm_inputs_report.get("items", [])),
            "vlm_outputs": len(vlm_outputs),
            "priority_events": len(final_summary.get("priority_suspicious_events", [])),
            "report_path": demo_report.get("html_report_path"),
            "tracking_report": tracking_report,
            "yolo_scoring_summary": yolo_scoring_report.get("summary", {}),
            "motion_state_summary": motion_state_report.get("summary", {}),
            "plate_color_summary": plate_color_report,
            "export_summary": export_report,
        },
    )

    print("[tender-demo-yolo-first] Done")
    print(f"[tender-demo-yolo-first] Run dir: {run_dir}")
    print(f"[tender-demo-yolo-first] Summary file: {run_dir / '00_yolo_first_testcase_summary.json'}")


if __name__ == "__main__":
    main()
