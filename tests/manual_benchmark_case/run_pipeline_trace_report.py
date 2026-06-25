import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT
from app.core.utils import format_timestamp_human
from app.services.object_detection.schemas import FrameDetection
from app.services.object_tracker import ObjectTrackerService


INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-777777777777")

CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
OUTPUT_ROOT = CASE_ROOT / "data" / "output"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames" / BENCHMARK_VIDEO_ID
DETECTION_ROOT = PROJECT_ROOT / "data" / "detections" / BENCHMARK_VIDEO_ID
REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs"

EVENT_SUMMARY_PATH = OUTPUT_ROOT / "event_candidate_benchmark_summary.json"
EVENT_TIMELINE_PATH = OUTPUT_ROOT / "event_candidate_timeline.json"
PROMPT_EXAMPLES_PATH = OUTPUT_ROOT / "reasoning_prompt_examples.json"

TRACE_JSON_PATH = OUTPUT_ROOT / "pipeline_trace_report.json"
TRACE_MD_PATH = OUTPUT_ROOT / "PIPELINE_TRACE_REPORT.md"

PRIMARY_MODE = "candidate_only"
PRIMARY_VARIANT = "strip_tokens150_batch1"


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_detections() -> List[FrameDetection]:
    files = sorted(DETECTION_ROOT.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No detection JSON files found in {DETECTION_ROOT}")
    return [FrameDetection.model_validate_json(path.read_text(encoding="utf-8")) for path in files]


def _build_frame_paths() -> Dict[str, Path]:
    files = sorted(FRAME_ROOT.glob("frame_*.jpg"))
    frame_paths: Dict[str, Path] = {}
    for index, path in enumerate(files, start=1):
        frame_id = f"{BENCHMARK_VIDEO_ID}_f{index:04d}"
        frame_paths[frame_id] = path
    return frame_paths


def _prompt_lookup(prompt_examples: List[Dict[str, Any]]) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in prompt_examples:
        key = (row["mode"], row["variant_name"], row["event_id"])
        lookup[key] = row
    return lookup


def _summarize_detections(frame_detection: FrameDetection) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for detection in frame_detection.detections:
        rows.append(
            {
                "class_name": detection.class_name,
                "confidence": round(float(detection.confidence), 4),
                "bbox": [round(float(v), 2) for v in detection.bbox],
            }
        )
    return rows


def _frame_trace_rows(
    frame_detections: List[FrameDetection],
    tracking_map: Dict[str, Dict[str, Any]],
    candidate_events: List[Dict[str, Any]],
    mode_jobs: List[Dict[str, Any]],
    frame_paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    frame_to_events: Dict[str, List[str]] = defaultdict(list)
    frame_to_selected_events: Dict[str, List[str]] = defaultdict(list)
    for event in candidate_events:
        for frame_id in event["frame_ids"]:
            frame_to_events[frame_id].append(event["event_id"])
        for frame_id in event["selected_frames"]:
            frame_to_selected_events[frame_id].append(event["event_id"])

    frame_to_periodic_jobs: Dict[str, List[str]] = defaultdict(list)
    for job in mode_jobs:
        if not job.get("periodic"):
            continue
        for frame_id in job["selected_frame_ids"]:
            frame_to_periodic_jobs[frame_id].append(job["event_id"])

    rows: List[Dict[str, Any]] = []
    for frame_detection in frame_detections:
        frame_id = frame_detection.frame_id
        tracking = tracking_map.get(frame_id, {})
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_seconds": frame_detection.timestamp_seconds,
                "timestamp_human": format_timestamp_human(frame_detection.timestamp_seconds),
                "frame_path": str(frame_paths.get(frame_id, "")),
                "detector_source": "YOLO (ObjectDetector.detect_frame)",
                "detector_output": {
                    "detection_count": len(frame_detection.detections),
                    "detections": _summarize_detections(frame_detection),
                },
                "tracker_source": "IoU tracker (ObjectTrackerService.track_frames)",
                "tracker_output": {
                    "track_ids": tracking.get("track_ids", []),
                    "class_counts": tracking.get("class_counts", {}),
                    "new_track_count": tracking.get("new_track_count", 0),
                    "ended_track_count": tracking.get("ended_track_count", 0),
                    "tracked_entities": tracking.get("tracked_entities", []),
                },
                "candidate_event_ids": frame_to_events.get(frame_id, []),
                "selected_for_candidate_reasoning": frame_to_selected_events.get(frame_id, []),
                "selected_for_periodic_reasoning": frame_to_periodic_jobs.get(frame_id, []),
            }
        )
    return rows


def _event_trace_rows(
    candidate_events: List[Dict[str, Any]],
    primary_results_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[tuple[str, str, str], Dict[str, Any]],
    frame_paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in candidate_events:
        event_id = event["event_id"]
        result = primary_results_lookup.get(event_id, {})
        prompt_row = prompt_lookup.get((PRIMARY_MODE, PRIMARY_VARIANT, event_id), {})
        reasoning_image = REASONING_INPUT_ROOT / f"{PRIMARY_MODE}_{PRIMARY_VARIANT}_{event_id}.jpg"
        rows.append(
            {
                "event_id": event_id,
                "event_source": "Benchmark event-candidate heuristic clustering",
                "event_window": {
                    "start_seconds": event["start_seconds"],
                    "end_seconds": event["end_seconds"],
                    "start_human": format_timestamp_human(event["start_seconds"]),
                    "end_human": format_timestamp_human(event["end_seconds"]),
                },
                "candidate_metadata": {
                    "event_type": event["event_type"],
                    "reason": event["reason"],
                    "persons": event["persons"],
                    "bags": event["bags"],
                    "dwell_seconds": event["dwell_seconds"],
                    "track_ids": event["track_ids"],
                    "ocr_text": event["ocr_text"],
                },
                "frame_ids": event["frame_ids"],
                "frame_paths": [str(frame_paths[frame_id]) for frame_id in event["frame_ids"] if frame_id in frame_paths],
                "selected_frame_ids": event["selected_frames"],
                "selected_frame_paths": [
                    str(frame_paths[frame_id]) for frame_id in event["selected_frames"] if frame_id in frame_paths
                ],
                "reasoning_input_image": str(reasoning_image) if reasoning_image.exists() else "",
                "prompt_context_source": "Structured context assembled from tracker counts, OCR text, dwell, and heuristic reason",
                "prompt_context": prompt_row.get("structured_context", result.get("structured_context")),
                "reasoning_prompt": prompt_row.get("prompt", ""),
                "vlm_source": "Qwen HF benchmark run",
                "vlm_result": {
                    "success": result.get("success"),
                    "raw_output": result.get("raw_output"),
                    "cleaned_output": result.get("cleaned_output"),
                    "parsed_output": result.get("parsed_output"),
                    "output_tokens": result.get("output_tokens"),
                    "failure_category": result.get("failure_category"),
                },
            }
        )
    return rows


def _write_markdown(report: Dict[str, Any]) -> None:
    stage = report["stage_summary"]
    primary = report["primary_reasoning_variant"]
    lines = [
        "# Pipeline Trace Report",
        "",
        "## Scope",
        "",
        f"- Input video: `{report['input_video_path']}`",
        f"- Video id: `{report['video_id']}`",
        f"- Frames folder: `{report['frames_folder']}`",
        f"- Detections folder: `{report['detections_folder']}`",
        f"- Reasoning inputs folder: `{report['reasoning_inputs_folder']}`",
        "",
        "## Stage Summary",
        "",
        f"- Extracted frames: `{stage['extracted_frames']}`",
        f"- Detection JSON files: `{stage['detection_json_files']}`",
        f"- Candidate events: `{stage['candidate_events']}`",
        f"- Primary reasoning mode: `{primary['mode']}` / `{primary['variant']}`",
        f"- Frames sent to Qwen in primary mode: `{primary['frames_sent_to_qwen']}`",
        f"- Successful VLM responses: `{primary['successful_responses']}`",
        f"- Failed VLM responses: `{primary['failed_responses']}`",
        f"- Primary-mode latency: `{primary['wall_clock_runtime_seconds']:.2f}s`",
        "",
        "## Who Generates What",
        "",
        "- `ObjectDetector.detect_frame` (YOLO): object boxes, classes, confidences.",
        "- `ObjectTrackerService.track_frames`: track ids, class counts, new/ended tracks.",
        "- `OCRService.extract_text`: OCR text used in event context.",
        "- benchmark event-candidate logic: candidate events, selected keyframes, dwell, heuristic event type/reason.",
        "- Qwen HF benchmark: compact semantic JSON for each selected event.",
        "",
        "## Event Overview",
        "",
    ]

    for event in report["event_traces"]:
        lines.extend(
            [
                f"### {event['event_id']}",
                f"- Window: `{event['event_window']['start_human']} -> {event['event_window']['end_human']}`",
                f"- Heuristic event type: `{event['candidate_metadata']['event_type']}`",
                f"- Reason: `{event['candidate_metadata']['reason']}`",
                f"- Persons/Bags: `{event['candidate_metadata']['persons']}` / `{event['candidate_metadata']['bags']}`",
                f"- Track ids: `{event['candidate_metadata']['track_ids']}`",
                f"- Selected frames: `{event['selected_frame_ids']}`",
                f"- Reasoning image: `{event['reasoning_input_image']}`",
                f"- VLM parsed output: `{event['vlm_result']['parsed_output']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- `frame_traces` in the JSON report shows every extracted frame and whether it participated in candidate or periodic reasoning.",
            "- `event_traces` in the JSON report connects each event to its frames, OCR, prompt context, and VLM output.",
        ]
    )

    TRACE_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = _load_json(EVENT_SUMMARY_PATH)
    timeline = _load_json(EVENT_TIMELINE_PATH)
    prompt_examples = _load_json(PROMPT_EXAMPLES_PATH)
    frame_detections = _load_detections()
    tracking_map = ObjectTrackerService.track_frames(frame_detections)
    frame_paths = _build_frame_paths()

    candidate_events = timeline["candidate_events"]
    primary_mode_summary = summary["modes"][PRIMARY_MODE][PRIMARY_VARIANT]
    primary_mode_jobs = timeline["modes"][PRIMARY_MODE][PRIMARY_VARIANT]
    primary_results_lookup = {row["event_id"]: row for row in primary_mode_summary["results"]}
    prompts_lookup = _prompt_lookup(prompt_examples)

    report = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "frames_folder": str(FRAME_ROOT),
        "detections_folder": str(DETECTION_ROOT),
        "reasoning_inputs_folder": str(REASONING_INPUT_ROOT),
        "stage_summary": {
            "extracted_frames": len(frame_detections),
            "detection_json_files": len(list(DETECTION_ROOT.glob("*.json"))),
            "candidate_events": len(candidate_events),
        },
        "primary_reasoning_variant": {
            "mode": PRIMARY_MODE,
            "variant": PRIMARY_VARIANT,
            "frames_sent_to_qwen": primary_mode_summary["frames_sent_to_qwen"],
            "successful_responses": primary_mode_summary["successful_responses"],
            "failed_responses": primary_mode_summary["failed_responses"],
            "wall_clock_runtime_seconds": primary_mode_summary["wall_clock_runtime_seconds"],
        },
        "frame_traces": _frame_trace_rows(
            frame_detections=frame_detections,
            tracking_map=tracking_map,
            candidate_events=candidate_events,
            mode_jobs=timeline["modes"]["candidate_plus_periodic10s"][PRIMARY_VARIANT],
            frame_paths=frame_paths,
        ),
        "event_traces": _event_trace_rows(
            candidate_events=candidate_events,
            primary_results_lookup=primary_results_lookup,
            prompt_lookup=prompts_lookup,
            frame_paths=frame_paths,
        ),
    }

    TRACE_JSON_PATH.write_text(json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report)

    print("PIPELINE_TRACE_REPORT_START")
    print(json.dumps({"json": str(TRACE_JSON_PATH), "markdown": str(TRACE_MD_PATH)}))
    print("PIPELINE_TRACE_REPORT_END")


if __name__ == "__main__":
    main()
