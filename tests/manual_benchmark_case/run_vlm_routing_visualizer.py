import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT
from app.core.utils import format_timestamp_human


INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-777777777777")

CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
OUTPUT_ROOT = CASE_ROOT / "data" / "output"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames" / BENCHMARK_VIDEO_ID

TRACE_REPORT_PATH = OUTPUT_ROOT / "pipeline_trace_report.json"
TIMELINE_PATH = OUTPUT_ROOT / "event_candidate_timeline.json"

ROUTING_JSON_PATH = OUTPUT_ROOT / "vlm_routing_map.json"
ROUTING_MD_PATH = OUTPUT_ROOT / "VLM_ROUTING_MAP.md"
VLM_RAW_VIDEO_PATH = OUTPUT_ROOT / "vlm_raw_selected_frames.mp4"
VLM_STRIP_VIDEO_PATH = OUTPUT_ROOT / "vlm_reasoning_inputs_video.mp4"

PRIMARY_MODE = "candidate_only"
PRIMARY_VARIANT = "strip_tokens150_batch1"


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_index(frame_id: str) -> int:
    return int(frame_id.rsplit("_f", 1)[-1])


def _put_lines(image: np.ndarray, lines: List[str]) -> np.ndarray:
    out = image.copy()
    panel_height = 34 + 28 * len(lines)
    cv2.rectangle(out, (0, 0), (out.shape[1], panel_height), (0, 0, 0), thickness=-1)
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _write_video(frames: List[np.ndarray], out_path: Path, fps: float = 1.0) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def _build_raw_selected_video(event_traces: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    routed_rows: List[Dict[str, Any]] = []
    video_frames: List[np.ndarray] = []
    seen: set[str] = set()

    for event in sorted(event_traces, key=lambda row: row["event_window"]["start_seconds"]):
        for frame_id, frame_path in zip(event["selected_frame_ids"], event["selected_frame_paths"]):
            if frame_id in seen:
                continue
            seen.add(frame_id)

            image = cv2.imread(frame_path)
            if image is None:
                continue

            lines = [
                f"VLM RAW FRAME | {frame_id}",
                f"time={event['event_window']['start_human']}..{event['event_window']['end_human']} | event={event['event_id']}",
                f"heuristic={event['candidate_metadata']['event_type']} | persons={event['candidate_metadata']['persons']} | bags={event['candidate_metadata']['bags']}",
            ]
            annotated = _put_lines(image, lines)
            video_frames.append(annotated)
            routed_rows.append(
                {
                    "frame_id": frame_id,
                    "frame_path": frame_path,
                    "route": "selected_for_vlm_raw_context",
                    "event_id": event["event_id"],
                    "event_type": event["candidate_metadata"]["event_type"],
                }
            )

    return routed_rows, video_frames


def _build_reasoning_input_video(event_traces: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    routed_rows: List[Dict[str, Any]] = []
    video_frames: List[np.ndarray] = []

    for event in sorted(event_traces, key=lambda row: row["event_window"]["start_seconds"]):
        path = event["reasoning_input_image"]
        if not path:
            continue
        image = cv2.imread(path)
        if image is None:
            continue

        lines = [
            f"ACTUAL QWEN INPUT IMAGE | {event['event_id']}",
            f"mode={PRIMARY_MODE} | variant={PRIMARY_VARIANT}",
            f"time={event['event_window']['start_human']}..{event['event_window']['end_human']}",
        ]
        annotated = _put_lines(image, lines)
        video_frames.append(annotated)
        routed_rows.append(
            {
                "event_id": event["event_id"],
                "reasoning_input_image": path,
                "route": "actual_qwen_image_input",
            }
        )

    return routed_rows, video_frames


def _build_routing_map(trace_report: Dict[str, Any], timeline: Dict[str, Any]) -> Dict[str, Any]:
    frame_rows = sorted(trace_report["frame_traces"], key=lambda row: _frame_index(row["frame_id"]))
    event_rows = sorted(trace_report["event_traces"], key=lambda row: row["event_window"]["start_seconds"])

    frame_routes: List[Dict[str, Any]] = []
    for row in frame_rows:
        frame_routes.append(
            {
                "frame_id": row["frame_id"],
                "timestamp_seconds": row["timestamp_seconds"],
                "timestamp_human": row["timestamp_human"],
                "frame_path": row["frame_path"],
                "went_to_yolo": True,
                "went_to_tracker": True,
                "candidate_event_ids": row["candidate_event_ids"],
                "selected_for_candidate_reasoning": row["selected_for_candidate_reasoning"],
                "selected_for_periodic_reasoning": row["selected_for_periodic_reasoning"],
            }
        )

    raw_selected_rows, raw_video_frames = _build_raw_selected_video(event_rows)
    reasoning_rows, strip_video_frames = _build_reasoning_input_video(event_rows)

    _write_video(raw_video_frames, VLM_RAW_VIDEO_PATH, fps=1.0)
    _write_video(strip_video_frames, VLM_STRIP_VIDEO_PATH, fps=1.0)

    return {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "primary_mode": PRIMARY_MODE,
        "primary_variant": PRIMARY_VARIANT,
        "summary": {
            "total_extracted_frames": len(frame_routes),
            "total_candidate_events": len(event_rows),
            "raw_selected_frames_for_vlm_context": len(raw_selected_rows),
            "actual_qwen_image_inputs": len(reasoning_rows),
            "raw_selected_video": str(VLM_RAW_VIDEO_PATH),
            "reasoning_inputs_video": str(VLM_STRIP_VIDEO_PATH),
        },
        "frame_routes": frame_routes,
        "event_routes": event_rows,
        "vlm_raw_frame_sequence": raw_selected_rows,
        "vlm_image_input_sequence": reasoning_rows,
    }


def _write_markdown(report: Dict[str, Any]) -> None:
    lines = [
        "# VLM Routing Map",
        "",
        f"- Input video: `{report['input_video_path']}`",
        f"- Primary mode/variant: `{report['primary_mode']} / {report['primary_variant']}`",
        f"- Extracted frames: `{report['summary']['total_extracted_frames']}`",
        f"- Candidate events: `{report['summary']['total_candidate_events']}`",
        f"- Raw selected frames used to build VLM context: `{report['summary']['raw_selected_frames_for_vlm_context']}`",
        f"- Actual Qwen image inputs: `{report['summary']['actual_qwen_image_inputs']}`",
        f"- Raw selected frame video: `{report['summary']['raw_selected_video']}`",
        f"- Reasoning input video: `{report['summary']['reasoning_inputs_video']}`",
        "",
        "## Event Order",
        "",
    ]

    for event in report["event_routes"]:
        lines.extend(
            [
                f"### {event['event_id']}",
                f"- Window: `{event['event_window']['start_human']} -> {event['event_window']['end_human']}`",
                f"- Heuristic type: `{event['candidate_metadata']['event_type']}`",
                f"- Selected frame ids: `{event['selected_frame_ids']}`",
                f"- Qwen input image: `{event['reasoning_input_image']}`",
                "",
            ]
        )

    ROUTING_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trace_report = _load_json(TRACE_REPORT_PATH)
    timeline = _load_json(TIMELINE_PATH)
    report = _build_routing_map(trace_report, timeline)
    ROUTING_JSON_PATH.write_text(json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report)

    print("VLM_ROUTING_VISUALIZER_START")
    print(
        json.dumps(
            {
                "routing_json": str(ROUTING_JSON_PATH),
                "routing_markdown": str(ROUTING_MD_PATH),
                "raw_selected_video": str(VLM_RAW_VIDEO_PATH),
                "reasoning_inputs_video": str(VLM_STRIP_VIDEO_PATH),
            }
        )
    )
    print("VLM_ROUTING_VISUALIZER_END")


if __name__ == "__main__":
    main()
