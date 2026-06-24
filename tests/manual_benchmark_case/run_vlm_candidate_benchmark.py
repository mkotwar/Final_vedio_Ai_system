import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.services.frame import FrameExtractionService
from app.services.pipeline_contract import event_catalog_path, frame_catalog_path, frame_metadata_dir
from app.services.status_service import JobStatusService


INPUT_VIDEO_PATH = Path(r"C:\Mukul K\test_video\V_ai_test_2min.mp4")
BENCHMARK_VIDEO_ID = "11111111-2222-4333-8444-555555555555"
CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
PROJECT_VIDEO_PATH = settings.VIDEOS_DIR / f"{BENCHMARK_VIDEO_ID}{INPUT_VIDEO_PATH.suffix.lower()}"
PROJECT_METADATA_PATH = settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "summary.md"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "vlm_candidate_timeline.json"
OUTPUT_VIDEO_PATH = OUTPUT_ROOT / "vlm_candidate_frames.mp4"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)
    shutil.copy2(INPUT_VIDEO_PATH, PROJECT_VIDEO_PATH)


def _write_project_metadata() -> None:
    metadata = {
        "video_id": BENCHMARK_VIDEO_ID,
        "filename": INPUT_VIDEO_PATH.name,
        "upload_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_size": PROJECT_VIDEO_PATH.stat().st_size,
        "upload_duration_ms": 0.0,
    }
    with open(PROJECT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_artifacts() -> None:
    _safe_remove(settings.FRAMES_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_metadata_dir(BENCHMARK_VIDEO_ID))
    _safe_remove(settings.EVENTS_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_catalog_path(BENCHMARK_VIDEO_ID))
    _safe_remove(event_catalog_path(BENCHMARK_VIDEO_ID))
    _safe_remove(settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}_status.json")


def _get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0:
        return 0.0
    return float(total_frames / fps)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_candidate_video(frames: List[Dict[str, Any]]) -> None:
    if not frames:
        return

    first_frame_path = PROJECT_ROOT / frames[0]["frame_path"]
    first_image = cv2.imread(str(first_frame_path))
    if first_image is None:
        return

    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (width, height),
    )

    try:
        for frame in frames:
            image_path = PROJECT_ROOT / frame["frame_path"]
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            cv2.rectangle(image, (0, 0), (width, 92), (0, 0, 0), thickness=-1)
            cv2.putText(
                image,
                f"Frame: {frame.get('frame_id', '')}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Timestamp: {frame.get('timestamp_human', '')} ({frame.get('timestamp_seconds', 0):.1f}s)",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            reasons = ", ".join(frame.get("candidate_reasons", []))
            cv2.putText(
                image,
                f"Reasons: {reasons[:110]}",
                (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()


def _build_timeline(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    for frame in frames:
        timeline.append(
            {
                "frame_id": frame.get("frame_id"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "timestamp_human": frame.get("timestamp_human"),
                "candidate_reasons": frame.get("candidate_reasons", []),
                "activities": frame.get("activities", []),
                "caption": frame.get("caption", ""),
                "detected_objects": [item.get("class_name") for item in frame.get("detected_objects", [])],
                "track_ids": frame.get("track_ids", []),
            }
        )
    return timeline


def _write_summary(summary: Dict[str, Any]) -> None:
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    md = [
        "# Manual Benchmark Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Input copy: `{summary['input_copy_path']}`",
        f"- Output candidate video: `{summary['output_video_path']}`",
        f"- Video ID: `{summary['video_id']}`",
        f"- VLM engine: `{summary['vlm_engine_type']}`",
        f"- Configured batch size: `{summary['batch_size']}`",
        f"- Source video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Wall-clock latency: `{summary['wall_clock_seconds']:.2f}s`",
        f"- Realtime ratio: `{summary['realtime_ratio']:.3f}x`",
        f"- Faster than video length: `{summary['faster_than_video_length']}`",
        f"- Total frames extracted: `{summary['total_frames_extracted']}`",
        f"- Frames retained for coverage: `{summary['frames_retained_for_coverage']}`",
        f"- Frames sent to Qwen: `{summary['frames_sent_to_qwen']}`",
        f"- Frames filtered before VLM: `{summary['frames_filtered_before_vlm']}`",
        f"- Events generated: `{summary['event_count']}`",
    ]
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


async def main() -> None:
    _ensure_dirs()
    _clean_previous_artifacts()
    _copy_input_video()
    _write_project_metadata()

    old_batch_size = settings.BATCH_SIZE
    settings.BATCH_SIZE = 4
    JobStatusService.initialize(BENCHMARK_VIDEO_ID)

    try:
        start = time.perf_counter()
        stats = await FrameExtractionService.extract_frames(BENCHMARK_VIDEO_ID)
        wall_clock_seconds = time.perf_counter() - start
    finally:
        settings.BATCH_SIZE = old_batch_size

    frames = stats.get("frames", [])
    events = _load_json(event_catalog_path(BENCHMARK_VIDEO_ID), [])
    timeline = _build_timeline(frames)
    with open(TIMELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=4)

    _render_candidate_video(frames)

    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    realtime_ratio = (wall_clock_seconds / video_duration_seconds) if video_duration_seconds > 0 else 0.0
    faster_than_video_length = wall_clock_seconds < video_duration_seconds if video_duration_seconds > 0 else None

    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "output_video_path": str(OUTPUT_VIDEO_PATH),
        "timeline_json_path": str(TIMELINE_JSON_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "vlm_engine_type": settings.VLM_ENGINE_TYPE,
        "batch_size": 4,
        "video_duration_seconds": video_duration_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "realtime_ratio": realtime_ratio,
        "faster_than_video_length": faster_than_video_length,
        "total_frames_extracted": stats.get("total_frames_extracted", 0),
        "frames_retained_for_coverage": stats.get("frames_retained_for_coverage", 0),
        "frames_sent_to_qwen": stats.get("frames_sent_to_qwen", 0),
        "frames_filtered_before_vlm": stats.get("frames_filtered_before_vlm", 0),
        "processed_frames": stats.get("processed_frames", 0),
        "successful_frames": stats.get("successful_frames", 0),
        "failed_frames": stats.get("failed_frames", 0),
        "event_count": len(events),
        "event_ids": [event.get("event_id") for event in events],
    }
    _write_summary(summary)

    print("MANUAL_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary))
    print("MANUAL_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
