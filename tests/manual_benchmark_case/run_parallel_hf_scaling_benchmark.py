import asyncio
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.services.ocr import OCRService
from app.services.qwen_vlm_hf import NativeQwenTransformersService


CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
OUTPUT_ROOT = CASE_ROOT / "data" / "output"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "event_driven_candidate_timeline.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "parallel_hf_scaling_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "parallel_hf_scaling_summary.md"

SAMPLE_FRAME_COUNT = 12
TEST_BATCH_SIZES = [4, 8, 12]


def _ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_timeline() -> List[Dict[str, Any]]:
    if not TIMELINE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {TIMELINE_JSON_PATH}. Run event-driven benchmark first."
        )
    with open(TIMELINE_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_timeline_rows(rows: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if len(rows) <= sample_size:
        return rows

    sampled: List[Dict[str, Any]] = []
    seen = set()
    step = (len(rows) - 1) / max(1, sample_size - 1)
    for idx in range(sample_size):
        row_index = min(len(rows) - 1, int(round(idx * step)))
        frame_id = rows[row_index]["frame_id"]
        if frame_id in seen:
            continue
        seen.add(frame_id)
        sampled.append(rows[row_index])

    if len(sampled) < sample_size:
        for row in rows:
            frame_id = row["frame_id"]
            if frame_id in seen:
                continue
            sampled.append(row)
            seen.add(frame_id)
            if len(sampled) >= sample_size:
                break

    return sampled


def _build_batch_tuples(rows: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    tuples: List[Tuple[Any, ...]] = []
    for row in rows:
        frame_path = Path(row["frame_path"])
        detection_context = {
            "candidate_reasons": row.get("candidate_reasons", []),
            "track_ids": row.get("track_ids", []),
            "detected_objects": row.get("detected_objects", []),
        }
        tuples.append(
            (
                row["frame_id"],
                row["video_id"],
                float(row["timestamp_seconds"]),
                frame_path,
                frame_path,
                detection_context,
            )
        )
    return tuples


def _precompute_ocr_cache(frame_paths: List[Path]) -> Dict[str, Dict[str, List[str]]]:
    cache: Dict[str, Dict[str, List[str]]] = {}

    def run_one(path: Path) -> Tuple[str, Dict[str, List[str]]]:
        return str(path), OCRService.extract_text(path)

    max_workers = min(4, max(1, len(frame_paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, value in executor.map(run_one, frame_paths):
            cache[key] = value
    return cache


async def _run_config(
    label: str,
    batch_size: int,
    batch_tuples: List[Tuple[Any, ...]],
    ocr_cache: Dict[str, Dict[str, List[str]]] | None = None,
) -> Dict[str, Any]:
    old_batch_size = settings.BATCH_SIZE
    settings.BATCH_SIZE = batch_size
    original_extract_text = OCRService.extract_text

    if ocr_cache is not None:
        OCRService.extract_text = classmethod(lambda cls, image_path: ocr_cache.get(str(Path(image_path)), {"detected_text": [], "license_plates": []}))

    try:
        start = time.perf_counter()
        results = await NativeQwenTransformersService.generate_metadata_batch(batch_tuples)
        wall_clock_seconds = time.perf_counter() - start
    finally:
        settings.BATCH_SIZE = old_batch_size
        OCRService.extract_text = original_extract_text

    successful_frames = len(results)
    failed_frames = max(0, len(batch_tuples) - successful_frames)
    total_vlm_ms = sum(item[1].get("vlm_ms", 0.0) for item in results)
    total_ocr_ms = sum(item[1].get("ocr_ms", 0.0) for item in results)
    total_validation_ms = sum(item[1].get("validation_ms", 0.0) for item in results)
    total_json_repair_ms = sum(item[1].get("json_repair_ms", 0.0) for item in results)

    return {
        "label": label,
        "batch_size": batch_size,
        "frames_processed": len(batch_tuples),
        "successful_frames": successful_frames,
        "failed_frames": failed_frames,
        "wall_clock_seconds": wall_clock_seconds,
        "avg_wall_clock_per_frame_seconds": wall_clock_seconds / max(1, len(batch_tuples)),
        "avg_vlm_ms_reported": (total_vlm_ms / max(1, successful_frames)) if successful_frames else 0.0,
        "avg_ocr_ms_reported": (total_ocr_ms / max(1, successful_frames)) if successful_frames else 0.0,
        "avg_validation_ms_reported": (total_validation_ms / max(1, successful_frames)) if successful_frames else 0.0,
        "avg_json_repair_ms_reported": (total_json_repair_ms / max(1, successful_frames)) if successful_frames else 0.0,
    }


def _build_conclusion(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = next(item for item in results if item["label"] == "warm_batch4_uncached")
    best = min(results, key=lambda item: item["avg_wall_clock_per_frame_seconds"])
    improvement_pct = 0.0
    if baseline["avg_wall_clock_per_frame_seconds"] > 0:
        improvement_pct = (
            (baseline["avg_wall_clock_per_frame_seconds"] - best["avg_wall_clock_per_frame_seconds"])
            / baseline["avg_wall_clock_per_frame_seconds"]
        ) * 100.0

    answer_lines = [
        f"Baseline warm batch-4 uncached cost is {baseline['avg_wall_clock_per_frame_seconds']:.2f}s/frame.",
        f"Best observed configuration is {best['label']} at {best['avg_wall_clock_per_frame_seconds']:.2f}s/frame.",
        f"That is a {improvement_pct:.1f}% per-frame improvement versus the warm uncached batch-4 baseline.",
    ]

    if best["label"] == "warm_batch4_uncached":
        answer_lines.append(
            "On this benchmark slice, larger batches and cached OCR did not materially improve the steady-state path."
        )
    else:
        answer_lines.append(
            "On this benchmark slice, keeping the model warm and moving OCR off the critical path helped, and batch scaling changed effective per-frame cost."
        )

    return {
        "best_label": best["label"],
        "improvement_percent_vs_baseline": improvement_pct,
        "answer_lines": answer_lines,
    }


async def main() -> None:
    _ensure_output_dir()
    rows = _load_timeline()
    sampled_rows = _sample_timeline_rows(rows, SAMPLE_FRAME_COUNT)
    batch_tuples = _build_batch_tuples(sampled_rows)
    frame_paths = [Path(row["frame_path"]) for row in sampled_rows]

    load_start = time.perf_counter()
    NativeQwenTransformersService.load_model()
    model_load_seconds = time.perf_counter() - load_start

    ocr_cache = _precompute_ocr_cache(frame_paths)

    results: List[Dict[str, Any]] = []
    results.append(await _run_config("warm_batch4_uncached", 4, batch_tuples, ocr_cache=None))
    for batch_size in TEST_BATCH_SIZES:
        results.append(await _run_config(f"warm_batch{batch_size}_ocr_cached", batch_size, batch_tuples, ocr_cache=ocr_cache))

    conclusion = _build_conclusion(results)
    summary = {
        "model_id": settings.QWEN_MODEL_ID,
        "engine": "native_hf",
        "sample_frame_count": len(batch_tuples),
        "sample_frame_ids": [item[0] for item in batch_tuples],
        "sample_timestamps": [item[2] for item in batch_tuples],
        "model_load_seconds": model_load_seconds,
        "results": results,
        "conclusion": conclusion,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    md_lines = [
        "# Parallel HF Scaling Summary",
        "",
        f"- Model: `{settings.QWEN_MODEL_ID}`",
        f"- Sampled frames: `{len(batch_tuples)}`",
        f"- One-time warm load: `{model_load_seconds:.2f}s`",
        "",
        "## Results",
        "",
    ]
    for result in results:
        md_lines.append(
            f"- `{result['label']}`: {result['avg_wall_clock_per_frame_seconds']:.2f}s/frame "
            f"({result['successful_frames']}/{result['frames_processed']} succeeded)"
        )
    md_lines.extend(["", "## Answer", ""])
    md_lines.extend(f"- {line}" for line in conclusion["answer_lines"])

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("PARALLEL_HF_SCALING_SUMMARY_START")
    print(json.dumps(summary))
    print("PARALLEL_HF_SCALING_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
