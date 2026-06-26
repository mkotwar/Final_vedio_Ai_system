import asyncio
import importlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT
from app.services.qwen_vlm_hf import NativeQwenTransformersService


CASE_ROOT = Path(
    os.getenv(
        "BENCHMARK_PARENT_CASE_ROOT",
        str(PROJECT_ROOT / "tests" / "manual_benchmark_case" / "multi_video_runs"),
    )
)
SUMMARY_JSON_PATH = CASE_ROOT / "multi_video_benchmark_summary.json"
SUMMARY_MD_PATH = CASE_ROOT / "multi_video_benchmark_summary.md"


def _parse_video_paths() -> List[Path]:
    raw = os.getenv("BENCHMARK_INPUT_VIDEOS", "").strip()
    if not raw:
        raise ValueError(
            "BENCHMARK_INPUT_VIDEOS is required. Provide one or more full video paths separated by ';'."
        )
    return [Path(item.strip()) for item in raw.split(";") if item.strip()]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "video"


async def main() -> None:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    videos = _parse_video_paths()

    preload_start = time.perf_counter()
    NativeQwenTransformersService.get_runtime()
    preload_seconds = time.perf_counter() - preload_start

    benchmark_module = None
    runs: List[Dict[str, Any]] = []
    overall_start = time.perf_counter()

    for index, video_path in enumerate(videos, start=1):
        run_name = f"{index:02d}_{_safe_name(video_path.stem)}"
        run_root = CASE_ROOT / run_name
        os.environ["BENCHMARK_CASE_ROOT"] = str(run_root)
        os.environ["BENCHMARK_INPUT_VIDEO"] = str(video_path)
        os.environ["BENCHMARK_VIDEO_ID"] = f"manual-test-{uuid.uuid4()}"

        if benchmark_module is None:
            benchmark_module = importlib.import_module(
                "tests.manual_benchmark_case.run_event_candidate_reasoning_benchmark"
            )
        else:
            benchmark_module = importlib.reload(benchmark_module)

        run_start = time.perf_counter()
        await benchmark_module.main()
        run_seconds = time.perf_counter() - run_start

        summary_path = run_root / "data" / "output" / "event_candidate_benchmark_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

        runs.append(
            {
                "video_path": str(video_path),
                "case_root": str(run_root),
                "runtime_seconds": run_seconds,
                "summary_path": str(summary_path),
                "selected_variants": summary.get("selected_variants", []),
                "selected_modes": summary.get("selected_modes", []),
                "total_candidate_events": summary.get("total_candidate_events"),
                "overall_runtime_seconds": summary.get("overall_runtime_seconds"),
            }
        )

    combined = {
        "video_count": len(videos),
        "preload_seconds": preload_seconds,
        "wall_clock_seconds": time.perf_counter() - overall_start,
        "runs": runs,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4)

    lines = [
        "# Multi-Video Event Candidate Benchmark",
        "",
        f"- Video count: `{combined['video_count']}`",
        f"- One-time Qwen preload: `{combined['preload_seconds']:.2f}s`",
        f"- Total wall-clock runtime: `{combined['wall_clock_seconds']:.2f}s`",
        "",
        "## Runs",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"- Video: `{run['video_path']}`",
                f"  - Case root: `{run['case_root']}`",
                f"  - Runtime: `{run['runtime_seconds']:.2f}s`",
                f"  - Summary: `{run['summary_path']}`",
                f"  - Modes: `{run['selected_modes']}`",
                f"  - Variants: `{run['selected_variants']}`",
                f"  - Candidate events: `{run['total_candidate_events']}`",
                "",
            ]
        )

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("MULTI_VIDEO_EVENT_CANDIDATE_BENCHMARK_START")
    print(json.dumps(combined))
    print("MULTI_VIDEO_EVENT_CANDIDATE_BENCHMARK_END")


if __name__ == "__main__":
    asyncio.run(main())
