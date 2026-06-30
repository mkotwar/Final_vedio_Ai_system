from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    ENV_FINAL_DEMO_CHUNK_OVERLAP_SECONDS,
    ENV_FINAL_DEMO_CHUNK_SECONDS,
    build_chunk_manifest,
    read_json,
    read_non_negative_float_env,
    read_positive_float_env,
    update_run_manifest,
)
from tests.final_demo.services.video_io import write_json


def run_step_02_chunk_planner(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 2: chunk planner")

    video_info_path = run_dir / "01_video_info.json"
    if not video_info_path.exists():
        raise FileNotFoundError(f"Missing Step 1 video info file: {video_info_path}")

    video_info = read_json(video_info_path)
    duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
    video_path = str(video_info.get("video_path", ""))
    if not video_path:
        raise ValueError(f"Missing video_path in Step 1 video info file: {video_info_path}")

    chunk_seconds = read_positive_float_env(
        ENV_FINAL_DEMO_CHUNK_SECONDS,
        DEFAULT_CHUNK_SECONDS,
    )
    overlap_seconds = read_non_negative_float_env(
        ENV_FINAL_DEMO_CHUNK_OVERLAP_SECONDS,
        DEFAULT_CHUNK_OVERLAP_SECONDS,
    )

    chunk_manifest = build_chunk_manifest(
        video_path=video_path,
        duration_seconds=duration_seconds,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    chunk_manifest_path = run_dir / "02_chunk_manifest.json"
    write_json(chunk_manifest_path, chunk_manifest)

    update_run_manifest(run_dir / "run_manifest.json")

    print(f"[final-demo] Chunk size: {chunk_manifest['chunk_seconds']} seconds")
    print(f"[final-demo] Overlap: {chunk_manifest['overlap_seconds']} seconds")
    print(f"[final-demo] Total chunks: {chunk_manifest['total_chunks']}")
    print(f"[final-demo] Manifest path: {chunk_manifest_path}")

    return {
        "run_dir": run_dir,
        "chunk_manifest_path": chunk_manifest_path,
        "chunk_manifest": chunk_manifest,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 2 from the final demo pipeline or call run_step_02_chunk_planner(run_dir)."
    )


if __name__ == "__main__":
    main()
