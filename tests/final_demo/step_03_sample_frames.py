from __future__ import annotations

from pathlib import Path

from tests.final_demo.services.chunk_planner import read_json, write_json
from tests.final_demo.services.frame_sampler import (
    read_max_chunks,
    read_sample_fps,
    sample_frames_for_chunks,
    update_run_manifest_for_sampling,
)


def run_step_03_sample_frames(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 3: frame sampling per chunk")

    video_info_path = run_dir / "01_video_info.json"
    chunk_manifest_path = run_dir / "02_chunk_manifest.json"

    if not video_info_path.exists():
        raise FileNotFoundError(f"Missing Step 1 video info file: {video_info_path}")
    if not chunk_manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 2 chunk manifest file: {chunk_manifest_path}")

    video_info = read_json(video_info_path)
    chunk_manifest = read_json(chunk_manifest_path)

    sample_fps = read_sample_fps()
    max_chunks = read_max_chunks()

    sampling_result = sample_frames_for_chunks(
        run_dir=run_dir,
        video_info=video_info,
        chunk_manifest=chunk_manifest,
        sample_fps=sample_fps,
        max_chunks=max_chunks,
    )

    updated_chunk_manifest = sampling_result["updated_chunk_manifest"]
    index_payload = sampling_result["index_payload"]
    report_payload = sampling_result["report_payload"]

    sampled_frames_index_path = run_dir / "03_sampled_frames_index.json"
    sampling_report_path = run_dir / "03_sampling_report.json"

    write_json(chunk_manifest_path, updated_chunk_manifest)
    write_json(sampled_frames_index_path, index_payload)
    write_json(sampling_report_path, report_payload)
    update_run_manifest_for_sampling(run_dir / "run_manifest.json")

    print(f"[final-demo] Sample FPS: {index_payload['sample_fps']}")
    print(f"[final-demo] Chunks processed: {index_payload['chunks_processed']}")
    print(f"[final-demo] Total sampled frames: {index_payload['total_frames_sampled']}")
    print(f"[final-demo] Output index path: {sampled_frames_index_path}")

    return {
        "run_dir": run_dir,
        "sampled_frames_index_path": sampled_frames_index_path,
        "sampling_report_path": sampling_report_path,
        "sampling_result": sampling_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 3 from the final demo pipeline or call run_step_03_sample_frames(run_dir)."
    )


if __name__ == "__main__":
    main()
