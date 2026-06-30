from __future__ import annotations

from typing import Any

from tests.final_demo.services.video_io import (
    create_run_directory,
    read_input_video_path,
    read_video_metadata,
    write_json,
)


def run_step_01_video_ingest() -> dict[str, Any]:
    print("[final-demo] Starting Step 1: video ingestion")
    video_path = read_input_video_path()
    print(f"[final-demo] Input video path: {video_path}")

    video_info = read_video_metadata(video_path)
    run_dir = create_run_directory(video_path)

    video_info_path = run_dir / "01_video_info.json"
    manifest_path = run_dir / "run_manifest.json"

    write_json(video_info_path, video_info)
    write_json(
        manifest_path,
        {
            "run_dir": str(run_dir),
            "input_video": str(video_path),
            "created_at": video_info["created_at"],
            "pipeline_name": "final_demo",
            "completed_steps": ["01_video_ingest"],
            "next_step": "02_chunk_planner",
        },
    )

    print(f"[final-demo] Duration: {video_info['duration_seconds']} seconds")
    print(f"[final-demo] FPS: {video_info['fps']}")
    print(f"[final-demo] Frame count: {video_info['frame_count']}")
    print(f"[final-demo] Resolution: {video_info['resolution']}")
    print(f"[final-demo] Run directory: {run_dir}")

    return {
        "run_dir": run_dir,
        "video_info_path": video_info_path,
        "manifest_path": manifest_path,
        "video_info": video_info,
    }


def main() -> None:
    run_step_01_video_ingest()


if __name__ == "__main__":
    main()
