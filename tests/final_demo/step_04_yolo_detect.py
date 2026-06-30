from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import write_json
from tests.final_demo.services.yolo_detector import (
    run_yolo_detection_on_sampled_frames,
    update_run_manifest_for_yolo,
)


def run_step_04_yolo_detect(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 4: YOLO detection per chunk")

    frames_index_path = run_dir / "03_sampled_frames_index.json"
    chunk_manifest_path = run_dir / "02_chunk_manifest.json"

    if not frames_index_path.exists():
        raise FileNotFoundError(f"Missing Step 3 sampled frames index file: {frames_index_path}")
    if not chunk_manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 2 chunk manifest file: {chunk_manifest_path}")

    frames_index_payload = read_json(frames_index_path)
    chunk_manifest = read_json(chunk_manifest_path)

    yolo_result = run_yolo_detection_on_sampled_frames(
        run_dir=run_dir,
        frames_index_payload=frames_index_payload,
        chunk_manifest=chunk_manifest,
    )

    detections_path = run_dir / "04_yolo_detections.json"
    report_path = run_dir / "04_yolo_detection_report.json"

    write_json(chunk_manifest_path, yolo_result["updated_chunk_manifest"])
    write_json(detections_path, yolo_result["detections_payload"])
    write_json(report_path, yolo_result["report_payload"])
    update_run_manifest_for_yolo(run_dir / "run_manifest.json")

    detections_payload = yolo_result["detections_payload"]
    print(f"[final-demo] Model name: {detections_payload['model_name']}")
    print(f"[final-demo] Confidence threshold: {detections_payload['confidence_threshold']}")
    print(f"[final-demo] Frames processed: {detections_payload['total_frames_processed']}")
    print(f"[final-demo] Detections found: {detections_payload['total_detections']}")
    print(f"[final-demo] Output path: {detections_path}")

    return {
        "run_dir": run_dir,
        "detections_path": detections_path,
        "report_path": report_path,
        "yolo_result": yolo_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 4 from the final demo pipeline or call run_step_04_yolo_detect(run_dir)."
    )


if __name__ == "__main__":
    main()
