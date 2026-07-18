"""CPU-only Step 2 validation runner for sequential contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import JsonlPacketSink
from .config import PipelineConfig
from .mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from .schemas import BoundingBox, DetectionRecord, TrackedObject
from .sequential_pipeline import SequentialContractPipeline
from .serialization import read_jsonl, write_json
from .sources import SyntheticFrameSource


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 2 streaming contract validation.")
    parser.add_argument("--output-dir", default=None)
    return parser


def run_validation(output_dir: str | Path | None = None) -> dict[str, object]:
    config = PipelineConfig.from_env()
    output_root = Path(output_dir or Path(config.output.output_root) / "step2_contract_validation")
    output_root.mkdir(parents=True, exist_ok=True)

    source = SyntheticFrameSource(
        source_id="step2_synthetic_cam",
        total_frames=30,
        source_fps=30.0,
        frame_width=640,
        frame_height=360,
        target_processing_fps=7.0,
    )
    selected_indices = source.selected_frame_indices
    detections_by_frame = {
        selected_indices[0]: [DetectionRecord(BoundingBox(10, 20, 80, 90), 0.9, 2, "car")],
        selected_indices[1]: [DetectionRecord(BoundingBox(20, 20, 90, 90), 0.8, 2, "car")],
    }

    def track_factory(packet):
        detections = detections_by_frame.get(packet.frame_index, [])
        if not detections:
            return []
        detection = detections[0]
        return [
            TrackedObject(
                track_id=1,
                source_track_id="vehicle_track_0001",
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                frame_index=packet.frame_index,
                timestamp_sec=packet.timestamp_sec,
            )
        ]

    detection_stage = DeterministicMockDetectionStage(detections_by_frame=detections_by_frame)
    tracking_stage = DeterministicMockTrackingStage(track_factory=track_factory)
    with JsonlPacketSink(output_root) as sink:
        report = SequentialContractPipeline(
            source=source,
            detection_stage=detection_stage,
            tracking_stage=tracking_stage,
            sink=sink,
        ).run()

    payload = report.to_dict()
    payload["selected_frame_indices"] = list(selected_indices)
    payload["artifact_files"] = {
        "frames": str(output_root / "frame_packets.jsonl"),
        "detections": str(output_root / "detection_packets.jsonl"),
        "tracked_frames": str(output_root / "tracked_frame_packets.jsonl"),
        "summary": str(output_root / "step2_contract_validation_report.json"),
    }
    frame_records = read_jsonl(output_root / "frame_packets.jsonl")
    detection_records = read_jsonl(output_root / "detection_packets.jsonl")
    tracked_records = read_jsonl(output_root / "tracked_frame_packets.jsonl")
    if len(frame_records) != report.selected_frames_processed:
        raise RuntimeError("Frame JSONL count did not match report.")
    if len(detection_records) != report.detection_packets_created:
        raise RuntimeError("Detection JSONL count did not match report.")
    if len(tracked_records) != report.tracked_packets_created:
        raise RuntimeError("Tracked JSONL count did not match report.")
    write_json(output_root / "step2_contract_validation_report.json", payload)
    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_validation(args.output_dir)
    print(
        "Step 2 contract validation passed: "
        f"{payload['selected_frames_processed']} frames, "
        f"{payload['detection_packets_created']} detection packets, "
        f"{payload['tracked_packets_created']} tracked packets."
    )
    print(f"Artifacts: {payload['artifact_files']['summary']}")


if __name__ == "__main__":
    main()
