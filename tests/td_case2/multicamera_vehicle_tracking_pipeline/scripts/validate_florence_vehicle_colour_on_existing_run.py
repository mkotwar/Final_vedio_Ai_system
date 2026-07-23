from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..enrichment.florence_config import load_florence_config
from ..enrichment.florence_vehicle_colour_extractor import FlorenceVehicleColourExtractor
from ..enrichment.media_resolver import resolve_local_media_path
from ..enrichment.vehicle_colour_config import load_vehicle_colour_config
from ..models.florence_runtime_factory import FlorenceRuntimeFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Florence vehicle-colour extraction on an existing run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--camera-code", default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--florence-config", required=True)
    parser.add_argument("--vehicle-colour-config", required=True)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    florence_config = load_florence_config(args.florence_config, overrides={"enabled": True})
    vehicle_colour_config = load_vehicle_colour_config(args.vehicle_colour_config, overrides={"enabled": True})
    runtime_factory = FlorenceRuntimeFactory(project_root=Path.cwd())
    started = time.perf_counter()
    runtime = runtime_factory.get_runtime(
        config=florence_config,
        model_path_cli=args.florence_model_path,
        adapter_path_cli=args.florence_adapter_path,
        processor_path_cli=args.florence_processor_path,
        device_override=args.device,
    )
    model_load_seconds = time.perf_counter() - started
    if runtime is None:
        raise RuntimeError("Florence runtime could not be created.")
    extractor = FlorenceVehicleColourExtractor(
        runtime=runtime,
        prompt=florence_config.colour_prompt,
        allowed_colours=vehicle_colour_config.allowed_colours,
        minimum_confidence=vehicle_colour_config.minimum_confidence,
    )
    run_root = artifact_root / args.run_id
    candidates = sorted(run_root.glob("**/best_overall.jpg"))
    if args.camera_code:
        candidates = [path for path in candidates if args.camera_code in path.parts]
    if args.max_tracks is not None:
        candidates = candidates[: int(args.max_tracks)]
    results: list[dict[str, object]] = []
    inference_times: list[float] = []
    for image_path in candidates:
        storage_uri = image_path.resolve().relative_to(artifact_root).as_posix()
        resolved = resolve_local_media_path(storage_uri=storage_uri, artifact_root=artifact_root)
        inference_started = time.perf_counter()
        colour_result = extractor.extract(
            resolved,
            track_uuid=_infer_track_uuid_from_path(image_path),
            camera_code=_infer_camera_code_from_path(image_path),
            source_storage_uri=storage_uri,
        )
        inference_times.append(time.perf_counter() - inference_started)
        results.append(
            {
                "image_path": str(resolved),
                "storage_uri": storage_uri,
                "track_uuid": _infer_track_uuid_from_path(image_path),
                "camera_code": _infer_camera_code_from_path(image_path),
                "vehicle_colour": colour_result.to_report_payload(),
            }
        )
    report = {
        "run_id": args.run_id,
        "artifact_root": str(artifact_root),
        "tracks_considered": len(candidates),
        "model_load_seconds": model_load_seconds,
        "average_inference_seconds": (sum(inference_times) / len(inference_times)) if inference_times else 0.0,
        "maximum_inference_seconds": max(inference_times) if inference_times else 0.0,
        "results": results,
    }
    output_path = Path(args.output_report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _infer_camera_code_from_path(image_path: Path) -> str:
    for part in image_path.parts:
        if part.startswith("CAM_"):
            return part
    return "UNKNOWN_CAMERA"


def _infer_track_uuid_from_path(image_path: Path) -> str:
    if len(image_path.parts) >= 2:
        return image_path.parent.name.replace("_TRACK_", ":TRACK_").replace("_CAM_", ":CAM_").replace("RUN_", "RUN_")
    return image_path.stem


if __name__ == "__main__":
    main()
