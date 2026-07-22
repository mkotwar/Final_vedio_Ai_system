from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from .config import FlorenceConfig, PipelineConfig, VisionBackendConfig
from .crop_selection import SelectedCropJob
from .run_anpr_video_10fps_validation import (
    DEFAULT_FLORENCE_ADAPTER,
    DEFAULT_FLORENCE_MODEL,
    DEFAULT_PLATE_MODEL,
    DEFAULT_VEHICLE_MODEL,
    _cuda_available,
    _gpu_name,
    _resolve_optional_existing_path,
    _resolve_required_path,
    _resolve_vehicle_model,
)
from .run_combined_vehicle_person_pipeline import DEFAULT_PERSON_MODEL, DEFAULT_VIDEO
from .video_source import OpenCvVideoSource
from .vision_backends.gemini_backend import prepare_gemini_image
from .vision_backends.factory import create_vision_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight validation for the combined vehicle/person streaming pipeline.")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-root", default="debug_runs")
    parser.add_argument("--vehicle-detector-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--vehicle-detector-fallback-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--person-detector-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--plate-detector-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--florence-model", default=DEFAULT_FLORENCE_MODEL)
    parser.add_argument("--florence-adapter", default=DEFAULT_FLORENCE_ADAPTER)
    parser.add_argument("--vision-backend", default=None, choices=["auto", "florence", "gemini", "disabled"])
    parser.add_argument("--allow-real-api", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gemini-preflight-max-long-edge", type=int, default=512)
    parser.add_argument("--gemini-preflight-jpeg-quality", type=int, default=85)
    return parser


def _status(name: str, state: str, details: str, **extra: Any) -> dict[str, Any]:
    payload = {"name": name, "status": state, "details": details}
    payload.update(extra)
    return payload


def _import_check(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _build_representative_preflight_image(video_path: Path, output_dir: Path, *, max_long_edge: int, jpeg_quality: int) -> tuple[Path, dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read representative frame from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_path = output_dir / "gemini_preflight_probe.jpg"
    Image.fromarray(rgb).save(image_path, format="JPEG", quality=jpeg_quality, optimize=True)
    prepared = prepare_gemini_image(image_path, max_long_edge=max_long_edge, jpeg_quality=jpeg_quality)
    return image_path, {
        "image_width": prepared.width,
        "image_height": prepared.height,
        "image_bytes": prepared.image_bytes,
        "image_mime_type": prepared.mime_type,
        "resized": prepared.resized,
        "source_path": str(video_path),
        "prepared_path": str(image_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_config = PipelineConfig.from_env()
    requested_backend = (args.vision_backend or env_config.vision.backend_mode or "auto").strip().lower()
    vision_config = VisionBackendConfig(backend_mode=requested_backend)
    checks: list[dict[str, Any]] = []

    try:
        video_path = _resolve_required_path(args.video, "video")
        checks.append(_status("input_video", "pass", "Input video exists.", path=str(video_path)))
    except Exception as exc:
        video_path = None
        checks.append(_status("input_video", "fail", str(exc)))

    try:
        vehicle_model, vehicle_fallback_reason = _resolve_vehicle_model(args.vehicle_detector_model, args.vehicle_detector_fallback_model)
        checks.append(
            _status(
                "vehicle_model",
                "pass",
                "Vehicle detector model resolved.",
                path=str(vehicle_model),
                fallback_reason=vehicle_fallback_reason,
            )
        )
    except Exception as exc:
        vehicle_model = None
        checks.append(_status("vehicle_model", "fail", str(exc)))

    for name, label, raw_path in (
        ("person_model", "person detector model", args.person_detector_model),
        ("plate_model", "plate detector model", args.plate_detector_model),
    ):
        try:
            resolved = _resolve_required_path(raw_path, label)
            checks.append(_status(name, "pass", f"{label.capitalize()} exists.", path=str(resolved)))
        except Exception as exc:
            checks.append(_status(name, "fail", str(exc)))

    florence_model = _resolve_optional_existing_path(env_config.florence.base_model_path or args.florence_model)
    florence_adapter = _resolve_optional_existing_path(env_config.florence.adapter_path or args.florence_adapter)
    checks.append(
        _status(
            "florence_assets",
            "pass" if florence_model or requested_backend == "gemini" else "warning",
            "Florence assets resolved." if florence_model else "Florence assets not resolved locally.",
            model_path=str(florence_model) if florence_model else None,
            adapter_path=str(florence_adapter) if florence_adapter else None,
        )
    )

    required_imports = ["cv2", "torch", "ultralytics", "PIL"]
    optional_imports = ["transformers", "peft"]
    if requested_backend == "gemini":
        required_imports.append("google.genai")
    if env_config.tracking.backend == "supervision_bytetrack":
        required_imports.append("supervision")

    import_results = {}
    import_failures = []
    for module_name in required_imports + optional_imports:
        ok, detail = _import_check(module_name)
        import_results[module_name] = {"ok": ok, "detail": detail}
        if module_name in required_imports and not ok:
            import_failures.append(f"{module_name}: {detail}")
    checks.append(
        _status(
            "imports",
            "pass" if not import_failures else "fail",
            "All required imports resolved." if not import_failures else "; ".join(import_failures),
            modules=import_results,
        )
    )

    try:
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(_status("output_directory", "pass", "Output directory is writable.", path=str(output_root)))
    except Exception as exc:
        output_root = None
        checks.append(_status("output_directory", "fail", str(exc)))

    cuda_available = _cuda_available()
    gpu_name = _gpu_name()
    checks.append(
        _status(
            "compute",
            "warning" if not cuda_available else "pass",
            "CUDA unavailable; pipeline will rely on CPU fallback." if not cuda_available else "CUDA is available.",
            cuda_available=cuda_available,
            gpu_name=gpu_name,
        )
    )

    api_env_source = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else None
    api_key_present = bool(str(env_config.gemini.api_key or "").strip())
    checks.append(
        _status(
            "gemini_api_key",
            "pass" if api_key_present else "fail",
            "Gemini API key detected from environment." if api_key_present else "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY.",
            env_source=api_env_source,
            supported_priority=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        )
    )

    checks.append(
        _status(
            "vision_backend_selection",
            "pass" if requested_backend == "gemini" else "warning",
            "Gemini backend explicitly selected." if requested_backend == "gemini" else f"Selected backend mode is {requested_backend!r}, not 'gemini'.",
            selected_backend=requested_backend,
        )
    )

    if video_path is not None:
        try:
            source = OpenCvVideoSource(video_path, source_id=video_path.stem, use_source_fps=True, max_processed_frames=3)
            source.open()
            metadata = source.metadata_report()
            source.close()
            checks.append(_status("video_open", "pass", "Video opened successfully with OpenCV.", metadata=metadata))
        except Exception as exc:
            checks.append(_status("video_open", "fail", str(exc)))

    florence_config = FlorenceConfig(
        enabled=requested_backend != "disabled",
        base_model_path=str(florence_model) if florence_model is not None else None,
        adapter_path=str(florence_adapter) if florence_adapter is not None else None,
        device=env_config.florence.device,
        dtype=env_config.florence.dtype,
        local_files_only=env_config.florence.local_files_only,
        trust_remote_code=env_config.florence.trust_remote_code,
        load_in_4bit=False,
        load_in_8bit=False,
        max_new_tokens=env_config.florence.max_new_tokens,
        num_beams=env_config.florence.num_beams,
        do_sample=env_config.florence.do_sample,
        ocr_task_prompt=env_config.florence.ocr_task_prompt,
        colour_task_prompt=env_config.florence.colour_task_prompt,
    )

    gemini_probe_result: dict[str, Any] | None = None
    if requested_backend == "gemini" and api_key_present and args.allow_real_api:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if video_path is None:
                    raise RuntimeError("Input video path is unavailable for Gemini preflight.")
                image_path, image_details = _build_representative_preflight_image(
                    video_path,
                    Path(temp_dir),
                    max_long_edge=args.gemini_preflight_max_long_edge,
                    jpeg_quality=args.gemini_preflight_jpeg_quality,
                )
                backend = create_vision_backend(
                    vision_config=vision_config,
                    florence_config=florence_config,
                    gemini_config=env_config.gemini,
                    run_dir=temp_dir,
                )
                job = SelectedCropJob(
                    source_id="preflight",
                    track_id=1,
                    track_generation=0,
                    source_track_id="preflight_1",
                    object_class="car",
                    lifecycle_completion_reason="preflight",
                    crop_role="primary",
                    crop_rank=1,
                    frame_index=1,
                    timestamp_sec=0.0,
                    vehicle_crop_path=str(image_path),
                    full_frame_path=None,
                    selection_score=1.0,
                )
                colour_result = backend.run_colour(job)
                gemini_probe_result = {
                    "backend_name": backend.backend_name,
                    "status": colour_result.status,
                    "metadata": colour_result.metadata,
                    "metrics": backend.metrics,
                    "image_details": image_details,
                }
                florence_attempts = int(backend.metrics.get("florence_load_attempts", 0) or 0)
                checks.append(
                    _status(
                        "gemini_client_and_request",
                        "pass" if colour_result.status in {"success", "empty_output"} and florence_attempts == 0 else "fail",
                        "Gemini initialized and completed a minimal request." if colour_result.status in {"success", "empty_output"} and florence_attempts == 0 else "Gemini request failed or Florence was touched.",
                        result=gemini_probe_result,
                    )
                )
        except Exception as exc:
            checks.append(_status("gemini_client_and_request", "fail", str(exc)))
    elif requested_backend == "gemini":
        checks.append(
            _status(
                "gemini_client_and_request",
                "fail",
                "Skipped real Gemini probe because the API key is unavailable." if not api_key_present else "Skipped real Gemini probe because --no-allow-real-api was used.",
            )
        )

    checks.append(
        _status(
            "expensive_stages",
            "warning",
            "YOLO detection, ByteTrack, plate detection, and Gemini API calls are the main expensive stages; CPU-only execution will be substantially slower.",
        )
    )

    failed = [item for item in checks if item["status"] == "fail"]
    overall = "passed" if not failed else "failed"
    return {
        "status": overall,
        "requested_backend": requested_backend,
        "api_key_env_priority": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "checks": checks,
        "gemini_probe_result": gemini_probe_result,
    }


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
