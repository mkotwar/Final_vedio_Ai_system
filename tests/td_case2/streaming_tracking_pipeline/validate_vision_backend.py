from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .config import FlorenceConfig, PipelineConfig
from .crop_selection import SelectedCropJob
from .plate_detection import resolve_image_path
from .run_anpr_video_10fps_validation import _resolve_optional_existing_path
from .vision_backends.factory import create_vision_backend


class _FakeGeminiResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.parsed = payload


class _FakeGeminiModels:
    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> _FakeGeminiResponse:
        prompt = str(contents[-1]) if isinstance(contents, list) and contents else ""
        if "normalized_colour" in prompt:
            return _FakeGeminiResponse({"raw_text": "white", "normalized_colour": "white", "confidence": 0.95, "notes": "mock"})
        return _FakeGeminiResponse({"raw_text": "MH12AB1234", "confidence": 0.92, "notes": "mock"})


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the isolated Step 7 vision backend configuration.")
    parser.add_argument("--allow-real-api", action="store_true", default=False)
    parser.add_argument("--florence-model", default=None)
    parser.add_argument("--florence-adapter", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_config = PipelineConfig.from_env()
    florence_model = _resolve_optional_existing_path(args.florence_model or env_config.florence.base_model_path)
    florence_adapter = _resolve_optional_existing_path(args.florence_adapter or env_config.florence.adapter_path)
    florence_config = FlorenceConfig(
        enabled=env_config.vision.backend_mode != "disabled",
        base_model_path=str(florence_model) if florence_model is not None else None,
        adapter_path=str(florence_adapter) if florence_adapter is not None else None,
        device=env_config.florence.device,
        dtype=env_config.florence.dtype,
        local_files_only=env_config.florence.local_files_only,
        trust_remote_code=env_config.florence.trust_remote_code,
        load_in_4bit=env_config.florence.load_in_4bit,
        load_in_8bit=env_config.florence.load_in_8bit,
        max_new_tokens=env_config.florence.max_new_tokens,
        num_beams=env_config.florence.num_beams,
        do_sample=env_config.florence.do_sample,
        ocr_task_prompt=env_config.florence.ocr_task_prompt,
        colour_task_prompt=env_config.florence.colour_task_prompt,
    )
    client_factory = None if args.allow_real_api else _FakeGeminiClient
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "vehicle.jpg"
        Image.new("RGB", (96, 64), "white").save(image_path)
        backend = create_vision_backend(
            vision_config=env_config.vision,
            florence_config=florence_config,
            gemini_config=env_config.gemini,
            run_dir=directory,
            gemini_client_factory=client_factory,
        )
        job = SelectedCropJob(
            source_id="validator",
            track_id=1,
            track_generation=0,
            source_track_id="validator_1",
            object_class="car",
            lifecycle_completion_reason="validation",
            crop_role="primary",
            crop_rank=1,
            frame_index=1,
            timestamp_sec=0.1,
            vehicle_crop_path=str(resolve_image_path(str(image_path), run_dir=directory)),
            full_frame_path=None,
            selection_score=1.0,
        )
        colour = backend.run_colour(job)
    return {
        "status": "passed" if colour.status in {"success", "empty_output", "model_disabled", "input_missing"} else "failed",
        "vision_backend_mode": env_config.vision.backend_mode,
        "selected_backend": colour.metadata.get("vision_backend"),
        "florence_model_resolved": str(florence_model) if florence_model is not None else None,
        "florence_adapter_resolved": str(florence_adapter) if florence_adapter is not None else None,
        "used_real_api": bool(args.allow_real_api),
        "colour_status": colour.status,
        "colour_metadata": colour.metadata,
        "metrics": backend.metrics,
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
