from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
from google import genai  # type: ignore

from .config import PipelineConfig
from .vision_backends.gemini_backend import COLOUR_PROMPT, PreparedGeminiImage, prepare_gemini_image, _exception_chain, _sanitize_error_message
from .vision_backends.schemas import GeminiStructuredColour


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated Gemini text/image probes against the installed SDK.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-long-edge", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser


def _extract_probe_image(video_path: Path, temp_dir: Path, *, max_long_edge: int, jpeg_quality: int) -> PreparedGeminiImage:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read representative frame from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    probe_path = temp_dir / "gemini_probe_frame.jpg"
    from PIL import Image

    Image.fromarray(rgb).save(probe_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return prepare_gemini_image(probe_path, max_long_edge=max_long_edge, jpeg_quality=jpeg_quality)


def _run_probe(
    *,
    client: Any,
    model: str,
    name: str,
    contents: Any,
    config: Any = None,
    prepared_image: PreparedGeminiImage | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
        elapsed = round(time.perf_counter() - started, 6)
        return {
            "probe": name,
            "status": "pass",
            "elapsed_sec": elapsed,
            "model": model,
            "image_width": prepared_image.width if prepared_image is not None else None,
            "image_height": prepared_image.height if prepared_image is not None else None,
            "image_bytes": prepared_image.image_bytes if prepared_image is not None else None,
            "image_mime_type": prepared_image.mime_type if prepared_image is not None else None,
            "exception_class": None,
            "exception_message": None,
            "exception_chain": [],
            "response_text_preview": (getattr(response, "text", "") or "")[:200],
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 6)
        return {
            "probe": name,
            "status": "fail",
            "elapsed_sec": elapsed,
            "model": model,
            "image_width": prepared_image.width if prepared_image is not None else None,
            "image_height": prepared_image.height if prepared_image is not None else None,
            "image_bytes": prepared_image.image_bytes if prepared_image is not None else None,
            "image_mime_type": prepared_image.mime_type if prepared_image is not None else None,
            "exception_class": type(exc).__name__,
            "exception_message": _sanitize_error_message(str(exc), None),
            "exception_chain": _exception_chain(exc, None),
            "response_text_preview": None,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_config = PipelineConfig.from_env()
    api_key = str(env_config.gemini.api_key or "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key missing from environment.")
    model = str(args.model or env_config.gemini.model_name or "gemini-2.5-flash")
    client = genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(timeout=int(env_config.gemini.timeout_seconds) * 1000),
    )
    with tempfile.TemporaryDirectory() as directory:
        prepared_image = _extract_probe_image(Path(args.video), Path(directory), max_long_edge=args.max_long_edge, jpeg_quality=args.jpeg_quality)
        simple_json_prompt = 'Reply with JSON only: {"label":"ok","notes":"one short sentence description"}'
        schema_config = genai.types.GenerateContentConfig(
            temperature=0.0,
            candidate_count=1,
            response_mime_type="application/json",
            response_schema=GeminiStructuredColour,
        )
        simple_json_config = genai.types.GenerateContentConfig(
            temperature=0.0,
            candidate_count=1,
            response_mime_type="application/json",
        )
        probes = [
            _run_probe(client=client, model=model, name="A_text_only", contents="Reply with only the word OK"),
            _run_probe(client=client, model=model, name="B_image_simple_prompt", contents=[prepared_image.part, "Describe this image in one sentence."], prepared_image=prepared_image),
            _run_probe(client=client, model=model, name="C_image_json_no_schema", contents=[prepared_image.part, simple_json_prompt], config=simple_json_config, prepared_image=prepared_image),
            _run_probe(client=client, model=model, name="D_image_production_prompt_schema", contents=[prepared_image.part, COLOUR_PROMPT], config=schema_config, prepared_image=prepared_image),
        ]
    return {
        "status": "passed" if all(item["status"] == "pass" for item in probes) else "failed",
        "model": model,
        "image_width": prepared_image.width,
        "image_height": prepared_image.height,
        "image_bytes": prepared_image.image_bytes,
        "image_mime_type": prepared_image.mime_type,
        "production_prompt_chars": len(COLOUR_PROMPT),
        "production_schema": genai._transformers.t_schema(None, GeminiStructuredColour).model_dump(exclude_none=True),  # type: ignore[attr-defined]
        "production_schema_chars": len(json.dumps(genai._transformers.t_schema(None, GeminiStructuredColour).model_dump(exclude_none=True), separators=(",", ":"))),  # type: ignore[attr-defined]
        "probes": probes,
    }


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
