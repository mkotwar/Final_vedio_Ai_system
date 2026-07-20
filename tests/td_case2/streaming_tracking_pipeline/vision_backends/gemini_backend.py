from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..anpr_schemas import FlorenceColourResult, FlorenceOcrResult, normalize_raw_plate_text, normalize_vehicle_colour
from ..crop_selection import SelectedCropJob
from ..plate_detection import resolve_image_path
from .base import VisionInferenceBackend
from .schemas import GeminiStructuredColour, GeminiStructuredOcr


OCR_SCHEMA_VERSION = "gemini_ocr_v1"
COLOUR_SCHEMA_VERSION = "gemini_colour_v1"


class GeminiVisionBackend(VisionInferenceBackend):
    def __init__(
        self,
        *,
        config: Any,
        run_dir: str | Path | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._client_factory = client_factory
        self._client: Any = None
        self._content_hash_cache: dict[str, str] = {}
        self._response_cache: dict[str, dict[str, Any]] = {}
        self._cache_limit = 256
        self._metrics = {
            "gemini_load_attempts": 0,
            "gemini_load_errors": 0,
            "gemini_requests": 0,
            "gemini_retries": 0,
            "gemini_cache_hits": 0,
            "gemini_failures": 0,
            "gemini_ocr_successes": 0,
            "gemini_colour_successes": 0,
            "gemini_total_latency_sec": 0.0,
        }

    @property
    def backend_name(self) -> str:
        return "gemini"

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    def load(self) -> Any:
        if self._client is not None:
            return self._client
        self._metrics["gemini_load_attempts"] += 1
        if not getattr(self.config, "enabled", True):
            raise RuntimeError("Gemini backend is disabled.")
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        api_key = str(getattr(self.config, "api_key", "") or "").strip()
        if not api_key:
            self._metrics["gemini_load_errors"] += 1
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini backend.")
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            self._metrics["gemini_load_errors"] += 1
            raise RuntimeError("google-genai is required for the Gemini backend.") from exc
        timeout_seconds = max(1, int(getattr(self.config, "timeout_seconds", 60) or 60))
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(timeout=timeout_seconds),
        )
        return self._client

    def run_ocr(self, candidate: Any) -> FlorenceOcrResult:
        prompt = "Return JSON only with keys raw_text, confidence, notes for the vehicle number plate visible in this image."
        image_path = getattr(candidate, "plate_crop_path", None)
        if not image_path:
            return self._ocr_result(candidate, "", "input_missing", prompt, {"vision_backend": "gemini"})
        resolved = resolve_image_path(str(image_path), run_dir=self.run_dir)
        if not resolved.exists():
            return self._ocr_result(candidate, "", "input_missing", prompt, {"vision_backend": "gemini"})
        try:
            payload, latency_sec, cache_hit = self._request_json(
                image_path=resolved,
                task_name="ocr",
                schema_version=OCR_SCHEMA_VERSION,
                prompt=prompt,
                response_schema=GeminiStructuredOcr,
            )
        except Exception as exc:
            self._metrics["gemini_failures"] += 1
            return self._ocr_result(candidate, "", "inference_error", prompt, self._safe_error_metadata(exc))
        raw_text = str(payload.get("raw_text", "") or "")
        normalized = normalize_raw_plate_text(raw_text)
        status = "success" if normalized else "empty_output"
        if status == "success":
            self._metrics["gemini_ocr_successes"] += 1
        metadata = {
            "vision_backend": "gemini",
            "backend_model": getattr(self.config, "model_name", None),
            "gemini_confidence": float(payload.get("confidence", 0.0) or 0.0),
            "gemini_notes": str(payload.get("notes", "") or ""),
            "latency_sec": latency_sec,
            "cache_hit": cache_hit,
        }
        return self._ocr_result(candidate, raw_text, status, prompt, metadata)

    def run_colour(self, job: SelectedCropJob) -> FlorenceColourResult:
        prompt = "Return JSON only with keys raw_text, normalized_colour, confidence, notes for the primary vehicle body colour in this image."
        resolved = resolve_image_path(job.vehicle_crop_path, run_dir=self.run_dir)
        if not resolved.exists():
            return self._colour_result(job, "", "input_missing", prompt, {"vision_backend": "gemini"})
        try:
            payload, latency_sec, cache_hit = self._request_json(
                image_path=resolved,
                task_name="colour",
                schema_version=COLOUR_SCHEMA_VERSION,
                prompt=prompt,
                response_schema=GeminiStructuredColour,
            )
        except Exception as exc:
            self._metrics["gemini_failures"] += 1
            return self._colour_result(job, "", "inference_error", prompt, self._safe_error_metadata(exc))
        raw_text = str(payload.get("raw_text", "") or "")
        normalized_colour = normalize_vehicle_colour(str(payload.get("normalized_colour", "") or raw_text))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        minimum_confidence = float(getattr(self.config, "min_confidence", 0.75) or 0.75)
        status = "success" if normalized_colour != "unknown" and confidence >= minimum_confidence else "empty_output"
        if status == "success":
            self._metrics["gemini_colour_successes"] += 1
        metadata = {
            "vision_backend": "gemini",
            "backend_model": getattr(self.config, "model_name", None),
            "gemini_confidence": confidence,
            "gemini_notes": str(payload.get("notes", "") or ""),
            "latency_sec": latency_sec,
            "cache_hit": cache_hit,
        }
        return self._colour_result(job, raw_text or normalized_colour, status, prompt, metadata)

    def _request_json(
        self,
        *,
        image_path: Path,
        task_name: str,
        schema_version: str,
        prompt: str,
        response_schema: Any,
    ) -> tuple[dict[str, Any], float, bool]:
        cache_key = self._cache_key(image_path, task_name, schema_version)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            self._metrics["gemini_cache_hits"] += 1
            return dict(cached), 0.0, True
        client = self.load()
        started = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(int(getattr(self.config, "max_retries", 2) or 2) + 1):
            self._metrics["gemini_requests"] += 1
            if attempt:
                self._metrics["gemini_retries"] += 1
            try:
                response = client.models.generate_content(
                    model=str(getattr(self.config, "model_name", "") or "gemini-2.5-flash"),
                    contents=[Image.open(image_path).convert("RGB"), prompt],
                    config=self._generate_config(response_schema),
                )
                payload = self._response_to_dict(response)
                latency_sec = round(time.perf_counter() - started, 6)
                self._metrics["gemini_total_latency_sec"] += latency_sec
                self._remember(cache_key, payload)
                return payload, latency_sec, False
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def _generate_config(self, response_schema: Any) -> Any:
        from google import genai  # type: ignore

        return genai.types.GenerateContentConfig(
            temperature=0.0,
            candidateCount=1,
            responseMimeType="application/json",
            responseSchema=response_schema,
        )

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "__dict__"):
                return dict(parsed.__dict__)
            if isinstance(parsed, dict):
                return dict(parsed)
        text = getattr(response, "text", "") or ""
        if not text:
            return {}
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Gemini response was not a JSON object.")
        return value

    def _cache_key(self, image_path: Path, task_name: str, schema_version: str) -> str:
        content_hash = self._content_hash(image_path)
        key = "|".join(
            [
                task_name,
                schema_version,
                str(getattr(self.config, "model_name", "") or ""),
                self.backend_name,
                content_hash,
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _content_hash(self, image_path: Path) -> str:
        cache_key = str(image_path.resolve())
        cached = self._content_hash_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        self._content_hash_cache[cache_key] = digest
        return digest

    def _remember(self, cache_key: str, payload: dict[str, Any]) -> None:
        if len(self._response_cache) >= self._cache_limit:
            first_key = next(iter(self._response_cache))
            self._response_cache.pop(first_key, None)
        self._response_cache[cache_key] = dict(payload)

    def _safe_error_metadata(self, exc: Exception) -> dict[str, Any]:
        return {
            "vision_backend": "gemini",
            "backend_model": getattr(self.config, "model_name", None),
            "error_type": type(exc).__name__,
        }

    def _ocr_result(self, candidate: Any, raw_text: str, status: str, prompt: str, metadata: dict[str, Any]) -> FlorenceOcrResult:
        return FlorenceOcrResult(
            source_id=candidate.source_id,
            track_id=candidate.track_id,
            track_generation=candidate.track_generation,
            crop_role=candidate.crop_role,
            crop_rank=candidate.crop_rank,
            frame_index=candidate.frame_index,
            plate_rank=candidate.plate_rank,
            plate_crop_path=candidate.plate_crop_path,
            raw_text=raw_text,
            normalized_text=normalize_raw_plate_text(raw_text),
            status=status,
            prompt=prompt,
            metadata=metadata,
        )

    def _colour_result(self, job: SelectedCropJob, raw_text: str, status: str, prompt: str, metadata: dict[str, Any]) -> FlorenceColourResult:
        return FlorenceColourResult(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            crop_role=job.crop_role,
            crop_rank=job.crop_rank,
            frame_index=job.frame_index,
            vehicle_crop_path=job.vehicle_crop_path,
            raw_text=raw_text,
            normalized_colour=normalize_vehicle_colour(raw_text),
            status=status,
            prompt=prompt,
            metadata=metadata,
        )
