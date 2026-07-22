from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import re
import time
from dataclasses import dataclass
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
OCR_PROMPT = "Return JSON only with keys raw_text, confidence, notes for the vehicle number plate visible in this image."
COLOUR_PROMPT = "Return JSON only with keys raw_text, normalized_colour, confidence, notes for the primary vehicle body colour in this image."
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
SAFE_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass(frozen=True)
class PreparedGeminiImage:
    part: Any
    width: int
    height: int
    image_bytes: int
    mime_type: str
    resized: bool
    source_path: str


class GeminiBackendRequestError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def prepare_gemini_image(
    image_path: str | Path,
    *,
    max_long_edge: int | None = None,
    jpeg_quality: int = 85,
) -> PreparedGeminiImage:
    from google.genai import types  # type: ignore

    path = Path(image_path).expanduser()
    with Image.open(path) as image:
        image.load()
        original_width, original_height = image.size
        resized = False
        mime_type = (mimetypes.guess_type(path.name)[0] or "").lower()
        if max_long_edge is not None and max(original_width, original_height) > max_long_edge:
            resized = True
            working = image.convert("RGB")
            if original_width >= original_height:
                new_width = max_long_edge
                new_height = max(1, round(original_height * (max_long_edge / original_width)))
            else:
                new_height = max_long_edge
                new_width = max(1, round(original_width * (max_long_edge / original_height)))
            working = working.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            payload = buffer.getvalue()
            mime_type = "image/jpeg"
            final_width, final_height = working.size
        elif mime_type in SAFE_IMAGE_MIME_TYPES:
            payload = path.read_bytes()
            final_width, final_height = original_width, original_height
        else:
            working = image.convert("RGB")
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            payload = buffer.getvalue()
            mime_type = "image/jpeg"
            final_width, final_height = working.size
        part = types.Part.from_bytes(data=payload, mime_type=mime_type or "image/jpeg")
    return PreparedGeminiImage(
        part=part,
        width=int(final_width),
        height=int(final_height),
        image_bytes=len(payload),
        mime_type=mime_type or "image/jpeg",
        resized=resized,
        source_path=str(path),
    )


def _status_code_from_exception(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _sanitize_error_message(message: str, api_key: str | None) -> str:
    sanitized = str(message or "")
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    sanitized = re.sub(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?bearer\s+)[^'\",\s]+", r"\1[REDACTED]", sanitized)
    sanitized = re.sub(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\",\s]+", r"\1[REDACTED]", sanitized)
    return sanitized[:500]


def _exception_chain(exc: Exception, api_key: str | None) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": _sanitize_error_message(str(current), api_key),
            }
        )
        next_exc = getattr(current, "__cause__", None)
        if next_exc is None:
            next_exc = getattr(current, "__context__", None)
        current = next_exc if isinstance(next_exc, Exception) else None
    return chain


def _error_category(exc: Exception) -> str:
    try:
        import httpx  # type: ignore
    except Exception:  # pragma: no cover
        httpx = None

    status_code = _status_code_from_exception(exc)
    if httpx is not None and isinstance(exc, httpx.ReadTimeout):
        return "timeout"
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if httpx is not None and isinstance(exc, httpx.NetworkError):
        return "network"
    if status_code == 401:
        return "authentication"
    if status_code == 403:
        return "permission_denied"
    if status_code == 404:
        return "invalid_model"
    if status_code == 429:
        return "rate_limited"
    if status_code in {400, 422}:
        return "malformed_request"
    if status_code in {500, 502, 503, 504}:
        return "server_error"
    if isinstance(exc, ValueError):
        return "schema_validation"
    return "unexpected"


def _is_retryable_exception(exc: Exception) -> bool:
    try:
        import httpx  # type: ignore
    except Exception:  # pragma: no cover
        httpx = None

    if httpx is not None and isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return True
    status_code = _status_code_from_exception(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return False


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
        timeout_seconds = max(1, int(getattr(self.config, "timeout_seconds", 90) or 90))
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        return self._client

    def run_ocr(self, candidate: Any) -> FlorenceOcrResult:
        prompt = OCR_PROMPT
        image_path = getattr(candidate, "plate_crop_path", None)
        if not image_path:
            return self._ocr_result(candidate, "", "input_missing", prompt, {"vision_backend": "gemini"})
        resolved = resolve_image_path(str(image_path), run_dir=self.run_dir)
        if not resolved.exists():
            return self._ocr_result(candidate, "", "input_missing", prompt, {"vision_backend": "gemini"})
        try:
            prepared = prepare_gemini_image(resolved)
            payload, latency_sec, cache_hit = self._request_json(
                prepared_image=prepared,
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
            "image_width": prepared.width,
            "image_height": prepared.height,
            "image_bytes": prepared.image_bytes,
            "image_mime_type": prepared.mime_type,
        }
        return self._ocr_result(candidate, raw_text, status, prompt, metadata)

    def run_colour(self, job: SelectedCropJob) -> FlorenceColourResult:
        prompt = COLOUR_PROMPT
        resolved = resolve_image_path(job.vehicle_crop_path, run_dir=self.run_dir)
        if not resolved.exists():
            return self._colour_result(job, "", "input_missing", prompt, {"vision_backend": "gemini"})
        try:
            prepared = prepare_gemini_image(resolved)
            payload, latency_sec, cache_hit = self._request_json(
                prepared_image=prepared,
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
            "image_width": prepared.width,
            "image_height": prepared.height,
            "image_bytes": prepared.image_bytes,
            "image_mime_type": prepared.mime_type,
        }
        return self._colour_result(job, raw_text or normalized_colour, status, prompt, metadata)

    def _request_json(
        self,
        *,
        prepared_image: PreparedGeminiImage,
        task_name: str,
        schema_version: str,
        prompt: str,
        response_schema: Any,
    ) -> tuple[dict[str, Any], float, bool]:
        cache_key = self._cache_key(Path(prepared_image.source_path), task_name, schema_version)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            self._metrics["gemini_cache_hits"] += 1
            return dict(cached), 0.0, True
        client = self.load()
        started_total = time.perf_counter()
        max_attempts = int(getattr(self.config, "max_retries", 1) or 1) + 1
        backoff_seconds = float(getattr(self.config, "retry_backoff_seconds", 2.0) or 0.0)
        last_exc: Exception | None = None
        last_metadata: dict[str, Any] | None = None
        for attempt_number in range(1, max_attempts + 1):
            self._metrics["gemini_requests"] += 1
            if attempt_number > 1:
                self._metrics["gemini_retries"] += 1
            attempt_started = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=str(getattr(self.config, "model_name", "") or "gemini-2.5-flash"),
                    contents=[prepared_image.part, prompt],
                    config=self._generate_config(response_schema),
                )
                payload = self._response_to_dict(response)
                elapsed_sec = round(time.perf_counter() - started_total, 6)
                self._metrics["gemini_total_latency_sec"] += elapsed_sec
                self._remember(cache_key, payload)
                return payload, elapsed_sec, False
            except Exception as exc:
                last_exc = exc
                attempt_elapsed = round(time.perf_counter() - attempt_started, 6)
                total_elapsed = round(time.perf_counter() - started_total, 6)
                last_metadata = {
                    "vision_backend": "gemini",
                    "backend_model": getattr(self.config, "model_name", None),
                    "error_type": type(exc).__name__,
                    "error_message": _sanitize_error_message(str(exc), str(getattr(self.config, "api_key", "") or "")),
                    "error_category": _error_category(exc),
                    "attempt_number": attempt_number,
                    "max_attempts": max_attempts,
                    "elapsed_sec": total_elapsed,
                    "attempt_elapsed_sec": attempt_elapsed,
                    "model": getattr(self.config, "model_name", None),
                    "image_width": prepared_image.width,
                    "image_height": prepared_image.height,
                    "image_bytes": prepared_image.image_bytes,
                    "image_mime_type": prepared_image.mime_type,
                    "prompt_chars": len(prompt),
                    "schema_enabled": response_schema is not None,
                    "exception_chain": _exception_chain(exc, str(getattr(self.config, "api_key", "") or "")),
                }
                if not _is_retryable_exception(exc) or attempt_number >= max_attempts:
                    break
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds)
        elapsed_sec = round(time.perf_counter() - started_total, 6)
        self._metrics["gemini_total_latency_sec"] += elapsed_sec
        if last_metadata is None:
            last_metadata = {
                "vision_backend": "gemini",
                "backend_model": getattr(self.config, "model_name", None),
                "error_type": "RuntimeError",
                "error_message": "Gemini request failed without an exception.",
                "error_category": "unexpected",
                "attempt_number": 0,
                "max_attempts": max_attempts,
                "elapsed_sec": elapsed_sec,
                "model": getattr(self.config, "model_name", None),
                "image_width": prepared_image.width,
                "image_height": prepared_image.height,
                "image_bytes": prepared_image.image_bytes,
                "image_mime_type": prepared_image.mime_type,
                "prompt_chars": len(prompt),
                "schema_enabled": response_schema is not None,
                "exception_chain": [],
            }
        last_metadata["elapsed_sec"] = elapsed_sec
        raise GeminiBackendRequestError("Gemini request failed.", metadata=last_metadata) from last_exc

    def _generate_config(self, response_schema: Any) -> Any:
        from google import genai  # type: ignore

        return genai.types.GenerateContentConfig(
            temperature=0.0,
            candidate_count=1,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                return dict(parsed.model_dump(exclude_none=True))
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
        if isinstance(exc, GeminiBackendRequestError):
            return dict(exc.metadata)
        return {
            "vision_backend": "gemini",
            "backend_model": getattr(self.config, "model_name", None),
            "error_type": type(exc).__name__,
            "error_message": _sanitize_error_message(str(exc), str(getattr(self.config, "api_key", "") or "")),
            "error_category": _error_category(exc),
            "exception_chain": _exception_chain(exc, str(getattr(self.config, "api_key", "") or "")),
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
