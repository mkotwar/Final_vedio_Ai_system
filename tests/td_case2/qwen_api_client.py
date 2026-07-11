from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_QWEN_API_MAX_RETRIES,
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_QWEN_API_TIMEOUT_SECONDS,
    ENV_QWEN_API_KEY,
    ENV_QWEN_API_MAX_RETRIES,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_QWEN_API_TIMEOUT_SECONDS,
)


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class QwenApiConfig:
    provider: str
    model: str
    api_key: str
    timeout_seconds: int
    max_retries: int
    endpoint: str


def _read_positive_int(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def load_api_config_from_env() -> QwenApiConfig:
    provider = os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER).strip().lower() or DEFAULT_QWEN_API_PROVIDER
    model = os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL).strip() or DEFAULT_QWEN_API_MODEL
    api_key = os.environ.get(ENV_QWEN_API_KEY, "").strip()
    if not api_key:
        raise RuntimeError(f"{ENV_QWEN_API_KEY} is required when TD_CASE2_VLM_BACKEND=api_qwen.")
    if provider != "openrouter":
        raise RuntimeError(f"Unsupported Qwen API provider: {provider!r}. Supported: openrouter.")
    return QwenApiConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        timeout_seconds=_read_positive_int(ENV_QWEN_API_TIMEOUT_SECONDS, DEFAULT_QWEN_API_TIMEOUT_SECONDS),
        max_retries=_read_positive_int(ENV_QWEN_API_MAX_RETRIES, DEFAULT_QWEN_API_MAX_RETRIES),
        endpoint=OPENROUTER_ENDPOINT,
    )


def image_path_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if not mime:
        raise RuntimeError(f"Unsupported image type for API upload: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_openrouter_request(config: QwenApiConfig, prompt_text: str, image_path: Path) -> tuple[dict[str, str], bytes, dict[str, Any]]:
    data_url = image_path_to_data_url(image_path)
    request_json = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    metadata = {
        "provider": config.provider,
        "model": config.model,
        "endpoint": config.endpoint,
        "image_name": image_path.name,
        "prompt_characters": len(prompt_text),
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    return headers, json.dumps(request_json).encode("utf-8"), metadata


def _extract_assistant_text(response_payload: dict[str, Any]) -> str:
    choices = list(response_payload.get("choices", []))
    if not choices:
        return ""
    message = dict(choices[0].get("message", {}))
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "") or ""))
        return "\n".join(part for part in text_parts if part.strip()).strip()
    return ""


def call_qwen_api_with_image(*, prompt_text: str, image_path: Path) -> dict[str, Any]:
    config = load_api_config_from_env()
    headers, body, request_metadata = _build_openrouter_request(config, prompt_text, image_path)
    last_error: str | None = None

    for attempt in range(1, config.max_retries + 2):
        started = time.perf_counter()
        request = urllib.request.Request(config.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
            response_json = json.loads(response_text)
            latency_seconds = time.perf_counter() - started
            return {
                "status": "success",
                "provider": config.provider,
                "model": config.model,
                "latency_seconds": round(latency_seconds, 3),
                "assistant_text": _extract_assistant_text(response_json),
                "request_metadata": request_metadata,
                "raw_response_text": response_text,
                "error_message": None,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            latency_seconds = time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt > config.max_retries:
                return {
                    "status": "failed",
                    "provider": config.provider,
                    "model": config.model,
                    "latency_seconds": round(latency_seconds, 3),
                    "assistant_text": "",
                    "request_metadata": request_metadata,
                    "raw_response_text": "",
                    "error_message": last_error,
                }
            time.sleep(min(2.0, attempt * 0.5))
    return {
        "status": "failed",
        "provider": config.provider,
        "model": config.model,
        "latency_seconds": 0.0,
        "assistant_text": "",
        "request_metadata": request_metadata,
        "raw_response_text": "",
        "error_message": last_error or "Unknown API error.",
    }
