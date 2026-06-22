import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import httpx

from app.core.config import settings, PROJECT_ROOT
from app.core.exceptions import MetadataGenerationError
from app.schemas.frame import FrameRichMetadata, ObjectMetadata
from app.core.utils import calculate_time_snippet, format_timestamp_human
from app.services.ocr import OCRService
from app.services.activity_recovery import ActivityRecoveryService
from app.services.vlm_utils import (
    clean_json_response,
    normalize_metadata_dict,
    format_timestamp_human_vlm,
    generate_search_text,
)


class QwenVLMService:
    """Service managing Qwen2.5-VL model loading and rich metadata inference batch execution."""

    _model: Any = None
    _processor: Any = None
    _device: str = "cpu"
    _model_id: str = ""

    @classmethod
    def load_model(cls) -> None:
        """Initializes connection to Ollama server (no heavy weights loaded in Python)."""
        if settings.MOCK_MODEL:
            logger.info("Mock model mode is enabled.")
            return

        logger.info(f"Connecting to Ollama using model: {settings.QWEN_MODEL_ID}...")
        cls._device = "ollama"
        cls._model_id = settings.QWEN_MODEL_ID

    @classmethod
    def _format_timestamp_human(cls, seconds: float) -> str:
        """Converts float seconds to playback timestamp formatted as HH:MM:SS."""
        return format_timestamp_human_vlm(seconds)

    @classmethod
    def _clean_json_response(cls, raw_response: str) -> str:
        """Strips markdown block wraps (e.g. ```json ... ```) from VLM answers robustly."""
        return clean_json_response(raw_response)

    @classmethod
    def _normalize_metadata_dict(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and repairs the parsed JSON dict to strictly match FrameRichMetadata schema."""
        return normalize_metadata_dict(parsed)

    @classmethod
    def _generate_search_text(cls, meta: Dict[str, Any]) -> str:
        """Autogenerates search indexing block by joining structural text descriptors."""
        return generate_search_text(meta)

    @classmethod
    def _encode_and_compress_image(cls, image_path: Path, max_dimension: int = 1024, quality: int = 85) -> str:
        """Compresses and resizes the image, returning a base64 encoded string."""
        import base64
        import io
        from PIL import Image

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            width, height = img.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((max_dimension / width) * height)
                else:
                    new_height = max_dimension
                    new_width = int((max_dimension / height) * width)
                # Use LANCZOS for high-quality downsampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    async def _call_ollama_async(cls, prompt: str, image_path: Path, client) -> Tuple[str, float]:
        """Calls the Ollama generate API for a single frame asynchronously."""
        import base64
        import time
        start_time = time.perf_counter()
        
        try:
            # Compress the image to dramatically reduce base64 size (~300KB) and prevent Ollama 400 errors
            base64_image = cls._encode_and_compress_image(image_path)
                
            # We don't use "format": "json" because it causes 400 Bad Request or empty content
            # bugs with Qwen3-VL reasoning models in Ollama. The prompt and json_repair will handle it.
            payload = {
                "model": settings.QWEN_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict JSON-only vision analysis assistant. Respond ONLY with a valid JSON object matching the requested schema. Do not use markdown backticks. Return raw JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image],
                    }
                ],
                "stream": False,
                "options": {
                    "num_predict": 256,
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 0.1,
                }

                
            }
            
            max_retries = 3
            base_delay = 2.0
            raw_out = ""
            result = {}
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Sending Ollama /api/chat request for {image_path.name} (Attempt {attempt+1}/{max_retries}) | model={settings.QWEN_MODEL_ID} | image_size={len(base64_image)} chars")
                    response = await client.post("http://localhost:11434/api/chat", json=payload, timeout=300.0)
                    logger.info(f"Ollama HTTP status for {image_path.name}: {response.status_code}")
                    response.raise_for_status()
                    result = response.json()
                    raw_out = result.get("message", {}).get("content", "")
                    break  # Success
                except httpx.HTTPStatusError as e:
                    error_body = e.response.text
                    if e.response.status_code in (400, 429, 502, 503) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Ollama API returned {e.response.status_code} for {image_path.name}. Body: {error_body}. Retrying in {delay}s...")
                        import asyncio
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Ollama error {e.response.status_code} for {image_path.name}: {error_body}")
                        raise e
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Ollama network error for {image_path.name}: {type(e).__name__}. Retrying in {delay}s...")
                        import asyncio
                        await asyncio.sleep(delay)
                    else:
                        raise e
            
            # Fallback: if content is empty, try to extract JSON from the thinking field
            # qwen3-vl sometimes puts the full analysis in "thinking" and leaves "content" empty
            if not raw_out:
                thinking_text = result.get("message", {}).get("thinking", "")
                if thinking_text:
                    logger.warning(f"Ollama returned empty content but has thinking text for {image_path.name} ({len(thinking_text)} chars). Attempting to extract JSON from thinking.")
                    # Use brace-counting to extract the largest top-level JSON object
                    best_candidate = ""
                    start_idx = thinking_text.find('{')
                    while start_idx != -1:
                        depth = 0
                        in_string = False
                        escape_next = False
                        end_idx = None
                        for i in range(start_idx, len(thinking_text)):
                            ch = thinking_text[i]
                            if escape_next:
                                escape_next = False
                                continue
                            if ch == '\\' and in_string:
                                escape_next = True
                                continue
                            if ch == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if in_string:
                                continue
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    end_idx = i
                                    break
                        if end_idx is not None:
                            candidate = thinking_text[start_idx:end_idx + 1]
                            if len(candidate) > len(best_candidate):
                                best_candidate = candidate
                        start_idx = thinking_text.find('{', start_idx + 1)
                    
                    if best_candidate:
                        try:
                            json.loads(best_candidate)
                            raw_out = best_candidate
                            logger.info(f"Successfully extracted JSON ({len(raw_out)} chars) from thinking field for {image_path.name}")
                        except json.JSONDecodeError:
                            try:
                                import json_repair
                                repaired = json_repair.repair_json(best_candidate, return_objects=False)
                                json.loads(repaired)
                                raw_out = repaired
                                logger.info(f"Extracted and repaired JSON ({len(raw_out)} chars) from thinking field for {image_path.name}")
                            except Exception:
                                logger.error(f"Found JSON-like text in thinking but could not parse it for {image_path.name}")
                    
                    # Last resort: if thinking was truncated mid-JSON, try to repair the whole tail
                    if not raw_out:
                        tail_start = thinking_text.find('{')
                        if tail_start != -1:
                            try:
                                import json_repair
                                repaired = json_repair.repair_json(thinking_text[tail_start:], return_objects=False)
                                json.loads(repaired)
                                raw_out = repaired
                                logger.info(f"Repaired truncated JSON ({len(raw_out)} chars) from thinking field for {image_path.name}")
                            except Exception:
                                pass
                    
                    if not raw_out:
                        logger.error(f"Could not extract usable JSON from thinking field for {image_path.name}")

            if not raw_out:
                logger.error(f"Ollama returned EMPTY response for {image_path.name}. Full JSON: {json.dumps(result, indent=2)[:2000]}")
            else:
                logger.info(f"Ollama returned {len(raw_out)} chars for {image_path.name}: {raw_out[:200]}...")
            
        except Exception as e:
            logger.error(f"Ollama API call failed for {image_path.name}: {type(e).__name__}: {e}")
            raw_out = ""
            
        duration = (time.perf_counter() - start_time) * 1000.0
        return raw_out, duration

    @classmethod
    async def generate_metadata_batch(
        cls, batch_frames: List[Tuple[str, str, float, Path]]
    ) -> List[FrameRichMetadata]:
        """Runs batch image inference on Qwen2.5-VL to extract structured metadata.

        Args:
            batch_frames: List of tuples (frame_id, video_id, timestamp_seconds, frame_absolute_path)

        Returns:
            list: List of validated FrameRichMetadata objects.
        """
        cls.load_model()

        if settings.MOCK_MODEL:
            from app.services.mock_vlm import MockVLMService
            return await MockVLMService.generate_metadata_batch(batch_frames)

        import httpx
        results: List[Tuple[FrameRichMetadata, Dict[str, float]]] = []

        prompt_guidelines = (
            "Analyze the image and return a raw JSON object detailing its visual contents objectively. "
            "You MUST return a single JSON object (enclosed in curly braces {}), NOT a JSON array. "
            "The JSON object MUST strictly adhere to this schema:\n"
            "{\n"
            '  "scene_type": "string",\n'
            '  "scene_description": "string",\n'
            '  "objects": [\n'
            "    {\n"
            '      "id": "string",\n'
            '      "type": "string",\n'
            '      "subtype": "string",\n'
            '      "color": "string",\n'
            '      "condition": "standing/walking/sitting/lying/bending/moving/stationary/unknown",\n'
            '      "attributes": ["string"]\n'
            "    }\n"
            "  ],\n"
            '  "events": [\n'
            "    {\n"
            '      "event_type": "interaction/observation/none",\n'
            '      "description": "string",\n'
            '      "actors": ["string"],\n'
            '      "severity": "low/medium/high/critical"\n'
            "    }\n"
            "  ],\n"
            '  "people_count": 0,\n'
            '  "activities": ["string"],\n'
            '  "keywords": ["string"],\n'
            '  "caption": "string"\n'
            "}\n"
            "CRITICAL RULES:\n"
            "- DO NOT output placeholder text like 'unique id e.g. person_1'. Generate actual values only.\n"
            "- If no value is known, return null. If no actor exists, return an empty array.\n"
            "- For 'subtype', you MUST use ONLY the following allowed values:\n"
            "    Actors: person, employee, customer\n"
            "    Vehicles: car, truck, motorcycle, bus\n"
            "    Objects: bag, backpack, suitcase, box\n"
            "- If uncertain, fallback to: person, vehicle, or object.\n"
            "- NEVER invent new subtype names. NEVER output: adult male, individual, pedestrian, visitor, shopper.\n"
            "- Give each object a unique 'id' so events can reference them via 'actors'.\n"
            "- Describe the scene objectively. Extract visual facts only.\n"
            "- NEVER infer incidents, causality, or intent from a single frame.\n"
            "- NEVER assume a person has fallen; use neutral posture descriptors like 'lying' or 'bending'.\n"
            "- NEVER assume a collision, speeding, abandonment, or criminal activity occurred.\n"
            "- If the scene has no notable interactions, return: \"events\": []\n"
            "- Respond ONLY with raw JSON. No markdown, no backticks, no commentary."
        )

        try:
            import httpx
            logger.debug(f"Preparing concurrent Ollama inputs for {len(batch_frames)} images...")
            
            # Semaphore to limit concurrent inference calls
            concurrency_limit = 2
            sem = asyncio.Semaphore(concurrency_limit)
            
            async def bounded_call(*args, **kwargs):
                async with sem:
                    return await cls._call_ollama_async(*args, **kwargs)

            # Fire requests to Ollama with bounded concurrency
            async with httpx.AsyncClient() as client:
                tasks = [bounded_call(prompt_guidelines, path, client) for _, _, _, path in batch_frames]
                ollama_results = await asyncio.gather(*tasks)

            # Fully parallel OCR batch extraction
            ocr_start_global = time.perf_counter()
            
            async def timed_ocr(path, frame_id):
                import datetime
                start_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                logger.info(f"{frame_id} OCR Start: {start_str}")
                
                res = await asyncio.to_thread(OCRService.extract_text, path)
                
                end_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                logger.info(f"{frame_id} OCR End:   {end_str}")
                return res

            ocr_tasks = [timed_ocr(path, batch_frames[idx][0]) for idx, path in enumerate([f[3] for f in batch_frames])]
            ocr_results = await asyncio.gather(*ocr_tasks)
            ocr_duration_ms_avg = ((time.perf_counter() - ocr_start_global) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            for idx, (raw_out, vlm_ms) in enumerate(ollama_results):
                frame_id, video_id, ts, path = batch_frames[idx]
                if not raw_out:
                    logger.warning(f"Empty response from Ollama for {frame_id}")
                    continue
                    
                try:
                    repair_start = time.perf_counter()
                    cleaned_out = cls._clean_json_response(raw_out)
                    parsed = json.loads(cleaned_out)
                    parsed = cls._normalize_metadata_dict(parsed)
                    repair_duration_ms = (time.perf_counter() - repair_start) * 1000.0

                    time_snippet = calculate_time_snippet(ts, interval_seconds=1.0)
                    parsed.update(time_snippet)

                    parsed["ocr"] = ocr_results[idx]
                    ocr_duration_ms = ocr_duration_ms_avg

                    parsed["frame_id"] = frame_id
                    parsed["video_id"] = video_id
                    parsed["timestamp_seconds"] = ts
                    parsed["timestamp_human"] = cls._format_timestamp_human(ts)
                    parsed["frame_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")

                    parsed = ActivityRecoveryService.apply(parsed)
                    parsed["search_text"] = cls._generate_search_text(parsed)

                    val_start = time.perf_counter()
                    rich_meta = FrameRichMetadata(**parsed)
                    val_duration_ms = (time.perf_counter() - val_start) * 1000.0

                    timings = {
                        "ocr_ms": ocr_duration_ms,
                        "vlm_ms": vlm_ms,
                        "json_repair_ms": repair_duration_ms,
                        "validation_ms": val_duration_ms,
                    }
                    logger.info(f"Performance for frame {frame_id} -> VLM (Ollama): {vlm_ms:.2f}ms, OCR: {ocr_duration_ms:.2f}ms, JSON Repair: {repair_duration_ms:.2f}ms")
                    results.append((rich_meta, timings))

                except Exception as exc:
                    logger.warning(
                        f"Validation/parsing failure on VLM result index {idx} "
                        f"for frame {frame_id}: {str(exc)}. Raw: {raw_out}"
                    )
                    continue

        except Exception as exc:
            logger.exception("VLM execution failure on batch generation")
            raise MetadataGenerationError(f"Batch generation failed: {str(exc)}")

        return results