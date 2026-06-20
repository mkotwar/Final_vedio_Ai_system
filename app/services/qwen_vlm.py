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
        return format_timestamp_human(seconds)

    @classmethod
    def _clean_json_response(cls, raw_response: str) -> str:
        """Strips markdown block wraps (e.g. ```json ... ```) from VLM answers robustly."""
        cleaned = raw_response.strip()
        # Find json/markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if json_match:
            cleaned = json_match.group(1).strip()

        # Strip any extra outer backticks just in case
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Remove trailing commentaries often generated outside markdown block
        bracket_match = re.search(r"([\[\{].*[\]\}])", cleaned, re.DOTALL)
        if bracket_match:
            cleaned = bracket_match.group(1).strip()

        try:
            import json_repair
            parsed = json_repair.repair_json(cleaned, return_objects=True)
            if not isinstance(parsed, (dict, list)):
                parsed = json.loads(cleaned)
        except Exception:
            try:
                parsed = json.loads(cleaned)
            except Exception:
                # Fallback: extract the first curly brace block
                dict_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                if dict_match:
                    try:
                        import json_repair
                        parsed = json_repair.repair_json(dict_match.group(1), return_objects=True)
                        if not isinstance(parsed, (dict, list)):
                            parsed = json.loads(dict_match.group(1))
                    except Exception:
                        parsed = json.loads(dict_match.group(1).strip())
                else:
                    raise

        # Handle list-wrapped responses
        if isinstance(parsed, list) and len(parsed) > 0:
            reconstructed = {}
            is_field_list = False
            field_keys = {
                "scene_type", "scene_description", "objects", "people_count",
                "activities", "keywords", "caption", "events",
            }

            for item in parsed:
                if isinstance(item, dict) and "type" in item:
                    t = str(item["type"]).lower()
                    if t in field_keys or t == "location" or t == "time" or t == "environment":
                        is_field_list = True
                        key_map = {
                            "location": "scene_type",
                            "time": "keywords",
                            "environment": "scene_description",
                        }
                        schema_key = key_map.get(t, t)

                        val = None
                        for field in [
                            "description", "value", "count", "summary", "summary_text",
                            "summary_caption", "text_summary", "tags", "search_tags", "counts",
                        ]:
                            if field in item:
                                val = item[field]
                                break
                        if val is None:
                            val = item.get("attributes")

                        if val is not None:
                            reconstructed[schema_key] = val

            if is_field_list:
                return json.dumps(reconstructed)

            # Check if it's a simple list of detected objects
            first_item = parsed[0]
            if isinstance(first_item, dict) and (
                "type" in first_item or "subtype" in first_item or "color" in first_item
            ):
                reconstructed = {
                    "objects": parsed,
                    "events": [],
                    "scene_type": "unknown",
                    "scene_description": "Detected objects in the frame.",
                    "caption": "A frame containing several objects.",
                    "people_count": sum(
                        1 for x in parsed
                        if isinstance(x, dict) and "person" in str(x.get("type", "")).lower()
                    ),
                    "activities": [],
                    "keywords": list(set(
                        str(x.get("type", "")) for x in parsed
                        if isinstance(x, dict) and x.get("type")
                    )),
                }
                return json.dumps(reconstructed)

            if isinstance(first_item, dict):
                return json.dumps(first_item)

        return json.dumps(parsed)

    @classmethod
    def _normalize_metadata_dict(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and repairs the parsed JSON dict to strictly match FrameRichMetadata schema."""
        # Ensure base string fields exist
        if "scene_type" not in parsed or not parsed["scene_type"]:
            parsed["scene_type"] = "unknown"
        if "scene_description" not in parsed or not parsed["scene_description"]:
            parsed["scene_description"] = ""
        if "caption" not in parsed or not parsed["caption"]:
            parsed["caption"] = "No description available."

        # Cross-populate scene_description and caption if one is empty
        if not parsed["scene_description"] and parsed["caption"]:
            parsed["scene_description"] = parsed["caption"]
        elif parsed["scene_description"] and not parsed["caption"]:
            parsed["caption"] = parsed["scene_description"]

        # Ensure lists are lists of strings
        for list_field in ["activities", "keywords"]:
            val = parsed.get(list_field)
            if val is None:
                parsed[list_field] = []
            elif isinstance(val, str):
                parsed[list_field] = [s.strip() for s in val.split(",") if s.strip()]
            elif not isinstance(val, list):
                parsed[list_field] = [str(val)]
            else:
                parsed[list_field] = [str(item) for item in val]

        # Ensure people_count is int
        pc = parsed.get("people_count")
        if pc is None:
            parsed["people_count"] = 0
        else:
            try:
                parsed["people_count"] = int(pc)
            except (ValueError, TypeError):
                parsed["people_count"] = 0

        # Ensure objects list exists and is normalized
        objs = parsed.get("objects")
        if not isinstance(objs, list):
            objs = []

        normalized_objs = []
        for obj in objs:
            if not isinstance(obj, dict):
                continue

            # Map camelCase subType to lowercase subtype if needed
            sub_type = obj.get("subtype")
            if sub_type is None:
                sub_type = obj.get("subType", "")

            obj_id = str(obj.get("id", "")).strip()
            obj_type = str(obj.get("type", "unknown"))
            obj_subtype = str(sub_type).lower().strip()

            # Normalize subtypes to prevent oscillation
            if obj_subtype in ["adult male", "male", "individual", "pedestrian", "visitor", "man", "woman", "female", "guard", "security"]:
                obj_subtype = "person"
            elif obj_subtype in ["shopper"]:
                obj_subtype = "customer"
            elif obj_subtype in ["staff", "worker"]:
                obj_subtype = "employee"

            obj_color = str(obj.get("color", ""))
            obj_condition = str(obj.get("condition", "normal")).lower().strip()

            # Normalize attributes list
            attrs = obj.get("attributes")
            if attrs is None:
                attrs_list = []
            elif isinstance(attrs, str):
                attrs_list = [s.strip() for s in attrs.split(",") if s.strip()]
            elif isinstance(attrs, list):
                attrs_list = []
                for attr in attrs:
                    if isinstance(attr, dict):
                        dict_parts = [f"{k}: {v}" if v else k for k, v in attr.items()]
                        attrs_list.append(", ".join(dict_parts))
                    else:
                        attrs_list.append(str(attr))
            else:
                attrs_list = [str(attrs)]

            normalized_objs.append({
                "id": obj_id,
                "type": obj_type,
                "subtype": obj_subtype,
                "color": obj_color,
                "condition": obj_condition,
                "attributes": attrs_list,
            })

        parsed["objects"] = normalized_objs

        # ── Normalize events list ──────────────────────────────────────────────
        events = parsed.get("events")
        if not isinstance(events, list):
            events = []

        valid_severities = {"low", "medium", "high", "critical"}
        normalized_events = []

        for evt in events:
            if not isinstance(evt, dict):
                continue

            event_type = str(evt.get("event_type", "unknown")).lower().strip()
            description = str(evt.get("description", "")).strip()
            severity = str(evt.get("severity", "medium")).lower().strip()

            # Normalize actors to list of strings
            actors = evt.get("actors", [])
            if isinstance(actors, str):
                actors = [a.strip() for a in actors.split(",") if a.strip()]
            elif not isinstance(actors, list):
                actors = []
            else:
                actors = [str(a) for a in actors]

            # Fallback severity if unrecognized value
            if severity not in valid_severities:
                severity = "medium"

            # Skip placeholder "none" events
            if event_type and event_type != "none":
                normalized_events.append({
                    "event_type": event_type,
                    "description": description,
                    "actors": actors,
                    "severity": severity,
                })

        parsed["events"] = normalized_events

        # Pre-merge event types into activities so ActivityRecoveryService
        # doesn't overwrite them with generic fallbacks
        if normalized_events:
            existing_activities = parsed.get("activities", [])
            event_activity_labels = [
                e["event_type"].replace("_", " ") for e in normalized_events
            ]
            # Merge preserving order, no duplicates
            merged = list(dict.fromkeys(existing_activities + event_activity_labels))
            parsed["activities"] = merged
        # ── End events normalization ───────────────────────────────────────────

        return parsed

    @classmethod
    def _generate_search_text(cls, meta: Dict[str, Any]) -> str:
        """Autogenerates search indexing block by joining structural text descriptors."""
        parts = [
            meta.get("scene_type", ""),
            meta.get("scene_description", ""),
            meta.get("caption", ""),
            ", ".join(meta.get("activities", [])),
            ", ".join(meta.get("keywords", [])),
        ]

        # Index event descriptions and types for searchability
        for evt in meta.get("events", []):
            parts.append(evt.get("event_type", ""))
            parts.append(evt.get("description", ""))

        # Append OCR detected text if present
        ocr_data = meta.get("ocr")
        if ocr_data:
            if isinstance(ocr_data, dict):
                detected = ocr_data.get("detected_text", [])
            else:
                detected = getattr(ocr_data, "detected_text", [])
            parts.extend(detected)

        # Append object specifics
        for obj in meta.get("objects", []):
            color = obj.get("color", "")
            subtype = obj.get("subtype", "")
            parts.append(f"{color} {subtype}".strip())
            parts.extend(obj.get("attributes", []))

        full_text = " ".join(parts).lower()
        cleaned_text = re.sub(r"\s+", " ", full_text).strip()
        return cleaned_text

    @classmethod
    def _generate_mock_metadata(
        cls, frame_id: str, video_id: str, timestamp_seconds: float
    ) -> FrameRichMetadata:
        """Generates synthetic plausible rich frame metadata for development and tests."""
        ts_human = cls._format_timestamp_human(timestamp_seconds)

        parts = frame_id.split("_f")
        frame_idx_str = parts[-1] if len(parts) > 1 else "0001"
        frame_path = f"data/frames/{video_id}/frame_{frame_idx_str}.jpg"

        is_even = int(timestamp_seconds) % 2 == 0

        if is_even:
            mock_data = {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_seconds": timestamp_seconds,
                "timestamp_human": ts_human,
                "frame_path": frame_path,
                "scene_type": "outdoor street",
                "scene_description": "An outdoor city street view under daylight.",
                "objects": [
                    {
                        "id": "car_1",
                        "type": "vehicle",
                        "subtype": "car",
                        "color": "blue",
                        "condition": "moving",
                        "attributes": ["moving", "sedan"],
                    },
                    {
                        "id": "person_1",
                        "type": "pedestrian",
                        "subtype": "person",
                        "color": "black",
                        "condition": "normal",
                        "attributes": ["walking", "carrying bag"],
                    },
                ],
                "events": [],
                "people_count": 1,
                "activities": ["driving", "walking"],
                "keywords": ["street", "traffic", "daylight", "city"],
                "caption": "A blue car driving down a busy city street while a pedestrian walks on the sidewalk.",
            }
        else:
            mock_data = {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_seconds": timestamp_seconds,
                "timestamp_human": ts_human,
                "frame_path": frame_path,
                "scene_type": "indoor office",
                "scene_description": "An indoor office meeting room workspace.",
                "objects": [
                    {
                        "id": "chair_1",
                        "type": "furniture",
                        "subtype": "chair",
                        "color": "grey",
                        "condition": "normal",
                        "attributes": ["office chair", "mesh back"],
                    },
                    {
                        "id": "laptop_1",
                        "type": "electronics",
                        "subtype": "laptop",
                        "color": "silver",
                        "condition": "normal",
                        "attributes": ["open", "on table"],
                    },
                ],
                "events": [],
                "people_count": 0,
                "activities": ["working", "sitting"],
                "keywords": ["office", "workplace", "corporate", "desk"],
                "caption": "A laptop open on a table next to a grey office chair in an empty conference room.",
            }

        time_snippet = calculate_time_snippet(timestamp_seconds, interval_seconds=1.0)
        mock_ocr = {
            "detected_text": ["GATE 1", "MH12AB1234"] if is_even else ["OFFICE ENTRY", "VISITOR"],
            "license_plates": ["MH12AB1234"] if is_even else [],
        }
        mock_data.update(time_snippet)
        mock_data["ocr"] = mock_ocr
        mock_data["search_text"] = cls._generate_search_text(mock_data)

        return FrameRichMetadata(**mock_data)

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
            res = []
            for frame_id, video_id, ts, _ in batch_frames:
                mock_meta = cls._generate_mock_metadata(frame_id, video_id, ts)
                res.append((mock_meta, {
                    "ocr_ms": 0.0,
                    "vlm_ms": 0.0,
                    "json_repair_ms": 0.0,
                    "validation_ms": 0.0,
                }))
            return res

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