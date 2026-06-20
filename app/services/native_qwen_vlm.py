import json
import torch
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger

from app.core.config import settings, PROJECT_ROOT
from app.schemas.frame import FrameRichMetadata
from app.core.utils import calculate_time_snippet, format_timestamp_human
from app.services.ocr import OCRService
from app.services.activity_recovery import ActivityRecoveryService
from app.services.qwen_vlm import QwenVLMService

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class NativeQwenVLMService:
    """Service managing Qwen2.5-VL model using vLLM for true batch inference."""

    _llm = None
    _sampling_params = None
    _is_initialized = False

    @classmethod
    def initialize(cls) -> None:
        """Initializes the vLLM engine with Qwen2.5-VL-7B-Instruct."""
        if cls._is_initialized:
            return

        if settings.MOCK_MODEL:
            logger.info("Mock model mode is enabled. Skipping vLLM initialization.")
            cls._is_initialized = True
            return

        if LLM is None:
            raise ImportError("vLLM is not installed. Please install vllm to use NativeQwenVLMService.")

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"Initializing vLLM engine for {model_id}...")
        
        try:
            # Task 3: vLLM Initialization Configuration
            cls._llm = LLM(
                model=model_id,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.90,
                quantization="awq", # Task 3: INT8 quantization
                limit_mm_per_prompt={"image": 1},
            )
            
            cls._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=256,
            )
            
            cls._is_initialized = True
            
            # Add startup logs
            vram_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            vram_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
            logger.info(f"Loaded model: {model_id} natively via vLLM.")
            logger.info(f"VRAM usage: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved by vLLM engine.")
            logger.info(f"Available memory: vLLM KV Cache initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise

    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Returns the health status and engine configuration."""
        is_cuda_avail = torch.cuda.is_available()
        gpu_memory = 0.0
        if is_cuda_avail:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
        return {
            "engine_type": "native_vllm",
            "model_loaded": "Qwen/Qwen2.5-VL-7B-Instruct" if cls._llm else "None",
            "gpu_memory": f"{gpu_memory:.2f} GB",
            "cuda_available": is_cuda_avail,
            "quantization": "INT8 (AWQ)" if cls._llm else "N/A",
            "batch_size": "Dynamic (Continuous Batching)",
            "status": "healthy" if cls._is_initialized else "uninitialized"
        }

    @classmethod
    def _create_vllm_prompt(cls, image_path: Path, prompt_text: str) -> List[Dict[str, Any]]:
        """Formats the prompt specifically for vLLM."""
        return [
            {
                "role": "system",
                "content": "You are a strict JSON-only vision analysis assistant. Respond ONLY with a valid JSON object matching the requested schema. Do not use markdown backticks. Return raw JSON."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path.absolute()}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    @classmethod
    def generate_single_frame(cls, frame_id: str, video_id: str, timestamp_seconds: float, frame_path: Path) -> FrameRichMetadata:
        """Processes a single frame."""
        batch = [(frame_id, video_id, timestamp_seconds, frame_path)]
        results = cls.generate_metadata_batch(batch)
        if not results:
            raise RuntimeError(f"Failed to generate metadata for single frame {frame_id}")
        return results[0][0]

    @classmethod
    async def _async_vllm_generate(cls, messages_batch: List[Any]) -> List[str]:
        """Wrapper for vLLM generate to keep the async interface."""
        import asyncio
        # vLLM LLM engine is technically synchronous and blocks, 
        # but in a real async server AsyncLLMEngine would be used. 
        # We wrap it in a thread for this implementation to avoid blocking event loop.
        def run_vllm():
            logger.info("--- TRUE BATCHING AUDIT ---")
            logger.info(f"Submitting {len(messages_batch)} frames in a single batched pass to vLLM.")
            outputs = cls._llm.chat(messages=messages_batch, sampling_params=cls._sampling_params)
            return [out.outputs[0].text for out in outputs]
            
        return await asyncio.to_thread(run_vllm)

    @classmethod
    async def generate_metadata_batch(
        cls, batch_frames: List[Tuple[str, str, float, Path]]
    ) -> List[Tuple[FrameRichMetadata, Dict[str, float]]]:
        """Runs batch image inference on Qwen2.5-VL to extract structured metadata using vLLM."""
        cls.initialize()

        if settings.MOCK_MODEL:
            # Fallback to Ollama mock generation
            return await QwenVLMService.generate_metadata_batch(batch_frames)

        import asyncio
        results: List[Tuple[FrameRichMetadata, Dict[str, float]]] = []

        prompt_guidelines = (
            "Analyze the image and return a raw JSON object detailing its visual contents objectively. "
            "You MUST return a single JSON object (enclosed in curly braces {}), NOT a JSON array. "
            "The JSON object MUST strictly adhere to this schema:\n"
            "{\n"
            '  "scene_type": "indoor/outdoor description",\n'
            '  "scene_description": "objective description of the environment",\n'
            '  "objects": [\n'
            "    {\n"
            '      "id": "unique id e.g. car_1, person_2",\n'
            '      "type": "object category",\n'
            '      "subtype": "specific type",\n'
            '      "color": "dominant color",\n'
            '      "condition": "standing/walking/sitting/lying/bending/moving/stationary/unknown",\n'
            '      "attributes": ["list of describing attributes"]\n'
            "    }\n"
            "  ],\n"
            '  "events": [\n'
            "    {\n"
            '      "event_type": "interaction/observation/none",\n'
            '      "description": "precise, objective sentence describing an observable interaction",\n'
            '      "actors": ["object ids involved e.g. car_1, person_2"],\n'
            '      "severity": "low"\n'
            "    }\n"
            "  ],\n"
            '  "people_count": 0,\n'
            '  "activities": ["list of ongoing activities"],\n'
            '  "keywords": ["search tag keywords"],\n'
            '  "caption": "neutral, objective scene description detailing exactly what is visible. Do not infer incidents."\n'
            "}\n"
            "CRITICAL RULES:\n"
            "- Give each object a unique 'id' so events can reference them via 'actors'.\n"
            "- Describe the scene objectively. Extract visual facts only.\n"
            "- NEVER infer incidents, causality, or intent from a single frame.\n"
            "- NEVER assume a person has fallen; use neutral posture descriptors like 'lying' or 'bending'.\n"
            "- NEVER assume a collision, speeding, abandonment, or criminal activity occurred.\n"
            "- If the scene has no notable interactions, return: \"events\": []\n"
            "- Respond ONLY with raw JSON. No markdown, no backticks, no commentary."
        )

        try:
            logger.debug(f"Preparing batch of {len(batch_frames)} images for vLLM...")
            
            # Format all prompts for vLLM True Batching
            vllm_messages = [cls._create_vllm_prompt(path, prompt_guidelines) for _, _, _, path in batch_frames]

            vlm_start = time.perf_counter()
            # Pass entire batch to vLLM at once (Task 4)
            vlm_outputs = await cls._async_vllm_generate(vllm_messages)
            vlm_ms_avg = ((time.perf_counter() - vlm_start) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            # Fully parallel OCR batch extraction
            ocr_start_global = time.perf_counter()
            
            async def timed_ocr(path, frame_id):
                res = await asyncio.to_thread(OCRService.extract_text, path)
                return res

            ocr_tasks = [timed_ocr(path, batch_frames[idx][0]) for idx, path in enumerate([f[3] for f in batch_frames])]
            ocr_results = await asyncio.gather(*ocr_tasks)
            ocr_duration_ms_avg = ((time.perf_counter() - ocr_start_global) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            for idx, raw_out in enumerate(vlm_outputs):
                frame_id, video_id, ts, path = batch_frames[idx]
                
                logger.info(f"\nRAW_QWEN_OUTPUT_START\n{raw_out}\nRAW_QWEN_OUTPUT_END\n")
                
                if not raw_out:
                    logger.warning(f"Empty response from vLLM for {frame_id}")
                    continue
                    
                try:
                    repair_start = time.perf_counter()
                    cleaned_out = QwenVLMService._clean_json_response(raw_out)
                    import json
                    parsed_raw = json.loads(cleaned_out)
                    parsed = QwenVLMService._normalize_metadata_dict(parsed_raw.copy())
                    repair_duration_ms = (time.perf_counter() - repair_start) * 1000.0

                    time_snippet = calculate_time_snippet(ts, interval_seconds=1.0)
                    parsed.update(time_snippet)

                    parsed["ocr"] = ocr_results[idx]
                    ocr_duration_ms = ocr_duration_ms_avg

                    parsed["frame_id"] = frame_id
                    parsed["video_id"] = video_id
                    parsed["timestamp_seconds"] = ts
                    parsed["timestamp_human"] = QwenVLMService._format_timestamp_human(ts)
                    parsed["frame_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")

                    parsed = ActivityRecoveryService.apply(parsed)
                    parsed["search_text"] = QwenVLMService._generate_search_text(parsed)
                    
                    if idx == 0:
                        try:
                            trace_data = {
                                "1_raw_qwen_output": raw_out,
                                "2_cleaned_json_string": cleaned_out,
                                "2b_parsed_dict": parsed_raw,
                                "3_normalized_metadata": parsed.copy(),
                                "4_final_framerichmetadata": None
                            }
                            with open("debug_metadata_trace.json", "w", encoding="utf-8") as f:
                                json.dump(trace_data, f, indent=4)
                        except Exception as e:
                            logger.error(f"Failed to write debug trace: {e}")

                    val_start = time.perf_counter()
                    rich_meta = FrameRichMetadata(**parsed)
                    val_duration_ms = (time.perf_counter() - val_start) * 1000.0

                    if idx == 0:
                        try:
                            import json
                            with open("debug_metadata_trace.json", "r", encoding="utf-8") as f:
                                trace_data = json.load(f)
                            trace_data["4_final_framerichmetadata"] = rich_meta.model_dump()
                            with open("debug_metadata_trace.json", "w", encoding="utf-8") as f:
                                json.dump(trace_data, f, indent=4)
                        except Exception as e:
                            logger.error(f"Failed to append to debug trace: {e}")

                    timings = {
                        "ocr_ms": ocr_duration_ms,
                        "vlm_ms": vlm_ms_avg,
                        "json_repair_ms": repair_duration_ms,
                        "validation_ms": val_duration_ms,
                    }
                    logger.info(f"Performance for frame {frame_id} -> vLLM: {vlm_ms_avg:.2f}ms, OCR: {ocr_duration_ms:.2f}ms")
                    results.append((rich_meta, timings))

                except Exception as exc:
                    logger.warning(
                        f"Validation/parsing failure on vLLM result index {idx} "
                        f"for frame {frame_id}: {str(exc)}. Raw: {raw_out}"
                    )
                    continue

        except Exception as exc:
            logger.exception("Native vLLM execution failure on batch generation")
            raise RuntimeError(f"Batch generation failed: {str(exc)}")

        return results
