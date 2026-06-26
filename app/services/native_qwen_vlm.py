import json
import torch
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger

from app.core.config import settings, PROJECT_ROOT
from app.schemas.frame import FrameRichMetadata
from app.services.ocr import OCRService
from app.services import vlm_prompt as vlm_prompt_module
from app.services.vlm_utils import (
    clean_json_response,
    finalize_frame_metadata,
)

VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "SHARED_VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    raise ImportError(
        "No supported VLM prompt symbol found in app.services.vlm_prompt. "
        "Expected SHARED_VLM_FRAME_METADATA_PROMPT or VLM_FRAME_METADATA_PROMPT."
    )


def _unpack_frame_tuple(frame_tuple: Tuple[Any, ...]) -> Tuple[str, str, float, Path, Path]:
    if len(frame_tuple) >= 5:
        frame_id, video_id, ts, analysis_path, original_path = frame_tuple[:5]
    else:
        frame_id, video_id, ts, analysis_path = frame_tuple[:4]
        original_path = analysis_path
    return frame_id, video_id, ts, Path(analysis_path), Path(original_path)


def _frame_detection_context(frame_tuple: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(frame_tuple) >= 6 and isinstance(frame_tuple[5], dict):
        return frame_tuple[5]
    return {}


def _prompt_with_detection_context(frame_tuple: Tuple[Any, ...]) -> str:
    context = _frame_detection_context(frame_tuple)
    if not context:
        return VLM_FRAME_METADATA_PROMPT

    lines = []
    detected_objects = context.get("detected_objects", [])
    if detected_objects:
        object_names = ", ".join(
            det.get("class_name", "unknown") for det in detected_objects[:10]
        )
        lines.append(f"Detector hints for CURRENT frame: {object_names}.")
    track_ids = context.get("track_ids", [])
    if track_ids:
        lines.append(f"Tracking hints: stable entities visible with track ids {track_ids}.")
    reasons = context.get("candidate_reasons", [])
    if reasons:
        lines.append(f"Frame selected because: {', '.join(reasons)}.")

    if not lines:
        return VLM_FRAME_METADATA_PROMPT
    return VLM_FRAME_METADATA_PROMPT + "\n\nAdditional detector context:\n- " + "\n- ".join(lines)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

FRAME_METADATA_MAX_TOKENS_CAP = 256

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
                max_tokens=min(max(int(settings.QWEN_MAX_NEW_TOKENS or FRAME_METADATA_MAX_TOKENS_CAP), 64), FRAME_METADATA_MAX_TOKENS_CAP),
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
            # Fallback to MockVLMService
            from app.services.mock_vlm import MockVLMService
            return await MockVLMService.generate_metadata_batch(batch_frames)

        import asyncio
        results: List[Tuple[FrameRichMetadata, Dict[str, float]]] = []

        try:
            logger.debug(f"Preparing batch of {len(batch_frames)} images for vLLM...")
            
            # Format all prompts for vLLM True Batching
            vllm_messages = [
                cls._create_vllm_prompt(_unpack_frame_tuple(frame_tuple)[3], _prompt_with_detection_context(frame_tuple))
                for frame_tuple in batch_frames
            ]

            vlm_start = time.perf_counter()
            # Pass entire batch to vLLM at once (Task 4)
            vlm_outputs = await cls._async_vllm_generate(vllm_messages)
            vlm_ms_avg = ((time.perf_counter() - vlm_start) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            # Fully parallel OCR batch extraction
            ocr_start_global = time.perf_counter()
            
            async def timed_ocr(path, frame_id):
                res = await asyncio.to_thread(OCRService.extract_text, path)
                return res

            ocr_paths = [_unpack_frame_tuple(frame_tuple)[4] for frame_tuple in batch_frames]
            ocr_tasks = [timed_ocr(path, batch_frames[idx][0]) for idx, path in enumerate(ocr_paths)]
            ocr_results = await asyncio.gather(*ocr_tasks)
            ocr_duration_ms_avg = ((time.perf_counter() - ocr_start_global) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            for idx, raw_out in enumerate(vlm_outputs):
                frame_id, video_id, ts, path, original_path = _unpack_frame_tuple(batch_frames[idx])
                
                logger.info(f"\nRAW_QWEN_OUTPUT_START\n{raw_out}\nRAW_QWEN_OUTPUT_END\n")
                
                if not raw_out:
                    logger.warning(f"Empty response from vLLM for {frame_id}")
                    continue
                    
                try:
                    repair_start = time.perf_counter()
                    cleaned_out = clean_json_response(raw_out)
                    import json
                    parsed_raw = json.loads(cleaned_out)
                    repair_duration_ms = (time.perf_counter() - repair_start) * 1000.0

                    ocr_duration_ms = ocr_duration_ms_avg

                    if idx == 0:
                        try:
                            trace_data = {
                                "1_raw_qwen_output": raw_out,
                                "2_cleaned_json_string": cleaned_out,
                                "2b_parsed_dict": parsed_raw,
                                "3_normalized_metadata": None,
                                "4_final_framerichmetadata": None
                            }
                            with open("debug_metadata_trace.json", "w", encoding="utf-8") as f:
                                json.dump(trace_data, f, indent=4)
                        except Exception as e:
                            logger.error(f"Failed to write debug trace: {e}")

                    val_start = time.perf_counter()
                    rich_meta = finalize_frame_metadata(
                        parsed_raw=parsed_raw,
                        frame_id=frame_id,
                        video_id=video_id,
                        timestamp_seconds=ts,
                        frame_path=original_path,
                        ocr_result=ocr_results[idx],
                        project_root=PROJECT_ROOT,
                        detection_context=_frame_detection_context(batch_frames[idx]),
                    )
                    val_duration_ms = (time.perf_counter() - val_start) * 1000.0

                    if idx == 0:
                        try:
                            import json
                            with open("debug_metadata_trace.json", "r", encoding="utf-8") as f:
                                trace_data = json.load(f)
                            trace_data["3_normalized_metadata"] = rich_meta.model_dump()
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


