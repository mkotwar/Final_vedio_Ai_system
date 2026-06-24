import os
import torch
import time
import asyncio
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from app.core.config import settings, PROJECT_ROOT
from app.schemas.frame import FrameRichMetadata
from app.services.ocr import OCRService
from app.services.vlm_prompt import VLM_FRAME_METADATA_PROMPT
from app.services.vlm_utils import (
    clean_json_response,
    finalize_frame_metadata,
)
from qwen_vl_utils import process_vision_info

MODEL_ID_ALIASES = {
    "qwen2.5vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2vl:2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl:2b": "Qwen/Qwen2-VL-2B-Instruct",
}
FRAME_METADATA_MAX_TOKENS_CAP = 256


def _resolve_model_source(model_id: str) -> Tuple[str, bool]:
    """Resolve configured model ID to either a local HF snapshot path or a repo ID."""
    configured = (model_id or "").strip()
    normalized = MODEL_ID_ALIASES.get(configured.lower(), configured or "Qwen/Qwen2.5-VL-7B-Instruct")

    if "/" not in normalized:
        return normalized, False

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{normalized.replace('/', '--')}"
    ref_path = cache_root / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cache_root / "snapshots" / revision
        if snapshot_path.exists():
            return str(snapshot_path), True

    return normalized, False


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

class NativeQwenTransformersService:
    _model = None
    _processor = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _batch_counter = 0
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Returns the health status and engine configuration."""
        is_cuda_avail = torch.cuda.is_available()
        gpu_memory = 0.0
        if is_cuda_avail:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "engine_type": "native_hf",
            "model_loaded": settings.QWEN_MODEL_ID if cls._model else "None",
            "gpu_memory": f"{gpu_memory:.2f} GB",
            "cuda_available": is_cuda_avail,
            "status": "healthy" if cls._model else "uninitialized"
        }

    @staticmethod
    def _effective_max_new_tokens() -> int:
        """Clamp frame-metadata generation to a sane token budget for speed."""
        configured = int(settings.QWEN_MAX_NEW_TOKENS or FRAME_METADATA_MAX_TOKENS_CAP)
        return max(64, min(configured, FRAME_METADATA_MAX_TOKENS_CAP))

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return
            
        configured_model_id = settings.QWEN_MODEL_ID
        model_source, local_files_only = _resolve_model_source(configured_model_id)
        logger.info(
            f"Loading {configured_model_id} via Transformers on {cls._device} "
            f"(resolved_source={model_source}, local_only={local_files_only})..."
        )
        
        # CRITICAL: 4-bit quantization reduces 14.5GB VRAM footprint to ~5GB, 
        # eliminating the PCIe thrashing OOM slowdown on RTX 5070 Ti.
        # Use device_map={"": "cuda:0"} to strictly bind to GPU and avoid CPU offload float32 bugs.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        try:
            cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                quantization_config=quantization_config,
                device_map={"": "cuda:0"},
                attn_implementation="sdpa",
                local_files_only=local_files_only,
            )
            cls._processor = AutoProcessor.from_pretrained(
                model_source,
                min_pixels=256*28*28, 
                max_pixels=512*28*28,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            logger.exception(f"Failed to load HF VLM model '{configured_model_id}'")
            raise RuntimeError(
                f"Unable to load VLM model '{configured_model_id}'. "
                "Use MOCK_MODEL=true or configure an accessible local/remote model."
            ) from exc
        
        # Add startup logging
        logger.info(f"Active VLM Backend: native_hf")
        is_cuda = torch.cuda.is_available()
        logger.info(f"CUDA Available: {is_cuda}")
        if is_cuda:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            logger.info(f"VRAM used after load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        logger.info(f"Model Loaded: {configured_model_id}")
        logger.info(f"Model dtype: {cls._model.dtype}")
        logger.info(f"First param dtype: {next(cls._model.parameters()).dtype}")
        logger.info(f"First param device: {next(cls._model.parameters()).device}")
        if hasattr(cls._model, 'hf_device_map'):
            logger.info(f"Device map: {cls._model.hf_device_map}")
        else:
            logger.info("No accelerate device map (all on single device). GOOD.")
        
    @classmethod
    def generate_batch(cls, image_paths: List[Path], prompts: Any) -> List[str]:
        cls.load_model()
        
        batch_start_time = time.perf_counter()
        batch_size = len(image_paths)
        if isinstance(prompts, str):
            prompt_list = [prompts] * batch_size
        else:
            prompt_list = list(prompts)
        if len(prompt_list) != batch_size:
            raise ValueError("Prompt count must match image batch size.")
        
        messages_batch = []
        for path, prompt in zip(image_paths, prompt_list):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{path.absolute()}",
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
            
        # Stage 1
        t0 = time.perf_counter()
        texts = [cls._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        t1 = time.perf_counter()
        template_build_ms = (t1 - t0) * 1000.0
        logger.info(f"Template Build Time: {template_build_ms:.2f} ms")
        
        # Stage 2
        t0 = time.perf_counter()
        image_inputs, video_inputs = process_vision_info(messages_batch)
        t1 = time.perf_counter()
        vision_processing_ms = (t1 - t0) * 1000.0
        logger.info(f"Vision Processing Time: {vision_processing_ms:.2f} ms")
        
        # Stage 3
        t0 = time.perf_counter()
        inputs = cls._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(cls._device)
        t1 = time.perf_counter()
        tensor_preparation_ms = (t1 - t0) * 1000.0
        logger.info(f"Tensor Preparation Time: {tensor_preparation_ms:.2f} ms")
        logger.info(f"Input ids shape: {inputs.input_ids.shape}")
        if 'pixel_values' in inputs:
            logger.info(f"Pixel values shape: {inputs.pixel_values.shape}")
        logger.info(f"Batch size: {batch_size}")
            
        # Stage 4
        mem_alloc_before = torch.cuda.memory_allocated() / (1024**3)
        mem_res_before = torch.cuda.memory_reserved() / (1024**3)
        
        t0 = time.perf_counter()
        effective_max_new_tokens = cls._effective_max_new_tokens()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = cls._model.generate(**inputs, max_new_tokens=effective_max_new_tokens)
        t1 = time.perf_counter()
        generate_ms = (t1 - t0) * 1000.0
        
        mem_alloc_after = torch.cuda.memory_allocated() / (1024**3)
        mem_res_after = torch.cuda.memory_reserved() / (1024**3)
        
        logger.info(f"Generate Time: {generate_ms:.2f} ms")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"max_new_tokens: {effective_max_new_tokens}")
        logger.info(f"GPU memory allocated before: {mem_alloc_before:.2f} GB")
        logger.info(f"GPU memory allocated after: {mem_alloc_after:.2f} GB")
        logger.info(f"GPU memory reserved before: {mem_res_before:.2f} GB")
        logger.info(f"GPU memory reserved after: {mem_res_after:.2f} GB")
        
        # Stage 5
        t0 = time.perf_counter()
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = cls._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        t1 = time.perf_counter()
        decode_ms = (t1 - t0) * 1000.0
        logger.info(f"Decode Time: {decode_ms:.2f} ms")
        
        # Stage 6
        total_chars = 0
        for i, out_text in enumerate(output_texts):
            chars = len(out_text)
            total_chars += chars
            logger.info(f"Frame {i+1} Output Length: {chars} chars")
            
        avg_output_length = total_chars / max(1, len(output_texts))
        logger.info(f"Average Output Length: {avg_output_length:.1f} chars")
        
        # Stage 7
        batch_end_time = time.perf_counter()
        total_batch_sec = batch_end_time - batch_start_time
        avg_runtime_per_frame = total_batch_sec / max(1, batch_size)
        
        logger.info(f"Frames Per Batch: {batch_size}")
        logger.info(f"Batch Runtime: {total_batch_sec:.2f} sec")
        logger.info(f"Average Runtime Per Frame: {avg_runtime_per_frame:.2f} sec")
        
        # Stage 8
        total_ms = template_build_ms + vision_processing_ms + tensor_preparation_ms + generate_ms + decode_ms
        if total_ms == 0: total_ms = 1
        
        report = f"""
==========================
QWEN PROFILING REPORT
==========================

Batch Size: {batch_size}
Max New Tokens: {settings.QWEN_MAX_NEW_TOKENS}

Template Build: {template_build_ms:.2f} ms
Vision Processing: {vision_processing_ms:.2f} ms
Tensor Preparation: {tensor_preparation_ms:.2f} ms
Generate: {generate_ms:.2f} ms
Decode: {decode_ms:.2f} ms

Total Batch Runtime: {total_batch_sec:.2f} sec
Average Output Length: {avg_output_length:.1f} chars

Runtime Breakdown:
Template Build = {(template_build_ms / total_ms) * 100:.1f}%
Vision Processing = {(vision_processing_ms / total_ms) * 100:.1f}%
Tensor Preparation = {(tensor_preparation_ms / total_ms) * 100:.1f}%
Generate = {(generate_ms / total_ms) * 100:.1f}%
Decode = {(decode_ms / total_ms) * 100:.1f}%
=========================="""
        logger.info(report)
        
        return output_texts

    @classmethod
    async def _async_hf_generate(cls, image_paths: List[Path], prompts: Any) -> List[str]:
        """Wrapper for HF generate to keep the async interface without blocking."""
        def run_hf():
            return cls.generate_batch(image_paths, prompts)
            
        return await asyncio.to_thread(run_hf)

    @classmethod
    async def generate_metadata_batch(
        cls, batch_frames: List[Tuple[str, str, float, Path]]
    ) -> List[Tuple[FrameRichMetadata, Dict[str, float]]]:
        """Runs batch image inference on Qwen2.5-VL to extract structured metadata using Native HF."""
        if settings.MOCK_MODEL:
            # Fallback to MockVLMService
            from app.services.mock_vlm import MockVLMService
            return await MockVLMService.generate_metadata_batch(batch_frames)

        cls.load_model()

        results: List[Tuple[FrameRichMetadata, Dict[str, float]]] = []

        try:
            logger.debug(f"Preparing batch of {len(batch_frames)} images for Native HF...")
            
            image_paths = [_unpack_frame_tuple(frame_tuple)[3] for frame_tuple in batch_frames]

            vlm_start = time.perf_counter()
            prompts = [_prompt_with_detection_context(frame_tuple) for frame_tuple in batch_frames]
            vlm_outputs = await cls._async_hf_generate(image_paths, prompts)
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
                
                if not raw_out:
                    logger.warning(f"Empty response from Native HF for {frame_id}")
                    continue
                    
                # Save raw JSON before repair to inspect for truncation or bloat
                dump_dir = PROJECT_ROOT / "data" / "logs"
                dump_dir.mkdir(parents=True, exist_ok=True)
                with open(dump_dir / f"raw_qwen_output_{frame_id}.json", "w", encoding="utf-8") as f:
                    f.write(raw_out)
                    
                try:
                    repair_start = time.perf_counter()
                    cleaned_out = clean_json_response(raw_out)
                    import json
                    parsed_raw = json.loads(cleaned_out)
                    repair_duration_ms = (time.perf_counter() - repair_start) * 1000.0

                    ocr_duration_ms = ocr_duration_ms_avg

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

                    logger.info(
                    f"Postprocessor: people={rich_meta.people_count}, "
                    f"objects={len(rich_meta.objects)}"
                    )
                    
                    val_duration_ms = (time.perf_counter() - val_start) * 1000.0
                    
                    timings = {
                        "ocr_ms": ocr_duration_ms,
                        "vlm_ms": vlm_ms_avg,
                        "json_repair_ms": repair_duration_ms,
                        "validation_ms": val_duration_ms,
                    }
                    logger.info(f"Performance for frame {frame_id} -> Native HF: {vlm_ms_avg:.2f}ms, OCR: {ocr_duration_ms:.2f}ms")
                    results.append((rich_meta, timings))

                except Exception as exc:
                    logger.warning(
                        f"Validation/parsing failure on Native HF result index {idx} "
                        f"for frame {frame_id}: {str(exc)}. Raw: {raw_out}"
                    )
                    continue

        except Exception as exc:
            logger.exception("Native HF execution failure on batch generation")
            raise RuntimeError(f"Batch generation failed: {str(exc)}")

        return results

