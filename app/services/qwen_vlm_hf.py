import os
import torch
import time
import asyncio
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from app.core.config import settings, PROJECT_ROOT
from app.schemas.frame import FrameRichMetadata
from app.core.utils import calculate_time_snippet, format_timestamp_human
from app.services.ocr import OCRService
from app.services.activity_recovery import ActivityRecoveryService
from app.services.qwen_vlm import QwenVLMService
from qwen_vl_utils import process_vision_info

class NativeQwenTransformersService:
    _model = None
    _processor = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Returns the health status and engine configuration."""
        is_cuda_avail = torch.cuda.is_available()
        gpu_memory = 0.0
        if is_cuda_avail:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "engine_type": "native_hf",
            "model_loaded": "Qwen/Qwen2.5-VL-7B-Instruct" if cls._model else "None",
            "gpu_memory": f"{gpu_memory:.2f} GB",
            "cuda_available": is_cuda_avail,
            "status": "healthy" if cls._model else "uninitialized"
        }

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return
            
        logger.info(f"Loading Qwen2.5-VL-7B-Instruct via Transformers on {cls._device}...")
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # CRITICAL: Do NOT use device_map="auto".
        # device_map="auto" activates accelerate dispatch hooks that offload some layers
        # to CPU. CPU-offloaded layers compute activations in float32. When those float32
        # activations are passed to GPU-resident bfloat16 layers (e.g. q_proj), PyTorch
        # raises: "RuntimeError: mat1 and mat2 must have the same dtype, but got Float
        # and BFloat16". The RTX 5070 Ti has 16GB VRAM — the 7B model in bf16 (~14.5GB)
        # fits entirely on GPU. Using explicit .to(device) avoids all accelerate hooks.
        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(cls._device)
        cls._processor = AutoProcessor.from_pretrained(model_id)
        
        # Add startup logging
        logger.info(f"Active VLM Backend: native_hf")
        is_cuda = torch.cuda.is_available()
        logger.info(f"CUDA Available: {is_cuda}")
        if is_cuda:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            logger.info(f"VRAM used after load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        logger.info(f"Model Loaded: {model_id}")
        logger.info(f"Model dtype: {cls._model.dtype}")
        logger.info(f"First param dtype: {next(cls._model.parameters()).dtype}")
        logger.info(f"First param device: {next(cls._model.parameters()).device}")
        if hasattr(cls._model, 'hf_device_map'):
            logger.info(f"Device map: {cls._model.hf_device_map}")
        else:
            logger.info("No accelerate device map (all on single device). GOOD.")
        
    @classmethod
    def generate_batch(cls, image_paths: List[Path], prompt: str) -> List[str]:
        cls.load_model()
        
        messages_batch = []
        for path in image_paths:
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
            
        # Prepare inputs
        texts = [cls._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = cls._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(cls._device)
        
        # Batching audit logging
        logger.info(f"--- TRUE BATCHING AUDIT ---")
        logger.info(f"Input ids shape: {inputs.input_ids.shape}")
        logger.info(f"Input ids dtype: {inputs.input_ids.dtype}")
        if 'pixel_values' in inputs:
            logger.info(f"Pixel values shape: {inputs.pixel_values.shape}")
            logger.info(f"Pixel values dtype: {inputs.pixel_values.dtype}")
        if 'attention_mask' in inputs:
            logger.info(f"Attention mask dtype: {inputs.attention_mask.dtype}")
        if 'image_grid_thw' in inputs:
            logger.info(f"Image grid thw dtype: {inputs.image_grid_thw.dtype}")
        logger.info(f"Model dtype: {cls._model.dtype}")
            
        # Generate with autocast to handle any intermediate float32 computations
        # (e.g. RMSNorm internally uses float32 for numerical stability then casts back)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = cls._model.generate(**inputs, max_new_tokens=256)
        
        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = cls._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts

    @classmethod
    async def _async_hf_generate(cls, image_paths: List[Path], prompt: str) -> List[str]:
        """Wrapper for HF generate to keep the async interface without blocking."""
        def run_hf():
            return cls.generate_batch(image_paths, prompt)
            
        return await asyncio.to_thread(run_hf)

    @classmethod
    async def generate_metadata_batch(
        cls, batch_frames: List[Tuple[str, str, float, Path]]
    ) -> List[Tuple[FrameRichMetadata, Dict[str, float]]]:
        """Runs batch image inference on Qwen2.5-VL to extract structured metadata using Native HF."""
        cls.load_model()

        if settings.MOCK_MODEL:
            # Fallback to Ollama mock generation
            return await QwenVLMService.generate_metadata_batch(batch_frames)

        results: List[Tuple[FrameRichMetadata, Dict[str, float]]] = []

        prompt_guidelines = (
            "Analyze the image and return a raw JSON object detailing its contents. "
            "You MUST return a single JSON object (enclosed in curly braces {}), NOT a JSON array. "
            "The JSON object MUST strictly adhere to this schema:\n"
            "{\n"
            '  "scene_type": "indoor/outdoor description",\n'
            '  "scene_description": "detailed text describing environment",\n'
            '  "objects": [\n'
            "    {\n"
            '      "id": "unique id e.g. car_1, person_2",\n'
            '      "type": "object category",\n'
            '      "subtype": "specific type",\n'
            '      "color": "dominant color",\n'
            '      "condition": "normal/damaged/displaced/moving/stationary/fallen",\n'
            '      "attributes": ["list of describing attributes"]\n'
            "    }\n"
            "  ],\n"
            '  "events": [\n'
            "    {\n"
            '      "event_type": "collision/intrusion/fall/fire/fight/abandonment/speeding/trespassing/none",\n'
            '      "description": "precise sentence: what happened, who/what was involved, outcome",\n'
            '      "actors": ["object ids involved e.g. car_1, car_2"],\n'
            '      "severity": "low/medium/high/critical",\n'
            '      "timestamp_hint": "approximate time in clip if detectable, else empty string"\n'
            "    }\n"
            "  ],\n"
            '  "people_count": 0,\n'
            '  "activities": ["list of ongoing activities"],\n'
            '  "keywords": ["search tag keywords"],\n'
            '  "caption": "incident-report style caption — describe WHAT HAPPENED. '
            'If objects are damaged, displaced, or in abnormal positions, state the likely cause."\n'
            "}\n"
            "CRITICAL RULES:\n"
            "- Give each object a unique 'id' so events can reference them via 'actors'.\n"
            "- 'events' MUST capture INTERACTIONS and INCIDENTS, not just presence of objects.\n"
            "- Damaged vehicle + nearby vehicle = collision event. Flag it.\n"
            "- Fallen person = fall event. Person in restricted zone = intrusion event.\n"
            "- Unattended bag/object = abandonment event. Unusual speed = speeding event.\n"
            "- If the scene is fully normal with zero incidents, return: \"events\": []\n"
            "- The 'caption' must read like a security incident report, not a photo description.\n"
            "- Respond ONLY with raw JSON. No markdown, no backticks, no commentary."
        )

        try:
            logger.debug(f"Preparing batch of {len(batch_frames)} images for Native HF...")
            
            image_paths = [path for _, _, _, path in batch_frames]

            vlm_start = time.perf_counter()
            vlm_outputs = await cls._async_hf_generate(image_paths, prompt_guidelines)
            vlm_ms_avg = ((time.perf_counter() - vlm_start) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            # Fully parallel OCR batch extraction
            ocr_start_global = time.perf_counter()
            
            async def timed_ocr(path, frame_id):
                res = await asyncio.to_thread(OCRService.extract_text, path)
                return res

            ocr_tasks = [timed_ocr(path, batch_frames[idx][0]) for idx, path in enumerate(image_paths)]
            ocr_results = await asyncio.gather(*ocr_tasks)
            ocr_duration_ms_avg = ((time.perf_counter() - ocr_start_global) * 1000.0) / len(batch_frames) if batch_frames else 0.0

            for idx, raw_out in enumerate(vlm_outputs):
                frame_id, video_id, ts, path = batch_frames[idx]
                if not raw_out:
                    logger.warning(f"Empty response from Native HF for {frame_id}")
                    continue
                    
                try:
                    repair_start = time.perf_counter()
                    cleaned_out = QwenVLMService._clean_json_response(raw_out)
                    parsed = json.loads(cleaned_out)
                    parsed = QwenVLMService._normalize_metadata_dict(parsed)
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

                    val_start = time.perf_counter()
                    rich_meta = FrameRichMetadata(**parsed)
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
