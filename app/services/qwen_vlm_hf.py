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
from app.core.utils import calculate_time_snippet, format_timestamp_human
from app.services.ocr import OCRService
from app.services.activity_recovery import ActivityRecoveryService
from app.services.qwen_vlm import QwenVLMService
from qwen_vl_utils import process_vision_info

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
        
        # CRITICAL: 4-bit quantization reduces 14.5GB VRAM footprint to ~5GB, 
        # eliminating the PCIe thrashing OOM slowdown on RTX 5070 Ti.
        # Use device_map={"": "cuda:0"} to strictly bind to GPU and avoid CPU offload float32 bugs.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        cls._processor = AutoProcessor.from_pretrained(
            model_id, 
            min_pixels=256*28*28, 
            max_pixels=512*28*28
        )
        
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
        
        batch_start_time = time.perf_counter()
        batch_size = len(image_paths)
        
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = cls._model.generate(**inputs, max_new_tokens=settings.QWEN_MAX_NEW_TOKENS)
        t1 = time.perf_counter()
        generate_ms = (t1 - t0) * 1000.0
        
        mem_alloc_after = torch.cuda.memory_allocated() / (1024**3)
        mem_res_after = torch.cuda.memory_reserved() / (1024**3)
        
        logger.info(f"Generate Time: {generate_ms:.2f} ms")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"max_new_tokens: {settings.QWEN_MAX_NEW_TOKENS}")
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
            '      "severity": "low",\n'
            '      "timestamp_hint": "approximate time in clip if detectable, else empty string"\n'
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
                    
                # Save raw JSON before repair to inspect for truncation or bloat
                dump_dir = PROJECT_ROOT / "data" / "logs"
                dump_dir.mkdir(parents=True, exist_ok=True)
                with open(dump_dir / f"raw_qwen_output_{frame_id}.json", "w", encoding="utf-8") as f:
                    f.write(raw_out)
                    
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
