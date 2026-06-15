import os
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from loguru import logger
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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
        
        # We use bfloat16 for efficiency and flash_attention_2 if available
        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        cls._processor = AutoProcessor.from_pretrained(model_id)
        
        # Add startup logging
        logger.info(f"Active VLM Backend: native_hf")
        is_cuda = torch.cuda.is_available()
        logger.info(f"CUDA Available: {is_cuda}")
        if is_cuda:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        logger.info(f"Model Loaded: {model_id}")
        
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
        
        # Print shape to prove true batching (Task 6)
        logger.info(f"--- TRUE BATCHING AUDIT ---")
        logger.info(f"Input ids shape: {inputs.input_ids.shape}")
        if 'pixel_values' in inputs:
            logger.info(f"Pixel values shape: {inputs.pixel_values.shape}")
        
        # Generate
        generated_ids = cls._model.generate(**inputs, max_new_tokens=256)
        
        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = cls._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts
