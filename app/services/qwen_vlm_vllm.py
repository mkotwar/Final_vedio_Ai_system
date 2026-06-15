import os
from pathlib import Path
from typing import List
from loguru import logger

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class NativeQwenVLLMService:
    _llm = None
    
    @classmethod
    def load_model(cls):
        if cls._llm is not None:
            return
            
        if LLM is None:
            raise ImportError("vLLM is not installed. Please install vllm.")
            
        logger.info("Loading Qwen2.5-VL-7B-Instruct via vLLM...")
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # vLLM automatically uses FlashAttention and PagedAttention if supported
        cls._llm = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
        )
        logger.info("vLLM Model loaded.")
        
    @classmethod
    def generate_batch(cls, image_paths: List[Path], prompt: str) -> List[str]:
        cls.load_model()
        
        # Format the inputs for vLLM
        messages_batch = []
        for path in image_paths:
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(msg)
            
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )
        
        logger.info(f"--- TRUE BATCHING AUDIT ---")
        logger.info(f"Submitting {len(messages_batch)} requests to vLLM engine...")
        
        # vLLM handles batching dynamically via continuous batching
        outputs = cls._llm.chat(messages=messages_batch, sampling_params=sampling_params)
        
        results = [output.outputs[0].text for output in outputs]
        return results
