from loguru import logger
from typing import Any
from app.core.config import settings
from app.services.qwen_vlm import QwenVLMService

def get_vlm_service() -> Any:
    """
    Factory function to retrieve the active VLM service based on settings.
    It lazily loads the required service to prevent initializing heavy 
    models like vLLM if they are not active.
    """
    if settings.VLM_ENGINE_TYPE == "native_vllm":
        from app.services.native_qwen_vlm import NativeQwenVLMService
        return NativeQwenVLMService
        
    elif settings.VLM_ENGINE_TYPE == "native_hf":
        from app.services.qwen_vlm_hf import NativeQwenTransformersService
        return NativeQwenTransformersService
        
    elif settings.VLM_ENGINE_TYPE == "ollama":
        return QwenVLMService
        
    else:
        logger.warning(f"Unknown VLM_ENGINE_TYPE '{settings.VLM_ENGINE_TYPE}', falling back to Ollama.")
        return QwenVLMService
