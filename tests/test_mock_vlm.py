import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from app.services.mock_vlm import MockVLMService
from app.services.vlm_factory import get_vlm_service

def test_mock_vlm_generate_metadata_batch():
    batch = [
        ("vid_f001", "vid", 1.0, Path("/dummy/path.jpg")),
        ("vid_f002", "vid", 2.0, Path("/dummy/path2.jpg")),
    ]
    
    results = asyncio.run(MockVLMService.generate_metadata_batch(batch))
    
    assert len(results) == 2
    
    meta1, timings1 = results[0]
    assert hasattr(meta1, "scene_type")
    assert hasattr(meta1, "objects")
    assert hasattr(meta1, "events")
    assert hasattr(meta1, "caption")
    
    assert timings1["vlm_ms"] == 0.0

def test_vlm_factory_returns_mock():
    with patch("app.services.vlm_factory.settings.MOCK_MODEL", True):
        service = get_vlm_service()
        assert service.__name__ == "MockVLMService"
        
    with patch("app.services.vlm_factory.settings.MOCK_MODEL", False):
        with pytest.raises(ValueError):
            with patch("app.services.vlm_factory.settings.VLM_ENGINE_TYPE", "unsupported"):
                get_vlm_service()

        with patch("app.services.vlm_factory.settings.VLM_ENGINE_TYPE", "native_hf"):
            service = get_vlm_service()
            assert service.__name__ == "NativeQwenTransformersService"

        with patch("app.services.vlm_factory.settings.VLM_ENGINE_TYPE", "native_vllm"):
            service = get_vlm_service()
            assert service.__name__ == "NativeQwenVLMService"
