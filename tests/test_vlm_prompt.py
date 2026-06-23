from pathlib import Path

from app.services.vlm_prompt import VLM_FRAME_METADATA_PROMPT


def test_vlm_prompt_contains_required_contract_fields():
    required_fields = [
        "scene_type",
        "scene_description",
        "caption",
        "people_count",
        "objects",
        "activities",
        "relationships",
        "location_context",
        "events",
        "keywords",
        "ocr",
    ]

    for field in required_fields:
        assert f'"{field}"' in VLM_FRAME_METADATA_PROMPT

    assert "Return ONLY one valid JSON object" in VLM_FRAME_METADATA_PROMPT
    assert "Do not infer intent" in VLM_FRAME_METADATA_PROMPT
    assert "crossing road" in VLM_FRAME_METADATA_PROMPT


def test_native_backends_use_shared_vlm_prompt():
    service_dir = Path(__file__).resolve().parents[1] / "app" / "services"

    for filename in ["qwen_vlm_hf.py", "native_qwen_vlm.py"]:
        source = (service_dir / filename).read_text(encoding="utf-8")
        assert "from app.services.vlm_prompt import VLM_FRAME_METADATA_PROMPT" in source
        assert "prompt_guidelines = VLM_FRAME_METADATA_PROMPT" in source
        assert "prompt_guidelines = \"\"\"" not in source
