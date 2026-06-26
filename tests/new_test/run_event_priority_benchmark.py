import asyncio
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

if str(os.environ.get("DEBUG", "")).lower() not in {"", "0", "1", "true", "false", "yes", "no", "on", "off"}:
    os.environ["DEBUG"] = "true"

DEFAULT_CASE_ROOT = PROJECT_ROOT_PATH / "tests" / "new_test" / "event_priority_benchmark"
os.environ.setdefault("BENCHMARK_CASE_ROOT", str(DEFAULT_CASE_ROOT))
os.environ.setdefault("BENCHMARK_MODES", "candidate_only")
os.environ.setdefault("BENCHMARK_VARIANTS", "strip_eventsfirst_tokens512_batch4")

from app.services import vlm_prompt as vlm_prompt_module


EVENT_PRIORITY_VLM_FRAME_METADATA_PROMPT = """
You analyze surveillance video frames or temporal multi-image event inputs.

Return ONLY one valid JSON object. No markdown. No extra text.

Priority:
1. Describe suspicious or notable behavior first.
2. Fill relationships and events before detailed object attributes.
3. Keep object lists short and relevant.

Rules:
- Report only visible evidence.
- Use objective incident labels such as possible_theft, possible_robbery, abandoned_object, object_removed, weapon_visible.
- If a person leaves an item behind, prefer abandoned_object.
- If someone appears to take or conceal property, prefer possible_theft.
- If force, threat, weapon, aggressive confrontation, or coordinated intimidation is visible, prefer possible_robbery.
- Mention masks, hoods, helmets, face covering, weapons, carried bags, dropped objects, and objects left on floors/counters.
- Ignore static furniture unless directly relevant to an interaction.
- Do not enumerate more than 4 most relevant objects unless absolutely necessary.

Schema and order:
{
  "scene_type": "street|entrance|parking_area|corridor|office|shop|warehouse|indoor|outdoor|unknown",
  "scene_description": "max 15 words",
  "caption": "one short objective sentence",
  "people_count": 0,
  "relationships": [
    {
      "subject_id": "person_1",
      "target_id": "bag_1",
      "relation": "holding|carrying|placing|dropping|picking_up|approaching|following|threatening|interacting_with|near|entering|exiting|walking_with|standing_with"
    }
  ],
  "events": [
    {
      "event_type": "normal_activity|group_activity|person_object_interaction|intrusion|unauthorized_entry|loitering|abandoned_object|object_removed|possible_theft|possible_robbery|weapon_visible|physical_altercation|crowd_formation",
      "description": "objective evidence-based description",
      "actors": ["person_1", "bag_1"],
      "severity": "low|medium|high|critical"
    }
  ],
  "objects": [
    {
      "id": "person_1",
      "type": "person|vehicle|object|animal",
      "subtype": "man|woman|child|guard|customer|car|truck|bag|backpack|box|phone|weapon|other|unknown",
      "color": "brown|red|orange|yellow|green|cyan|blue|purple|pink|white|grey|black|unknown",
      "condition": "standing|walking|running|bending|moving|stationary|parked|unknown",
      "headwear": "mask|hood|helmet|cap|none|unknown",
      "carried_object": "bag|backpack|box|phone|weapon|none|unknown",
      "attributes": ["short visible attributes only"]
    }
  ],
  "activities": [
    "walking",
    "running",
    "talking",
    "interacting",
    "holding",
    "carrying",
    "placing object",
    "dropping object",
    "picking up object",
    "entering",
    "exiting",
    "leaving"
  ],
  "keywords": ["short searchable tags"],
  "ocr": {
    "detected_text": ["clear text only"],
    "license_plates": ["clear plates only"]
  }
}

If no suspicious activity is visible, keep events empty but still complete the schema.
"""

vlm_prompt_module.SHARED_VLM_FRAME_METADATA_PROMPT = EVENT_PRIORITY_VLM_FRAME_METADATA_PROMPT
setattr(vlm_prompt_module, "VLM_FRAME_METADATA_PROMPT", EVENT_PRIORITY_VLM_FRAME_METADATA_PROMPT)

from tests.manual_benchmark_case import run_event_candidate_reasoning_benchmark as benchmark


benchmark.VLM_FRAME_METADATA_PROMPT = EVENT_PRIORITY_VLM_FRAME_METADATA_PROMPT

event_priority_variant = benchmark.BenchmarkVariant(
    name="strip_eventsfirst_tokens512_batch4",
    image_layout="strip",
    max_new_tokens=int(os.getenv("EVENT_PRIORITY_MAX_NEW_TOKENS", "512")),
    batch_size=int(os.getenv("EVENT_PRIORITY_BATCH_SIZE", "4")),
)

benchmark.BENCHMARK_VARIANTS = [event_priority_variant]
benchmark.BENCHMARK_VARIANT_BY_NAME = {event_priority_variant.name: event_priority_variant}
benchmark.DEFAULT_BENCHMARK_MODE_NAMES = ("candidate_only",)
benchmark.DEFAULT_BENCHMARK_VARIANT_NAMES = (event_priority_variant.name,)


if __name__ == "__main__":
    asyncio.run(benchmark.main())
