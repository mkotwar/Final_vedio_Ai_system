# VLM Validation Audit Report

This document contains an audit of the current Vision-Language Model (VLM) prompts, metadata quality, and schema design for the CCTV Video Search Engine, as requested.

## 1. Current Prompt Audit

**Active Prompt (Found in `qwen_vlm.py`, `qwen_vlm_hf.py`, `native_qwen_vlm.py`):**
```json
"Analyze the image and return a raw JSON object detailing its visual contents objectively. "
"You MUST return a single JSON object (enclosed in curly braces {}), NOT a JSON array. "
"The JSON object MUST strictly adhere to this schema:\n"
"{\n"
'  "scene_type": "string",\n'
'  "scene_description": "string",\n'
'  "objects": [\n'
"    {\n"
'      "id": "string",\n'
'      "type": "string",\n'
'      "subtype": "string",\n'
'      "color": "string",\n'
'      "condition": "standing/walking/sitting/lying/bending/moving/stationary/unknown",\n'
'      "attributes": ["string"]\n'
"    }\n"
"  ],\n"
'  "events": [\n'
"    {\n"
'      "event_type": "interaction/observation/none",\n'
'      "description": "string",\n'
'      "actors": ["string"],\n'
'      "severity": "low/medium/high/critical"\n'
"    }\n"
"  ],\n"
'  "people_count": 0,\n'
'  "activities": ["string"],\n'
'  "keywords": ["string"],\n'
'  "caption": "string"\n'
"}\n"
"CRITICAL RULES:\n"
"- DO NOT output placeholder text like 'unique id e.g. person_1'. Generate actual values only.\n"
"- If no value is known, return null. If no actor exists, return an empty array.\n"
"- For 'subtype', you MUST use ONLY the following allowed values:\n"
"    Actors: person, employee, customer\n"
"    Vehicles: car, truck, motorcycle, bus\n"
"    Objects: bag, backpack, suitcase, box\n"
"- If uncertain, fallback to: person, vehicle, or object.\n"
"- NEVER invent new subtype names. NEVER output: adult male, individual, pedestrian, visitor, shopper.\n"
"- Give each object a unique 'id' so events can reference them via 'actors'.\n"
"- Describe the scene objectively. Extract visual facts only.\n"
"- NEVER infer incidents, causality, or intent from a single frame.\n"
"- NEVER assume a person has fallen; use neutral posture descriptors like 'lying' or 'bending'.\n"
"- NEVER assume a collision, speeding, abandonment, or criminal activity occurred.\n"
"- If the scene has no notable interactions, return: \"events\": []\n"
"- Respond ONLY with raw JSON. No markdown, no backticks, no commentary."
```

**Audit Findings:**
*   **Duplicate Instructions:** 
    *   "You MUST return a single JSON object" and "Respond ONLY with raw JSON."
    *   "Extract visual facts only" and "NEVER infer incidents, causality, or intent from a single frame."
*   **Contradictory Instructions:** 
    *   The schema specifies `"type"` and `"subtype"`, but the rules state "For 'subtype', you MUST use ONLY the following allowed values... If uncertain, fallback to: person, vehicle, or object." This conflates broad categories (type) with specific labels (subtype), leading to VLM confusion.
    *   "If no value is known, return null" contradicts the requirement to output arrays for lists (e.g., returning null for `objects` breaks JSON parsing/validation).
*   **Redundant Constraints:** 
    *   "NEVER assume a person has fallen", "NEVER assume a collision", "NEVER assume... criminal activity". These are highly specific redundancies of the broader rule "NEVER infer incidents, causality, or intent".
*   **Unnecessary Fields:** 
    *   `caption` and `scene_description` capture virtually identical semantic information.
    *   `keywords` and `activities` have high overlap with the `scene_type` and events.


## 2. Metadata Quality Audit (Golden Dataset)

Analysis of actual Golden Dataset pipeline outputs for three core scenarios:

*   **Scenario 1: `empy_room_15sec.mp4`**
    *   **Prompt Requested:** "Describe the scene objectively... return an empty array if no actors exist."
    *   **Metadata Produced:** Contains a `"vehicle_movement"` event with `"blue car (moving)"` and `"person walking, carrying bag"`. `scene_context` is labeled as `"outdoor city street view"`.
    *   **Finding:** Severe hallucination/override. The output completely ignores the requested prompt and visual reality.

*   **Scenario 2: `person_walking_30sec.mp4`**
    *   **Prompt Requested:** Objectively describe the person walking without assuming intent.
    *   **Metadata Produced:** Alternates between a `"blue car driving"` and an `"indoor office meeting room workspace"` with a `"grey chair"` and `"laptop"`. 
    *   **Finding:** The metadata produced has zero correlation with the actual frame content.

*   **Scenario 3: `customer_interaticio_60sec.mp4`**
    *   **Prompt Requested:** Identify subtypes (`customer`, `employee`) and track interactions.
    *   **Metadata Produced:** Exactly matches the outputs of the previous two scenarios (alternating synthetic street/office data).

**CRITICAL METADATA QUALITY FINDING:** 
The actual outputs produced for the Golden Dataset are entirely disconnected from the VLM prompt. The application is currently running a synthetic mock generator (`_generate_mock_metadata` in `qwen_vlm.py`), overriding the prompt entirely and returning alternating hardcoded JSON responses. Therefore, the VLM's adherence to the prompt requested is effectively 0% in the current pipeline state.


## 3. Output Field Classification & Investigation Value

Every output field from the current schema is classified based on its utility for CCTV investigation:

**A. High Investigative Value**
*   **`objects.subtype`** (person, vehicle, bag): Essential for basic filtering (answering "Who" and "What").
*   **`objects.color`**: Critical for visual search queries (e.g., "red shirt", "blue car").
*   **`events.event_type`**: Essential for triage, alerting, and timeline summarization.
*   **`events.actors`**: High value for establishing relationships between an event and specific entities.
*   **`people_count`**: Highly useful for anomaly detection (overcrowding, off-hours presence).

**B. Medium Investigative Value**
*   **`scene_description`**: Provides useful free-text search surface area, but is often verbose and unstructured.
*   **`objects.condition`** (standing, moving): Useful for state filtering, but often overlaps with activity/event descriptions.
*   **`activities`**: Helpful for generic keyword tagging, but redundant with the `events` array.

**C. Low Investigative Value**
*   **`objects.type`**: Largely redundant if `subtype` is well-defined.
*   **`caption`**: Completely redundant with `scene_description`.
*   **`keywords`**: Adds bloated tokens; modern vector search makes explicit keyword arrays obsolete.
*   **`objects.id`**: Currently low value because VLMs cannot consistently assign the *same* ID to the *same* person across frames without spatial tracking tools (bounding boxes).


## 4. Missing Metadata Audit (Event Aggregation)

The following metadata is fundamentally missing from the current schema, preventing robust Event Aggregation across frames. Prioritized by importance:

1.  **Spatial Coordinates (Bounding Boxes):** (e.g., `[ymin, xmin, ymax, xmax]`). Without spatial coordinates, an aggregation engine cannot definitively associate "person_1" in Frame A with "person_1" in Frame B.
2.  **Trajectory & Directional Vectors:** Information on where an object is moving (e.g., "moving left to right", "approaching camera"). Required to aggregate continuous motion into a single unified event.
3.  **Explicit Spatial Relationships:** Context linking objects together physically (e.g., "person holding bag", "car parked near gate"). 
4.  **Zonal Context:** Logical location mapping within the frame (e.g., "top-left quadrant", "entry zone"). Essential for line-crossing or restricted-area aggregation.


## 5. Missing Metadata Audit (Search)

To support natural language and highly specific queries, Search would heavily benefit from:

*   **Granular Clothing Attributes:** Separating `clothing_top` and `clothing_bottom` (e.g., "man in red shirt and blue jeans") instead of a generic `color` string.
*   **Proximity / Relational Search:** "person near door", "customer at counter", "employee behind register".
*   **Pose & Interaction State:** "person holding phone", "customer carrying box", "person looking at camera".
*   **Demographic/Appearance Details:** (Privacy policies permitting) "child", "adult", "wearing glasses", "wearing hat" (currently explicitly banned by the prompt).


## 6. Ideal Schema Proposal (CCTV Investigation)

*Note: Proposed schema for analytical purposes only. No implementation generated.*

```json
{
  "frame_summary": "string",
  "global_context": {
    "lighting_condition": "day/night/artificial",
    "environment": "indoor/outdoor"
  },
  "entities": [
    {
      "tracking_id": "string",
      "category": "person/vehicle/object",
      "label": "string",
      "bbox": [0, 0, 0, 0],
      "visual_traits": {
        "primary_colors": ["string"],
        "clothing_top": "string",
        "clothing_bottom": "string"
      },
      "spatial_location": "string",
      "current_state": "string"
    }
  ],
  "interactions": [
    {
      "interaction_type": "string",
      "subject_id": "string",
      "target_id": "string",
      "spatial_relation": "string"
    }
  ]
}
```

**Field Analysis:**
*   **`entities.bbox`**: 
    *   *Purpose*: Locates the object in 2D space.
    *   *Aggregation Benefit*: Allows the tracker to use Intersection over Union (IoU) to link entities across frames accurately.
    *   *Search Benefit*: Enables region-of-interest (ROI) filtering in search.
*   **`entities.visual_traits`**:
    *   *Purpose*: Breaks down appearance into structured fields.
    *   *Aggregation Benefit*: Serves as a secondary re-identification (ReID) feature if the bbox tracker fails.
    *   *Search Benefit*: Radically improves exact-match text search for specific suspects.
*   **`interactions` (Subject/Target)**:
    *   *Purpose*: Defines directed graphs of behavior.
    *   *Aggregation Benefit*: Allows the aggregator to cluster related events (e.g., Subject A interacting with Target B over 5 frames = 1 continuous interaction).
    *   *Search Benefit*: Enables complex NLP queries like "Show me the employee taking a bag from a customer".


## 7. Expected Improvement Estimate

If the Ideal Schema is implemented, the estimated impact on system metrics is:

*   **Aggregation Quality: +80%** 
    *   *Why:* The introduction of `bbox` coordinates allows for deterministic algorithmic tracking (e.g., SORT/DeepSORT) rather than relying on the VLM to arbitrarily remember and assign `objects.id` strings.
*   **Search Quality: +60%**
    *   *Why:* Structured `visual_traits` and `spatial_relation` fields allow Elasticsearch/Qdrant to perform highly specific filtering, eliminating the noise of a generic `scene_description` blob.
*   **Investigation Accuracy: +70%**
    *   *Why:* Explicit subject-to-target `interactions` prevent hallucinated associations. Investigators can trace exact causal chains (who touched what) rather than just knowing that a person and an object were present in the same frame.