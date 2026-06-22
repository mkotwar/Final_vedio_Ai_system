# Production HF Prompt Audit & Gap Analysis

## 1. Current HF Prompt Schema
The currently deployed VLM prompt in `app/services/qwen_vlm_hf.py` requests the following schema from the model:
```json
{
  "scene_type": "string",
  "scene_description": "string",
  "objects": [
    {
      "id": "string",
      "type": "string",
      "subtype": "string",
      "color": "string",
      "condition": "standing/walking/sitting/lying/bending/moving/stationary/unknown",
      "attributes": ["string"]
    }
  ],
  "events": [
    {
      "event_type": "interaction/observation/none",
      "description": "string",
      "actors": ["string"],
      "severity": "low/medium/high/critical"
    }
  ],
  "people_count": 0,
  "activities": ["string"],
  "keywords": ["string"],
  "caption": "string"
}
```

## 2. Prompt V2 Schema
The previously evaluated Prompt V2 design explicitly requests the following richer structure:
```json
{
  "scene_type": "string",
  "scene_description": "string",
  "objects": [
    {
      "id": "string",
      "type": "string",
      "subtype": "string",
      "color": "string",
      "condition": "standing/walking/sitting/lying/bending/moving/stationary/unknown",
      "attributes": ["string"]
    }
  ],
  "location_context": [
    {
      "object_id": "string",
      "location": "string (e.g. near the door, in the center, left side)"
    }
  ],
  "relationships": [
    {
      "subject_id": "string",
      "target_id": "string",
      "relation": "string (e.g. holding, talking to, standing next to)"
    }
  ],
  "events": [
    {
      "event_type": "string",
      "description": "string",
      "actors": ["string"]
    }
  ],
  "activities": ["string (CHOOSE FROM: standing, walking, running, driving, working, talking, interacting, waiting, none)"],
  "people_count": 0
}
```

## 3. Schema Comparison Matrix
A direct comparison between the two schemas reveals key disparities:

| Field            | Current HF Prompt | Prompt V2 | Status  |
| ---------------- | ----------------- | --------- | ------- |
| `scene_type`     | YES               | YES       | MATCH   |
| `scene_description`| YES             | YES       | MATCH   |
| `objects`        | YES               | YES       | MATCH   |
| `events`         | YES               | YES       | MATCH   |
| `people_count`   | YES               | YES       | MATCH   |
| `activities`     | YES               | YES       | PARTIAL |
| `keywords`       | YES               | NO        | UNUSED  |
| `caption`        | YES               | NO        | UNUSED  |
| `relationships`  | NO                | YES       | MISSING |
| `location_context`| NO               | YES       | MISSING |

*(Note: `activities` is marked as PARTIAL because the Current HF Prompt allows unbound strings, whereas Prompt V2 explicitly constraints the vocabulary to a finite choice list.)*

## 4. Event Aggregator Dependency Matrix
Based on an audit of `app/services/event_aggregation.py`, the following VLM metadata fields are actively consumed (or conspicuously missing) during the event generation phase:

| Field            | Produced | Consumed |
| ---------------- | -------- | -------- |
| `caption`        | YES      | YES      |
| `scene_description`| YES    | YES      |
| `scene_type`     | YES      | YES      |
| `activities`     | YES      | YES      |
| `keywords`       | YES      | YES      |
| `objects`        | YES      | YES      |
| `events`         | YES      | YES      |
| `people_count`   | YES      | NO       |
| `relationships`  | NO       | NO       |
| `location_context`| NO       | NO       |

## 5. Bottleneck Analysis

### What metadata is missing from production HF output?
The production HF output entirely lacks **`relationships`** and **`location_context`**. In addition, the **`activities`** list is unbounded, leading to inconsistent verb usage which makes aggregation difficult.

### What metadata is required for high-quality Event Aggregation?
High-quality event aggregation demands highly structured physical interactions (`relationships`), spatial awareness (`location_context`), and a constrained taxonomy for verbs (`activities`). 

### Is the current Event Aggregator bottlenecked by:
**Both.**
- **Prompt Design:** The aggregator cannot consume `relationships` or `location_context` because Qwen is simply not instructed to generate them. 
- **Aggregator Logic:** The aggregator itself contains hardcoded placeholders specifically because it lacks this structured context (e.g., passing `"{location_placeholder}"` into `_build_narrative_sentence`). It resorts to fragile regex heuristics (`"robbery" in text`) to compensate for the lack of actual relationship modeling.

## 6. Final Recommendation

### Decision B
```text
HF prompt is missing critical metadata.
Prompt upgrade must occur before Event Aggregator work.
```
