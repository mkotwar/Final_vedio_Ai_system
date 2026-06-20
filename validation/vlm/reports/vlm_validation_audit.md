# VLM OUTPUT VALIDATION AUDIT
**VLM Engine**: native_hf
**Model**: Qwen/Qwen2.5-VL-7B-Instruct

## Video: customer_interaticio_60sec
### Frame @ 5.0s
* **Latency:** 6091.88 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['employee', 'office desk', 'desktop computer', 'person', 'furniture', 'electronics']
* **Detected Events Dump:** []

### Frame @ 15.0s
* **Latency:** 6091.88 ms
* **Expected Objects:** ['person'] | Found: 1/1
* **Expected Events:** ['customer_approach'] | Found: 0/1
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['customer', 'employee', 'person', 'person']
* **Detected Events Dump:** []

### Frame @ 35.0s
* **Latency:** 6091.88 ms
* **Expected Objects:** ['person'] | Found: 1/1
* **Expected Events:** ['customer_interaction'] | Found: 0/1
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['customer', 'employee', 'employee', 'person', 'person', 'person']
* **Detected Events Dump:** []

## Video: empy_room_15sec
### Frame @ 1.0s
* **Latency:** 5665.02 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['desk', 'office chair', 'office chair', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

### Frame @ 7.0s
* **Latency:** 5665.02 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['desk', 'office chair', 'office chair', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

### Frame @ 14.0s
* **Latency:** 5665.02 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['desk', 'office chair', 'cabinet', 'cabinet', 'furniture', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

## Video: person_walking_30sec
### Frame @ 5.0s
* **Latency:** 5848.15 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['office chair', 'office desk', 'cabinet', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

### Frame @ 18.0s
* **Latency:** 5848.15 ms
* **Expected Objects:** ['person'] | Found: 1/1
* **Expected Events:** ['person_walk'] | Found: 0/1
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['individual', 'office chair', 'office desk', 'cabinet', 'person', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

### Frame @ 28.0s
* **Latency:** 5848.15 ms
* **Expected Objects:** [] | Found: 0/0
* **Expected Events:** [] | Found: 0/0
* **Hallucinations Detected:** None (PASS)
* **Actor Consistency:** PASS
* **Detected Objects Dump:** ['desk', 'office chair', 'cabinet', 'cabinet', 'furniture', 'furniture', 'furniture', 'furniture']
* **Detected Events Dump:** []

## OVERALL METRICS
* **Total Frames Analyzed:** 9
* **Object Recall:** 100.0%
* **Event Recall:** 0.0%
* **Hallucination Rate (Frames with false events):** 0.0%
* **Actor Consistency Failure Rate:** 0.0%
* **Average Latency:** 5868.35 ms

---

# ARCHITECTURAL READINESS AUDIT

## 1. Raw Output Audit
A sample of ~20 raw frame outputs (from recent metadata executions like `094e564d_frames.json`) was manually reviewed. 
* **Positive Findings:** For most frames, Qwen effectively maps objects to conditions (`stationary`, `moving`) and highly descriptive attributes (`carrying backpack`, `wearing jeans`, `walking towards building`).
* **Critical Finding:** In some frames (e.g., `f0015`), Qwen completely **hallucinates the prompt schema itself** rather than analyzing the image, literally outputting: `"id": "unique id e.g. car_1, person_2"`, `"type": "object category"`. 

## 2. Object Consistency Audit
* **Finding:** Object labels and subtypes are highly volatile.
* **Example from Data:** 
  - Frame 1: `adult male`
  - Frame 2: `pedestrian`, `scooter rider`, `security guard`
  - Frame 3: `motorcyclist`
* **Score:** **POOR**. Because there are no bounding boxes, the Event Aggregator currently relies on text similarity to track actors across frames. Oscillation between `adult male` and `pedestrian` will break semantic tracking and create duplicated actors in the final event record.

## 3. Attribute Coverage Audit
* **Finding:** The attributes array is the strongest component of the current VLM output. 
* **Coverage Score:** **EXCELLENT**. 
* **Details:** Qwen successfully infers states like `carrying backpack`, `wearing helmet`, `conversing`, and `standing near fence`. These are perfectly detailed enough for the Event Aggregator to construct meaningful narratives.

## 4. Event Aggregation Readiness Assessment
Assume Event Aggregation receives ONLY: `objects`, `attributes`, `captions`, `timestamps`.
* `person_walk` -> **READY** (Attributes frequently report "walking" and "moving").
* `customer_approach` -> **PARTIALLY READY** (The model struggles to consistently identify "customer" vs "pedestrian" without context).
* `customer_interaction` -> **PARTIALLY READY** (Frame 4 successfully detected "conversing", but actor mapping might break).
* `person_enter` -> **READY** (Attributes explicitly captured "entering building" in Frame 2).
* `person_exit` -> **READY** (Similar to enter).

## 5. Schema Audit
| Field | Produced By Qwen? | Used By Aggregator? | Useful? | Missing? |
| :--- | :--- | :--- | :--- | :--- |
| `scene_type` | Yes | Yes | Yes | - |
| `objects.id` | Yes (Volatile) | Yes | **Broken** | Bounding boxes/Coordinates `[x1, y1, x2, y2]` are missing to enforce spatial tracking. |
| `objects.attributes` | Yes | Yes | Yes | - |
| `events` | Empty | No | No | - |
| `caption` | Yes | Yes | Yes | - |

## 6. Caption Quality Audit
* **Objective?** Mostly yes ("Normal activity observed at Gate 1 Entrance Area...").
* **Too generic?** Occasionally it defaults to "No description available" when it gets confused. 
* **Useful context?** Very high. The OCR injection ("Gate 1 Entrance Area") grounds the caption excellently.

## 7. Aggregation Simulation
1. Qwen returns Frame 1: `person_1 (moving, walking)`.
2. Qwen returns Frame 2: `person_1 (walking)`.
3. Event Aggregator groups Frame 1 and Frame 2 because timestamps are < 10s apart.
4. It deduplicates `person_1` based on text overlap.
5. **Final Event:** A pedestrian was observed walking near the gate.
*Simulation Result: Meaningful events CAN be produced today, but actor-tracking will occasionally split one person into two due to label volatility.*

## 8. Readiness Decision
**B. Current Qwen outputs are partially sufficient.**
* **Confidence Level:** **High**. 
* **Justification:** The attribute and activity data is rich enough to power the aggregator, but the occasional schema-hallucination (regurgitating the prompt template) and the volatile object subtyping will cause the Event Aggregator to generate slightly fragmented or duplicate event chains. 

## 9. Proposed Next Phase Improvements (NOT IMPLEMENTED)
1. **Schema Regurgitation Fix:** Add an explicit negative constraint to the VLM prompt: `"NEVER output the example placeholder text (e.g. 'unique id e.g. car_1'). Evaluate the image."` (Improves: Hallucination Rate, Aggregation Readiness).
2. **Subtype Constraints:** Force the VLM to use a predefined list for `subtype` (e.g., `pedestrian, guard, employee, customer`) rather than free-text, preventing oscillation between `adult male` and `pedestrian`. (Improves: Actor Consistency, Aggregation Readiness).
3. **Bounding Boxes (Future Phase):** Request `[x_min, y_min, x_max, y_max]` for all objects to allow spatial IoU tracking across frames, entirely replacing text-similarity actor tracking. (Improves: Everything).