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