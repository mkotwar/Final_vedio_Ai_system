# Prompt V1 vs Prompt V2 Evaluation Report

## Framework Execution Overview
**Frames Evaluated**: 9
**Model Engine**: Ollama (Qwen2.5-VL-7B)
**Metric Methodology**:
- Objects = 1 pt each
- Activities = 1 pt each
- Relationships = 2 pts each
- Location Context = 2 pts each

## Frame Comparison Breakdown

### Frame: empy_room_ts1.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 0 | 0 |
| Activities Recall | 0 | 0 |
| Relationship Recall | 0 | 0 |
| Location Context Recall | 0 | 0 |
| **Aggregation Readiness Score** | **0** | **0** |

*Note: Prompt V1 generated hallucinated vehicles initially, but under strict temperature=0 evaluation, both models correctly identified an empty room.*

### Frame: empy_room_ts7.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 0 | 0 |
| Activities Recall | 0 | 0 |
| Relationship Recall | 0 | 0 |
| Location Context Recall | 0 | 0 |
| **Aggregation Readiness Score** | **0** | **0** |

### Frame: empy_room_ts14.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 0 | 0 |
| Activities Recall | 0 | 0 |
| Relationship Recall | 0 | 0 |
| Location Context Recall | 0 | 0 |
| **Aggregation Readiness Score** | **0** | **0** |

### Frame: person_walk_ts5.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 1 | 1 |
| Activities Recall | 1 | 1 |
| Relationship Recall | 0 | 0 |
| Location Context Recall | 0 | 1 |
| **Aggregation Readiness Score** | **2** | **4** |

### Frame: person_walk_ts18.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 1 | 1 |
| Activities Recall | 1 | 1 |
| Relationship Recall | 0 | 0 |
| Location Context Recall | 0 | 1 |
| **Aggregation Readiness Score** | **2** | **4** |

### Frame: person_walk_ts28.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 2 | 2 |
| Activities Recall | 1 | 1 |
| Relationship Recall | 0 | 1 |
| Location Context Recall | 0 | 2 |
| **Aggregation Readiness Score** | **3** | **9** |

### Frame: customer_int_ts5.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 3 | 3 |
| Activities Recall | 2 | 2 |
| Relationship Recall | 0 | 2 |
| Location Context Recall | 0 | 3 |
| **Aggregation Readiness Score** | **5** | **15** |

### Frame: customer_int_ts15.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 4 | 4 |
| Activities Recall | 2 | 2 |
| Relationship Recall | 0 | 3 |
| Location Context Recall | 0 | 4 |
| **Aggregation Readiness Score** | **6** | **20** |

### Frame: customer_int_ts35.jpg
| Metric | Prompt V1 | Prompt V2 |
|---|---|---|
| Object Recall | 4 | 4 |
| Activities Recall | 2 | 2 |
| Relationship Recall | 0 | 3 |
| Location Context Recall | 0 | 4 |
| **Aggregation Readiness Score** | **6** | **20** |

## Overall Metrics
- **Total Frames Evaluated:** 9
- **Avg Aggregation Readiness Score (V1):** 2.66
- **Avg Aggregation Readiness Score (V2):** 8.00
- **Total Hallucination Incidents (V1):** 1
- **Total Hallucination Incidents (V2):** 0

## Conclusion
**Does Prompt V2 provide materially better metadata for Event Aggregation than Prompt V1?**
Yes, Prompt V2 provides materially better metadata for Event Aggregation. The introduction of specific relationship mappings (`subject_id`, `target_id`, `relation`) and spatial location context (`object_id`, `location`) significantly increased the Aggregation Readiness Score (from 2.66 to 8.00) without incurring higher hallucination penalties. This structured relational data enables deterministic downstream algorithms to build behavioral graphs rather than relying purely on text descriptions.
