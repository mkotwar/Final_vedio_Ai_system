# Phase 1 Gap Matrix

## Purpose

This matrix turns benchmark findings into an engineering priority list.

Use it after every benchmark update to answer three questions:

1. Where is the truth first being lost?
2. Which tender requirement is being broken most often?
3. What should we fix next to improve the highest number of failing videos?

---

## Current Seed Cases

| Video ID | Ground Truth Concern | Earliest Failure Stage | Severity | Tender Areas |
|---|---|---|---|---|
| `2cdfa63e-9545-4be3-95ce-e130db4f99f8` | pick/drop sequence visible in frames but collapsed into generic pedestrian activity | `event_aggregation` | `critical` | `C1`, `C4`, `D`, `E` |
| `2cdfa63e-9545-4be3-95ce-e130db4f99f8` | same actor represented as multiple participants across frames | `event_aggregation` | `high` | `C1`, `D`, `E` |
| `6f0feeea-8724-4c0f-9e9c-75c3e11cd0d2` | long blind windows in analyzed frame set | `frame_coverage` | `critical` | `B2` |
| `d4bdf88e-1798-476a-94f6-96a62bf7830f` | severe adaptive-sampling under-retention | `frame_coverage` | `critical` | `B2` |

---

## Repeated Failures By Stage

| Stage | Repeated Failure Pattern | Seed Evidence | Why It Matters |
|---|---|---|---|
| `frame_coverage` | long timestamp gaps remove incident windows before metadata is generated | `6f0feeea`, `d4bdf88e` | downstream fixes cannot recover incidents that were never analyzed |
| `metadata` | object interaction cues are weak or inconsistent even when a person action is visible | `2cdfa63e` | event builder cannot form reliable pick/drop events without grounded object cues |
| `event_aggregation` | single actors become multiple participants; object interactions are flattened into routine movement | `2cdfa63e`, `6f0feeea` | timeline becomes misleading even when frames are present |
| `summary` | narrative inherits wrong event types and participant counts | downstream symptom of above | summaries look polished while hiding core investigation failures |

---

## Repeated Failures By Requirement

| Checklist Ref | Requirement | Current Signal |
|---|---|---|
| `B2` | incident windows must remain visible in analyzed frames | repeated `critical` failures |
| `C1` | people and their actions must be grounded in the frame | repeated `partial` failures |
| `C4` | object handling must be captured when visible | active failure in seed office case |
| `D` | relationships and actor continuity must remain searchable and coherent | repeated `high` to `critical` failures |
| `E` | event aggregation must preserve incident meaning | repeated `critical` failures |

---

## What This Means

The current benchmarks point to two dominant failure clusters:

1. `frame_coverage`
   Incidents disappear before metadata exists. This is the first blocker because
   no later stage can repair missing visual evidence.

2. `event_aggregation`
   When frames do exist, the system still loses truth by turning object
   interactions into generic movement and by counting the same actor multiple
   times.

Metadata quality still matters, but the current seed cases show that we should
not treat every bad summary as a VLM problem. In two of the strongest examples,
the truth is lost either before metadata exists or after metadata already showed
useful cues.

---

## Fix Order

1. Stabilize `B2` frame coverage on benchmark videos.
2. Strengthen object-interaction extraction for visible pick/drop actions.
3. Tighten actor continuity and participant deduplication inside event aggregation.
4. Re-run benchmark rows and only then adjust summary wording.

---

## Exit Rule For This Matrix

Do not move a problem out of the active roadmap until its benchmark row changes
from `critical` or `high` to at least `medium`, with evidence from a fresh
re-test.
