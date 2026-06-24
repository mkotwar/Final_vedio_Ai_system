# Phase 1 Acceptance Checklist

## Purpose

This document converts the Phase 1 tender context into a practical acceptance checklist for the
current single-camera investigation system.

Use this checklist as the source of truth when evaluating:

* frame extraction
* metadata quality
* relationship detection
* event aggregation
* summary quality
* search readiness

The goal is to judge the system against explicit requirements rather than subjective impressions.

---

## Scope

This checklist applies only to:

* single-camera investigation
* uploaded video analysis
* event-centric investigation output

This checklist does not yet require:

* multi-camera tracking
* person ReID across cameras
* vehicle ReID across cameras
* operational analytics beyond single-video investigation output

---

## Acceptance Scale

Use one of these statuses for each item:

* `PASS`: requirement is implemented and validated on benchmark videos
* `PARTIAL`: requirement is implemented but unreliable, inconsistent, or weak on edge cases
* `FAIL`: requirement is missing or regularly produces unusable output
* `NOT TESTED`: requirement exists in scope but has not yet been benchmarked

---

## Phase 1 Outcome

At minimum, the system must be able to:

1. Accept a supported video file.
2. Extract investigation-relevant frames with reliable timestamp coverage.
3. Detect people, vehicles, animals, and objects present in those frames.
4. Produce structured frame metadata.
5. Detect searchable relationships between entities.
6. Convert frame observations into semantic events.
7. Generate an event timeline and an executive summary from events.
8. Support natural-language investigation queries.
9. Return timestamps and evidence frames for detected events.

If any of these fail consistently, the system is not yet Phase 1 compliant.

---

## A. Input And Ingestion

### A1. Supported video sources

Requirement:

* CCTV footage
* mobile phone recordings
* downloaded video files
* analog camera exports

Acceptance:

* system accepts these categories without manual code changes
* no source category should require a custom ingestion workflow

Status: `NOT TESTED`

### A2. Supported file formats

Requirement:

* MP4
* AVI
* MOV
* MKV
* MPEG4

Acceptance:

* upload succeeds
* file is stored
* frame extraction begins
* timestamps remain stable after ingestion

Status: `NOT TESTED`

---

## B. Frame Coverage

### B1. Timestamp fidelity

Requirement:

* every analyzed frame must retain correct video-relative timestamp

Acceptance:

* extracted frame timestamps match true playback position
* event start/end times correspond to visible evidence
* OCR-based wall-clock timestamps must never override playback timestamps incorrectly

Status: `PARTIAL`

Known issues:

* OCR wall-clock extraction has been brittle in some runs

### B2. Incident coverage

Requirement:

* relevant scene changes and incident windows must be captured in the analyzed frame set

Acceptance:

* incident windows do not disappear due to sampling gaps
* long stretches outside motion windows still retain sparse baseline evidence
* event-critical actions are visible in extracted frames, not only before or after them

Status: `PARTIAL`

Known issues:

* some recent videos still showed missing incident coverage due to extraction gaps

---

## C. Structured Frame Metadata

### C1. Person detection

Requirement:

Store:

* person type: man, woman, child, unknown

Attributes:

* upper wear color
* lower wear color
* upper wear type
* lower wear type
* hat
* backpack
* carried object

Actions:

* walking
* standing
* running
* sitting
* entering
* exiting
* talking
* holding object

Acceptance:

* visible people are not routinely missed
* clothing and carried-object attributes are grounded in the frame
* action labels match visible behavior
* empty scenes do not invent subjects

Status: `PARTIAL`

Known issues:

* hallucinated subjects in empty scenes were observed
* actor identity varies across frames
* object-related person actions are still inconsistent

### C2. Vehicle detection

Requirement:

Store:

* car
* motorcycle
* bicycle
* truck
* van
* bus

Attributes:

* color
* license plate (OCR)
* movement state

Actions:

* parked
* moving
* entering scene
* exiting scene

Acceptance:

* visible vehicles are typed and colored correctly
* OCR plate extraction is attached when readable
* movement state matches frame evidence

Status: `NOT TESTED`

### C3. Animal detection

Requirement:

* detect animal type and activity

Acceptance:

* visible animals are preserved in frame metadata and downstream events

Status: `NOT TESTED`

### C4. Object detection

Requirement:

Store:

* object type
* color
* activity

Examples:

* bag
* suitcase
* box
* phone
* bicycle

Acceptance:

* small but investigation-relevant objects are detected when visible
* carried, placed, dropped, or removed objects are not routinely missed
* unattended objects can be represented in metadata

Status: `FAIL`

Known issues:

* dropped and picked objects are frequently not detected even when incident frames are available

### C5. Activity labeling integrity

Requirement:

* actions must reflect the visible scene

Acceptance:

* office scenes must not be labeled as `crossing road`
* static scenes must not invent action
* ambiguous motion must not escalate into incorrect event labels

Status: `PARTIAL`

Known issues:

* false `crossing road` labels have appeared in office scenes
* generic walking/standing often replaces more specific object interaction

---

## D. Relationship Detection

Requirement:

Relationships must be identified and searchable.

Examples:

* person carrying bag
* person holding object
* person riding motorcycle
* person entering vehicle
* person exiting vehicle
* person talking to another person
* vehicle parked near person
* animal near person

Acceptance:

* relationships are stored in structured metadata
* relationships survive into event interpretation where relevant
* relationship entities refer to real detected actors and objects

Status: `PARTIAL`

Known issues:

* object-related relationships are weak because object detection itself is weak

---

## E. Event Aggregation

### E1. Event construction

Requirement:

* low-level frame observations must be converted into high-level events

Example target:

* `person_enters_vehicle`

Acceptance:

* events are not just renamed frame captions
* event boundaries match true start/end behavior
* event type reflects multi-frame behavior, not a single snapshot

Status: `PARTIAL`

Known issues:

* upstream metadata gaps cause flattened event meaning
* aggregation can still under-express high-value object interactions

### E2. Event structure

Requirement:

Every event should contain:

* `event_id`
* `video_id`
* `start_timestamp`
* `end_timestamp`
* `event_type`
* `event_summary`
* `actors`
* `vehicles`
* `objects`
* `animals`
* `confidence_score`

Acceptance:

* event catalog stores the required fields directly or via equivalent schema mapping
* event payload is sufficient for search, summary, and evidence review

Status: `PARTIAL`

Known issues:

* actor/object/animal separation is not yet consistently represented in the current event contract

### E3. Empty-scene behavior

Requirement:

* empty scenes should remain explicitly empty
* empty background windows should not become timeline events

Acceptance:

* no subject/person should be invented in an empty office
* no event should be emitted for an empty office window
* summaries and timelines should stay silent unless an actual action or incident is present

Status: `PARTIAL`

Known issues:

* empty windows were previously emitted as `empty_scene` events and still need benchmark confirmation after suppression

---

## G. Performance

### G1. Processing speed

Requirement:

* total processing time should remain below source video length for Phase 1 target runs

Acceptance:

* `total_ingestion_seconds < video_duration_seconds`
* performance report must expose video duration, ingestion time, and realtime ratio
* benchmark runs should clearly show whether the pipeline is faster than real time

Status: `NOT TESTED`

Known issues:

* the pipeline now reports realtime ratio, but benchmark validation against this target is still pending

### E4. Actor tracking and participant deduplication

Requirement:

* the same actor should not become multiple participants across adjacent frames

Acceptance:

* participant counts remain stable across appearance wording changes
* one tracked person should not produce false `multi_person` events
* event narratives should not list the same actor multiple times

Status: `FAIL`

Known issues:

* same person is often duplicated due to appearance variation and weak actor continuity

---

## F. Summary And Timeline

### F1. Executive summary

Requirement:

* summary must be investigation-grade and built from events

Acceptance:

* summary must not hallucinate incidents or actors
* summary must not omit major detected events
* summary language must reflect event evidence, not frame caption noise

Status: `PARTIAL`

Known issues:

* summaries inherit upstream misses and can still flatten or distort event meaning

### F2. Event timeline

Requirement:

* event timeline must reflect actual event catalog content

Acceptance:

* timeline order is correct
* timestamps match event windows
* evidence frame retrieval is possible per event

Status: `PARTIAL`

---

## G. Search Readiness

### G1. Natural-language search

Requirement:

* user must be able to search videos using natural language

Acceptance:

* search retrieves relevant events rather than raw disconnected frames
* timestamps and evidence frames are returned with results

Status: `NOT TESTED`

### G2. Relationship searchability

Requirement:

* relationships must be searchable

Acceptance:

* a query like `person holding bag` or `person near vehicle` can retrieve relevant events

Status: `NOT TESTED`

---

## H. Current Gap Summary

Based on observed testing to date, the largest compliance risks are:

1. incident coverage gaps in extracted/analyzed frames
2. weak small-object detection
3. missed object state changes like drop/pick/remove
4. unstable actor continuity across frames
5. event aggregation that still depends too heavily on noisy frame semantics
6. summary confidence that can outrun evidence quality

---

## I. How To Use This Checklist

For each benchmark video, fill a row like this:

| Video ID | Requirement | Expected | Actual | Status | Notes |
|---|---|---|---|---|---|
| sample_01 | C4 Object detection | bag should be detected | bag missed | FAIL | frame exists but metadata missed |

Recommended evaluation order:

1. B. Frame Coverage
2. C. Structured Frame Metadata
3. D. Relationship Detection
4. E. Event Aggregation
5. F. Summary And Timeline
6. G. Search Readiness

Do not evaluate later stages before earlier-stage evidence is confirmed.

---

## J. Immediate Next Deliverable

The next engineering step after this checklist is:

* create a benchmark validation sheet from real failing videos
* map each failing video to the checklist sections above
* mark every failure at the earliest stage where it first appears

That turns debugging into structured compliance work.
