# Phase 1 Benchmark Workbook

## Purpose

This workbook is the operational companion to
[PHASE1_ACCEPTANCE_CHECKLIST.md](C:\Mukul K\vinfo1\video-search-engine\docs\PHASE1_ACCEPTANCE_CHECKLIST.md).

Use it to evaluate real videos against Phase 1 tender expectations and identify
the earliest pipeline stage where each failure begins.

This is not a narrative report. It is a working investigation and validation log.

---

## How To Use

For each benchmark video:

1. Watch the original video and write the ground truth incident list.
2. Review extracted frames and note whether each incident window is represented.
3. Review frame metadata and note missed or hallucinated entities/actions.
4. Review event catalog and note event merge/split/count issues.
5. Review summary/timeline and note whether the narrative matches the event evidence.
6. Mark the earliest failing stage:
   `ingestion`, `frame_coverage`, `metadata`, `relationships`, `event_aggregation`, `summary`, `search`

Always assign the failure to the earliest stage where the truth was lost.

Example:

* if the incident never appears in analyzed frames, the failure is `frame_coverage`
* if the incident appears in frames but the object is missing, the failure is `metadata`
* if metadata is correct but the event type is wrong, the failure is `event_aggregation`
* if events are correct but the summary lies, the failure is `summary`

---

## Severity Guide

Use one severity per issue:

* `critical`: breaks investigation usefulness or hides the incident
* `high`: major distortion of actors, objects, or event meaning
* `medium`: partial loss of detail or noticeable inconsistency
* `low`: cosmetic or minor wording issue

---

## Benchmark Record Template

Fill one record per tested video.

### Video Record

* `video_id`:
* `source_type`: CCTV / mobile / downloaded / analog export
* `format`: MP4 / AVI / MOV / MKV / MPEG4
* `duration`:
* `camera_count`: single camera
* `scene_type`: office / corridor / road / parking / mixed / other
* `reviewer`:
* `review_date`:

### Ground Truth

* `incident_1`:
* `incident_2`:
* `incident_3`:
* `expected_empty_windows`:
* `expected_key_actors`:
* `expected_key_objects`:

### Pipeline Findings

* `frame_coverage_status`:
* `metadata_status`:
* `relationship_status`:
* `event_aggregation_status`:
* `summary_status`:
* `search_status`:

### Earliest Failure

* `earliest_failure_stage`:
* `severity`:
* `root_issue`:
* `notes`:

---

## Current Seed Videos

These are recent videos already discussed during debugging and should be moved
through the benchmark process first.

### 1. `2cdfa63e-9545-4be3-95ce-e130db4f99f8`

Observed concerns:

* object drop/pick sequence not reliably represented
* actor duplication across frames
* empty scene vs subject-present confusion
* summary flattened object interactions into routine pedestrian activity

Suggested earliest-stage checks:

* `frame_coverage`
* `metadata`
* `event_aggregation`

### 2. `6f0feeea-8724-4c0f-9e9c-75c3e11cd0d2`

Observed concerns:

* long extracted-frame gaps between incident windows
* some incidents not present in analyzed frame set
* later summary could only reflect retained windows

Suggested earliest-stage checks:

* `frame_coverage`

### 3. `d4bdf88e-1798-476a-94f6-96a62bf7830f`

Observed concerns:

* adaptive sampling retained only a few frames in prior diagnostics
* likely useful as a sampling regression case

Suggested earliest-stage checks:

* `frame_coverage`

---

## Phase 1 Scoring Table

Use this table during review.

| Video ID | Expected Incident | Expected Time Range | Evidence In Frames | Evidence In Metadata | Event Correct | Summary Correct | Earliest Failure Stage | Severity | Checklist Ref | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2cdfa63e-9545-4be3-95ce-e130db4f99f8 | person drops object | fill | pending | pending | pending | pending | pending | pending | C4 / D / E | seed case |
| 2cdfa63e-9545-4be3-95ce-e130db4f99f8 | person picks object | fill | pending | pending | pending | pending | pending | pending | C4 / D / E | seed case |
| 6f0feeea-8724-4c0f-9e9c-75c3e11cd0d2 | missing incident window | fill | pending | pending | pending | pending | pending | pending | B2 | seed case |
| d4bdf88e-1798-476a-94f6-96a62bf7830f | sampling gap regression | fill | pending | pending | pending | pending | pending | pending | B2 | seed case |

---

## Exit Criteria For Benchmark Pass

Do not call Phase 1 stable until:

1. Benchmark videos cover all major tender behaviors in scope.
2. Each benchmark video has ground truth documented.
3. Each failed case is assigned to the earliest failing stage.
4. Re-tested fixes improve the benchmark row that motivated the fix.
5. No critical benchmark failure remains open for:
   * frame coverage
   * object detection
   * actor consistency
   * event aggregation

---

## Next Step After This Workbook

Populate the CSV tracker with real findings from the first three seed videos, then
produce a gap matrix:

* repeated failures by stage
* repeated failures by requirement section
* repeated failures by incident type

That matrix becomes the engineering roadmap.
