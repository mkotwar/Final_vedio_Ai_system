# Tender Demo Full Architecture

## Scope
This document covers the isolated tender-demo implementation under `tests/tender_demo_case/` only. It describes the current code path, current UI behavior, generated artifacts, and the differences between the standard and fast pipelines.

## High-level purpose
The tender-demo pipeline is a local, test-isolated video triage flow for:

- sampling a source video
- finding motion-heavy candidate moments
- attaching YOLO object evidence
- selecting a smaller Top-K review set
- sending only those clips to Qwen VLM
- producing a human-readable summary
- exporting review clips and a compiled review video
- rendering a local HTML report and a Streamlit UI

It is intentionally separate from production services. The UI and pipeline operate from local files and environment variables.

## Main entrypoints

| File | Role |
| --- | --- |
| `run_tender_demo_pipeline.py` | Standard pipeline entrypoint. Full older demo flow. |
| `run_tender_demo_fast_parallel_pipeline.py` | Fast pipeline entrypoint. Current optimized Top-K path. |
| `tender_demo_ui.py` | Streamlit UI for launching runs and browsing results. |
| `tender_demo_vlm_adapter.py` | Isolated Qwen VLM adapter used by Step 8 and Step 16. |

## Pipeline families

### Standard demo pipeline
Entrypoint: `run_tender_demo_pipeline.py`

Current step order:

1. Step 1: read video info
2. Step 2: sample frames
3. Step 3: score motion
4. Step 4: select motion candidates
5. Step 5: group candidates into clips
6. Step 6: expand clips with context
7. Step 7: create temporal strip images for all expanded clips
8. Step 8: run Qwen on all temporal strips
9. Step 9: build older full-timeline summary
10. Step 10: run YOLO on selected frames
11. Step 11: score YOLO object evidence
12. Step 13: rank candidate clips
13. Step 14: select Top-K clips
14. Step 15: create Top-K VLM inputs
15. Step 16: run Qwen on Top-K only
16. Step 17: build Top-K final summary
17. Step 18: export event clips and compiled review video
18. Step 19: create HTML report
19. Step 12: fused evidence runs last

Important current behavior:

- Step 11B exists in the repo but is not currently wired into the standard pipeline entrypoint.
- Step 16B exists in the repo but is not currently wired into the standard pipeline entrypoint.
- Step 12 runs after Step 19 in the current standard orchestration, so the standard entrypoint order is not strictly numeric.

### Fast parallel Top-K pipeline
Entrypoint: `run_tender_demo_fast_parallel_pipeline.py`

Current fast flow:

1. Step 1: read video info
2. Step 2: sample frames
3. Step 3: score motion
4. Step 4: select motion candidates
5. Parallel section:
   - clip branch: Step 5 and Step 6
   - YOLO branch: Step 10, Step 11, Step 11B
6. Step 13: rank candidate clips
7. Step 14: select Top-K clips
8. Step 15: create Top-K VLM inputs
9. Step 16: run Qwen only on Top-K clips
10. Step 16B: optional incident recheck reasoning
11. Step 17: build final summary
12. Step 18: export clips and compiled review video
13. Step 19: create HTML report
14. Step 20: write runtime metrics

Fast pipeline intentionally skips the older full-VLM path:

- Step 7 skipped
- Step 8 skipped
- Step 9 skipped
- Step 12 skipped

## Why the fast pipeline is faster

- It samples frames at a configurable sparse interval.
- It ranks candidate moments before calling Qwen.
- It only sends Top-K selected clips to Qwen instead of every expanded clip.
- It runs clip-building and YOLO evidence generation in parallel.
- It writes explicit runtime metrics so bottlenecks are visible.

In observed runs, Qwen remains the dominant cost center. Example fast runs in `debug_runs/` show Step 16 taking far longer than frame sampling, motion scoring, or Step 13 through Step 15.

## Stage-by-stage architecture

### Step 1: video info
Reads the input video and records:

- path
- filename
- FPS
- frame count
- duration
- resolution
- file size

Output: `01_video_info.json`

### Step 2: sampled frames
Samples base frames at `TENDER_DEMO_SAMPLE_EVERY_SECONDS`.

Outputs:

- `02_sampled_frames/`
- `02_sampled_frames.json`

### Step 3: motion scoring
Scores motion between sampled frames using OpenCV frame differencing.

Output: `03_motion_scores.json`

### Step 4: motion candidate selection
Filters sampled frames using the normalized motion threshold.

Output: `04_motion_candidates.json`

### Step 5: candidate clips
Groups motion-heavy frames into raw clip windows.

Output: `05_candidate_clips.json`

### Step 6: expanded clips
Adds temporal context before and after each clip.

Output: `06_expanded_clips.json`

### Step 7: full-strip generation for standard pipeline
Creates three-panel temporal strips for all expanded clips.

Standard only.

Outputs:

- `07_vlm_inputs/` or equivalent strip folder created by the standard script
- `07_vlm_inputs.json`

### Step 8: full Qwen pass for standard pipeline
Runs Qwen on every Step 7 strip.

Standard only.

Output: `08_vlm_outputs.json`

### Step 9: older final summary
Builds the older clip timeline and summary from Step 8 outputs.

Standard only.

Output: `09_final_summary.json`

### Step 10: YOLO detection
Runs Ultralytics YOLO on either motion-candidate frames or sampled frames, depending on the configured input scope.

Output: `10_yolo_detections.json`

### Step 11: YOLO object scoring
Aggregates detections into richer frame-level evidence and annotated frames.

Outputs:

- `11_yolo_object_scores.json`
- `11_yolo_usefulness_report.json`
- `11_yolo_annotated_frames/`

### Step 11B: object motion state estimation
Matches YOLO detections across neighboring frames and estimates whether people, vehicles, and bag-like objects are moving or stationary.

Outputs:

- `11b_object_motion_states.json`
- `11b_object_motion_state_report.json`

Current usage:

- wired into the fast pipeline
- consumed by Step 13, Step 15, Step 16, Step 17, and search results

### Step 12: fused clip evidence
Combines motion, older VLM outputs, YOLO evidence, and older timeline data into a fusion report.

Outputs:

- `12_fused_clip_evidence.json`
- `12_fused_evidence_report.json`

Current usage:

- present for the standard family
- not part of the fast optimized path
- still read opportunistically by Step 14 if available

### Step 13: rank candidate clips
Ranks clips using:

- motion score
- person presence
- multi-person presence
- important object presence
- object density
- detection presence
- person-object interaction context
- object motion state hints from Step 11B

Outputs:

- `13_ranked_clips.json`
- `13_ranked_clips_report.json`

### Step 14: select Top-K clips
Chooses review clips with guardrails and a hard cap:

- requested Top-K
- maximum Top-K cap
- optional high-motion guardrail
- minimum selection fill

Outputs:

- `14_selected_top_clips.json`
- `14_selected_top_clips_report.json`

### Step 15: create Top-K VLM inputs
Creates three-panel strips only for selected Top-K clips and attaches motion-state hints.

Outputs:

- `15_topk_vlm_inputs/`
- `15_topk_vlm_inputs.json`

### Step 16: Qwen on Top-K only
Runs Qwen on selected strips only.

Current key behavior:

- uses a compact flat schema by default in fast mode
- strips markdown fences
- extracts the JSON region between the first `{` and last `}`
- attempts lightweight JSON repair
- falls back to a small best-effort parsed payload if strict parsing fails

Output: `16_topk_vlm_outputs.json`

### Step 16B: incident recheck reasoning
Optional second-pass reasoning focused on suspicious incident interpretation.

Current key behavior:

- can recheck all Top-K clips or a filtered subset
- uses an incident-specific prompt
- can still operate in a heuristic fallback mode if the VLM adapter is unavailable
- can override Step 17 categorization toward review-oriented outcomes

Outputs:

- `16b_incident_recheck_outputs.json`
- `16b_incident_recheck_report.json`

### Step 17: final summary
Builds the UI-facing and report-facing summary object.

Current summary includes:

- processing counts
- scene overview
- descriptive summary
- grouped event lists
- per-clip display records
- analysis settings snapshot
- incident fallback flags
- optional incident recheck summary

Outputs:

- `17_topk_final_summary.json`
- `17_topk_final_summary.md`

### Step 18: exported clips and compiled review video
Exports clip videos and assembles a compiled review video from temporal strips.

Current behavior:

- exports category-specific clips
- can fall back to normal clips if no suspicious clips exist
- tries FFmpeg H.264 MP4 first
- falls back to OpenCV MJPG AVI when direct MP4 export is unavailable
- may optionally create a browser-playable MP4 from the AVI
- writes a manifest that records recommended playback path

Outputs:

- `18_exported_clips/`
- `18_exported_clips.json`
- `18_compiled_review_video.json`

### Step 19: HTML demo report
Creates a local HTML report that reads the final summary and media outputs.

Output: `19_demo_report.html`

### Step 20: runtime metrics
Fast pipeline only.

Writes timings, parallel branch timings, settings snapshot, and skipped steps.

Output: `20_runtime_metrics.json`

### Search index
The UI also builds a search-friendly flattened index.

Output: `20_search_index.json`

## Current media path strategy

### Clip and image assets
Most manifests store repo-relative paths rooted under the run folder. The UI and HTML report resolve both:

- absolute paths
- repo-relative paths
- run-relative paths

### Compiled review video fallback chain
Current Step 18 and UI behavior prefer:

1. browser-playable MP4
2. playback-recommended file from the manifest
3. compiled MP4 path
4. fallback AVI path

This matters on Windows systems where FFmpeg may be installed but not on `PATH`, or where direct MP4 export fails and AVI remains the only playable artifact.

## Data flow summary

### Core fast-path lineage
`01 -> 02 -> 03 -> 04 -> (05,06 || 10,11,11B) -> 13 -> 14 -> 15 -> 16 -> 16B? -> 17 -> 18 -> 19 -> 20`

### Standard lineage
`01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 12`

## Key current limitations

- Standard and fast pipeline step numbering are not fully aligned in execution order.
- Standard pipeline still contains the older full-VLM path and does not yet orchestrate 11B or 16B.
- Step 12 is effectively legacy for the fast path.
- Qwen remains the major runtime bottleneck.
- Search, summary, and UI quality depend heavily on Step 16 parse quality.
- Incident-sensitive presets can materially increase runtime because 16B may exceed Step 16 runtime on long videos.

## Extension points

- wire 11B and 16B into the standard pipeline if parity is desired
- reduce Step 16 and 16B runtime through batching or model selection
- tighten ranking features in Step 13
- enrich Step 17 summary language with stronger scene-specific phrasing
- expand search index schema
- add tracking in a future isolated step without touching production code
