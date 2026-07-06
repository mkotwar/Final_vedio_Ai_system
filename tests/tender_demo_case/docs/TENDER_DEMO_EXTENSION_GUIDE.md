# Tender Demo Extension Guide

## Goal
This guide explains where to extend the isolated tender-demo pipeline safely without touching production code.

## Safe extension boundary
Keep changes inside `tests/tender_demo_case/` and its docs or run artifacts only. The isolated demo already has clean extension seams around:

- new intermediate JSON files
- extra optional steps
- UI-only controls
- summary enrichment
- search-index enrichment

## Best extension points

### Add new evidence after YOLO
Good place: after Step 11 or Step 11B

Why:

- raw detections already exist
- frame timestamps are available
- clip-level ranking can consume additional evidence in Step 13

Examples:

- object persistence score
- crowd density estimate
- dwell-time estimate
- zone-entry heuristics

### Add new reasoning after Step 16
Good place: model it like Step 16B

Why:

- Top-K is already narrowed
- strip images are already created
- expensive second-pass reasoning stays isolated

Examples:

- collision recheck
- fall-risk recheck
- theft-specific evidence extraction
- contradiction check between initial Qwen output and YOLO/motion evidence

### Add new ranking features in Step 13
Good place when you want better candidate prioritization before Qwen.

Current Step 13 already consumes:

- motion score
- YOLO counts
- interaction context
- motion-state hints

Examples:

- temporal persistence score
- scene-specific boosts
- per-class urgency scoring
- anomaly heuristics

### Add new UI controls
Good place: `tender_demo_ui.py`

Current pattern:

- add preset value or custom control
- pass it through `build_pipeline_env`
- read it in the relevant step file
- surface it in Step 17 analysis settings if it affects interpretation

### Add new report sections
Good places:

- `step_17_topk_final_summary.py` for structured JSON
- `step_19_create_demo_report.py` for HTML
- `tender_demo_ui.py` for Streamlit display

## Recommended extension workflow

1. Add a new isolated step file or extend an existing isolated step.
2. Write a new JSON artifact instead of mutating unrelated files heavily.
3. Wire the new artifact into the fast pipeline first if it is performance-sensitive.
4. Update Step 17 so the result becomes visible in the summary layer.
5. Update Step 19 and the UI only after the JSON contract is stable.

## Current contracts worth preserving

### Step 14 contract
`14_selected_top_clips.json` is the selection boundary for Top-K Qwen work.

### Step 15 contract
`15_topk_vlm_inputs.json` and the strip folder are the media boundary for Qwen reasoning.

### Step 16 contract
`16_topk_vlm_outputs.json` should always preserve:

- raw VLM output
- parse status
- parsed JSON when available
- fallback indicator

### Step 17 contract
`17_topk_final_summary.json` is the main UI/report API. New downstream features should usually appear here.

### Step 18 contract
The compiled video manifest should remain the single source of truth for playable output selection.

## Adding tracking later
Tracking was explicitly deferred, but the clean insertion point is after Step 10 or Step 11.

Recommended future shape:

1. create a new isolated file such as `step_11c_tracking.py`
2. read `10_yolo_detections.json`
3. write a clip- or frame-level tracking artifact
4. let Step 13 or Step 17 consume that artifact optionally

This keeps tracking out of production code and avoids entangling it with the core detector implementation.

## Adding scene-specific behavior
For scene-specific tuning, prefer:

- Step 13 boosts for ranking
- Step 16 prompt rules
- Step 16B recheck variants
- Step 17 wording adjustments

Avoid hard-coding scene logic directly into early generic steps unless it changes true candidate recall.

## Backward compatibility guidance

- Older `debug_runs/` folders will not contain every new artifact.
- New readers should load optional files defensively.
- UI and report rendering should degrade gracefully when files are missing.

This is already the dominant pattern in:

- Step 17
- Step 19
- `tender_demo_ui.py`

## Common pitfalls

### Adding too much schema to Step 16
Large nested output schemas increase parse failure risk, especially under low token budgets.

### Putting expensive work before Top-K
If new logic is expensive and runs before Step 14, it can erase most of the fast pipeline advantage.

### Forgetting UI fallback paths
If a new media output has multiple possible file variants, write the preferred path into a manifest and make the UI read the manifest instead of hard-coding one filename.

### Breaking old runs
Do not assume every debug run has `11b_*`, `16b_*`, or newer manifest fields.

## Suggested future isolated steps

| Candidate step | Purpose |
| --- | --- |
| `11c_tracking.py` | Persistent object tracks and dwell behavior. |
| `13b_scene_priors.py` | Scene-aware ranking prior layer. |
| `16c_consistency_check.py` | Compare VLM claims against YOLO and motion evidence. |
| `17b_alert_packaging.py` | Create downstream-friendly alert payloads from summary data. |
| `18b_storyboard_export.py` | Generate static storyboard images for sharing without video playback. |

## Rule of thumb
If a new feature changes interpretation, add it around Steps 16 to 17. If it changes candidate recall, add it before Step 14. If it changes user experience only, keep it in the UI or report layers.
