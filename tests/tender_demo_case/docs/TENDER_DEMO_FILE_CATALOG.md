# Tender Demo File Catalog

## Source files

| File | Purpose |
| --- | --- |
| `run_tender_demo_pipeline.py` | Standard isolated tender-demo orchestrator. |
| `run_tender_demo_fast_parallel_pipeline.py` | Fast parallel Top-K orchestrator with runtime metrics and optional fallback rerun logic. |
| `step_00_runtime_metrics.py` | Helpers for building Step 20 timing payloads. |
| `step_09_final_summary.py` | Older summary builder for the standard full-VLM path. |
| `step_10_yolo_detection.py` | YOLO detection pass over sampled or motion-candidate frames. |
| `step_11_yolo_object_scoring.py` | Aggregates detections, usefulness scoring, and annotated frames. |
| `step_11b_object_motion_state.py` | Estimates motion state from adjacent YOLO detections. |
| `step_12_fused_clip_evidence.py` | Fuses older VLM, motion, YOLO, and timeline evidence. |
| `step_13_rank_candidate_clips.py` | Ranks candidate clips using motion, object, and motion-state features. |
| `step_14_select_topk_clips.py` | Applies Top-K and guardrail selection logic. |
| `step_15_create_topk_vlm_inputs.py` | Creates Top-K temporal strips for Qwen. |
| `step_16_run_topk_qwen.py` | Runs Qwen on Top-K inputs with compact-schema parsing and fallback repair. |
| `step_16b_incident_recheck.py` | Optional incident-focused second-pass reasoning. |
| `step_17_topk_final_summary.py` | Main final summary for UI and report consumption. |
| `step_18_export_event_clips.py` | Exports event clips and assembles compiled review video with AVI fallback. |
| `step_19_create_demo_report.py` | Generates the local HTML report. |
| `tender_demo_ui.py` | Streamlit interface for running and reviewing the isolated pipeline. |
| `tender_demo_vlm_adapter.py` | Isolated Hugging Face Qwen VLM adapter. |

## Working folders

| Folder | Purpose |
| --- | --- |
| `debug_runs/` | One subfolder per pipeline execution. This is the main run artifact store. |
| `ui_uploads/` | Videos uploaded through Streamlit. Each upload gets its own timestamped folder. |
| `video_imports/` | Manually curated local videos shown in the UI import selector. |
| `tmp_step16b_robbery_smoke/` | Temporary isolated smoke-test artifacts for Step 16B work. Not a core pipeline run folder. |
| `__pycache__/` | Python bytecode cache. |

## Typical fast run contents

Observed example: `debug_runs/localcam2_20260702_183043`

| Artifact | Meaning |
| --- | --- |
| `01_video_info.json` | Source video metadata. |
| `02_sampled_frames/` | JPEGs sampled from the source video. |
| `02_sampled_frames.json` | Sample manifest with timestamps and frame paths. |
| `03_motion_scores.json` | Motion scores per sampled frame. |
| `04_motion_candidates.json` | Frames passing the motion threshold. |
| `05_candidate_clips.json` | Raw grouped clip windows. |
| `06_expanded_clips.json` | Context-expanded clip windows. |
| `10_yolo_detections.json` | Raw YOLO detections per frame. |
| `11_yolo_annotated_frames/` | YOLO-labeled JPEGs for inspection. |
| `11_yolo_object_scores.json` | Scored YOLO evidence. |
| `11_yolo_usefulness_report.json` | Summary statistics about YOLO usefulness. |
| `11b_object_motion_states.json` | Moving or stationary object summaries by clip. |
| `11b_object_motion_state_report.json` | Step 11B compact report. |
| `13_ranked_clips.json` | Ranked candidate clips with component reasons. |
| `13_ranked_clips_report.json` | Summary of ranking results. |
| `14_selected_top_clips.json` | Selected clip records. |
| `14_selected_top_clips_report.json` | Selection metadata such as requested Top-K and effective Top-K. |
| `15_topk_vlm_inputs/` | Top-K strip images passed to Qwen. |
| `15_topk_vlm_inputs.json` | Strip manifest and source-clip metadata. |
| `16_topk_vlm_outputs.json` | Raw Qwen output, parse result, fallback flags, and parsed JSON. |
| `16b_incident_recheck_outputs.json` | Optional incident-focused second-pass outputs. |
| `16b_incident_recheck_report.json` | Optional Step 16B summary counts. |
| `17_topk_final_summary.json` | Main UI and report summary payload. |
| `17_topk_final_summary.md` | Human-readable markdown summary. |
| `18_exported_clips/` | Individual exported clip videos and compiled review video artifacts. |
| `18_exported_clips.json` | Export manifest for individual clip files. |
| `18_compiled_review_video.json` | Compiled video manifest including MP4 and AVI fallback info. |
| `19_demo_report.html` | Local HTML report. |
| `20_runtime_metrics.json` | Fast pipeline timing manifest. |
| `20_search_index.json` | Flattened records used by the UI search tab. |

## Standard-only artifacts

These are part of the standard full-VLM path and are intentionally skipped by the fast optimized path:

| Artifact | Meaning |
| --- | --- |
| `07_vlm_inputs.json` | Full-set strip manifest for all expanded clips. |
| `08_vlm_outputs.json` | Qwen outputs for the full-set standard path. |
| `09_final_summary.json` | Older full timeline summary derived from Step 8 outputs. |
| `12_fused_clip_evidence.json` | Fusion output combining older VLM, motion, and YOLO evidence. |
| `12_fused_evidence_report.json` | Fusion summary report. |

## UI-owned folders

### `ui_uploads/`
Created by the Streamlit upload flow. Each upload is saved in a timestamped folder named like:

- `20260702_183041_localcam2/`
- `20260630_121627_ANPR2-D/`

These are input storage folders, not run output folders.

### `video_imports/`
Used by the UI import-folder selector. It is intended as a stable place to drop local videos that should appear in the UI without re-uploading them.

## Search index records

`20_search_index.json` contains one flattened record per final-summary clip. It merges:

- clip timing
- category
- summary fields
- incident recheck fields
- motion summary fields
- YOLO classes
- compiled video path
- raw Qwen parsed JSON
- raw text for keyword searching

The UI search tab reads this file instead of re-deriving searchable text on every page load.

## Compiled review video outputs

Step 18 can produce several related files inside `18_exported_clips/`:

| File | Meaning |
| --- | --- |
| `18_compiled_review_video.mp4` | Preferred direct MP4 export when FFmpeg H.264 succeeds. |
| `18_compiled_review_video_fallback.avi` | OpenCV MJPG fallback video. |
| `18_compiled_review_video_web.mp4` | Browser-friendly MP4 created from the AVI fallback when possible. |
| `_compiled_frames/` | Temporary image frames used during FFmpeg assembly. |

The manifest determines which file should actually be played.

## Notes on historical runs

Older `debug_runs/` folders reflect the code version active when they were generated. That means:

- older fast runs may not contain `11b_*` or `16b_*`
- older Step 20 payloads may not list `11B` or `16B`
- older Step 16 outputs may show parse failures that the current code is intended to reduce

When analyzing run folders, treat them as versioned artifacts rather than a perfect mirror of current source behavior.
