# Tender Demo UI Guide

## What the UI is
`tender_demo_ui.py` is a Streamlit application that:

- lets you choose an input video
- selects the standard or fast isolated pipeline
- applies analysis presets
- launches the pipeline as a subprocess
- streams user-friendly status updates
- loads the latest run artifacts from `debug_runs/`
- presents summary, events, search, evidence timeline, files, and compiled video playback

## Main UI areas

### Sidebar
The sidebar controls:

- pipeline engine
- input mode
- processing preset
- quick result mode
- custom analysis sensitivity controls
- compiled review video settings
- optional max video duration limit

### Tabs
Current tabs:

1. `Run Pipeline`
2. `Results Summary`
3. `Events`
4. `Search`
5. `Evidence Timeline`
6. `Files`

## Input modes

### Use existing local/server video path
The user pastes or selects a filesystem path.

### Upload video file
The UI writes the upload into `tests/tender_demo_case/ui_uploads/<timestamp>_<name>/`.

### Select from import folder
The UI lists videos from `tests/tender_demo_case/video_imports/`.

## Pipeline engine selector

### Fast parallel Top-K pipeline

- optimized for quick review
- skips standard Steps 7, 8, 9, and 12
- can include Step 11B and Step 16B
- writes `20_runtime_metrics.json`

### Standard demo pipeline

- preserves the older full-demo flow
- slower and more compatibility-oriented

## Processing presets

### Quick scan

- `sample_every_seconds = 4.0`
- `top_k = 3`
- `qwen_max_new_tokens = 192`
- no incident recheck

### Fast demo

- `sample_every_seconds = 3.0`
- `top_k = 5`
- `qwen_max_new_tokens = 256`

### Balanced

- `sample_every_seconds = 2.0`
- `top_k = 8`
- `qwen_max_new_tokens = 384`

### Sensitive Incident Review

- `sample_every_seconds = 1.0`
- `top_k = 20`
- `qwen_max_new_tokens = 512`
- incident recheck enabled
- incident fallback pass enabled

### High accuracy review

- `sample_every_seconds = 0.5`
- `top_k = 25`
- `qwen_max_new_tokens = 512`
- incident recheck enabled
- incident fallback pass enabled

### Custom
Exposes direct controls for:

- sample interval
- Top-K
- motion threshold
- Qwen token budget
- YOLO confidence
- YOLO image size
- incident recheck
- recheck-all-topk
- fallback pass

## Quick result mode
Quick result mode is a UI-level speed shortcut layered on top of preset selection. It forces a small, aggressive setting bundle oriented toward quick scans.

Current quick settings:

- sample interval: `4.0`
- Top-K: `3`
- Qwen tokens: `192`
- YOLO size: `416`
- YOLO confidence: `0.40`
- motion threshold: `0.25`

## How the UI launches a run
The UI builds an environment map and launches either:

- `tests/tender_demo_case/run_tender_demo_fast_parallel_pipeline.py`
- `tests/tender_demo_case/run_tender_demo_pipeline.py`

The most important environment variables passed by the UI are:

- `TENDER_DEMO_INPUT_VIDEO`
- `TENDER_DEMO_PIPELINE_ENGINE`
- `TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE`
- `TENDER_DEMO_SAMPLE_EVERY_SECONDS`
- `TENDER_DEMO_TOP_K_CLIPS`
- `TENDER_DEMO_TOP_K_MAX_CLIPS`
- `TENDER_DEMO_MOTION_THRESHOLD`
- `TENDER_DEMO_QWEN_*`
- `TENDER_DEMO_YOLO_*`
- `TENDER_DEMO_ENABLE_INCIDENT_RECHECK`
- `TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK`
- `TENDER_DEMO_INCIDENT_FALLBACK_PASS`
- compiled-video settings

## Progress reporting
The UI does not deeply inspect process internals. Instead, it translates known log lines into:

- stage labels
- progress percentages
- user-friendly status messages

It has separate progress maps for:

- standard pipeline
- fast pipeline

Both progress maps now include:

- Step `11B`
- Step `16B`

## Results Summary tab
This tab reads `17_topk_final_summary.json` and related manifests to show:

- final summary text
- counts by category
- scene overview
- analysis settings
- incident recheck summary when available
- runtime and performance metrics when available
- compiled review video playback

## Events tab
This tab renders one event card per clip with:

- clip timing
- category
- caption and description
- strip image
- YOLO annotated frame
- exported clip playback when present
- incident recheck evidence
- motion-state evidence
- optional raw Qwen output

## Search tab
This tab reads `20_search_index.json` and supports keyword-driven browsing over flattened evidence records.

Search includes:

- clip metadata
- summary text
- incident category and explanations
- motion summary
- YOLO classes
- ranking reasons
- raw parsed Qwen fields

## Evidence Timeline tab
This tab is oriented around time-ordered browsing of clip evidence rather than only grouped categories.

## Files tab
This tab exposes the generated files from the active run for inspection.

## Compiled review video playback
Current playback resolution order is intentionally resilient:

1. `browser_playable_video_path`
2. `playback_recommended_file`
3. `compiled_video_path`
4. `fallback_video_path`

This is important because some runs only produce:

- a direct MP4
- an AVI fallback
- or a browser-transcoded MP4 created from the AVI

If FFmpeg is not on `PATH`, the UI also checks the common WinGet links location for `ffmpeg.exe`.

## Active run selection
The UI can:

- use the latest debug run
- keep a run selected in session state
- load all result manifests from that run directory

## Common operator patterns

### Fast first pass

1. Select `Fast parallel Top-K pipeline`
2. Use `Fast demo` or `Balanced`
3. Run pipeline
4. Inspect `Results Summary`, `Events`, and `Search`

### Incident-sensitive pass

1. Select `Fast parallel Top-K pipeline`
2. Use `Sensitive Incident Review`
3. Confirm the higher latency tradeoff
4. Review Step 16B output in `Results Summary` and `Events`

### Compatibility pass

1. Select `Standard demo pipeline`
2. Run when older full-VLM path behavior is needed

## Current UI limitations

- Progress is log-driven, not a direct internal orchestration API.
- Older run folders may not contain newer artifacts such as `11b_*` or `16b_*`.
- Playback quality depends on whether FFmpeg is discoverable and whether browser-playable MP4 conversion succeeds.
- The UI is optimized for local artifact browsing, not multi-user shared state.
