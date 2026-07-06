# Adaptive Sampling Comparison

This folder contains an isolated audit and comparison testcase for checking whether important incident frames are reaching the tender-demo VLM input stage.

It does not modify:

- `app/`
- `tests/tender_demo_case/`
- `tests/manual_benchmark_case/`
- `tests/final_demo/`

It also does not call Qwen or rerun YOLO.

## What it compares

1. Existing tender-demo VLM input selection from an already completed tender-demo run
2. A standalone adaptive sampler prototype inspired by the production sampling stack

## Main outputs

For each comparison run, outputs are written under:

`tests/adaptive_sampling_comparison/debug_runs/<video_name>_<timestamp>/`

Key files:

- `production_adaptive_sampling_audit.md`
- `production_adaptive_sampling_audit.json`
- `adaptive_retained_frames/`
- `adaptive_retained_frames.json`
- `adaptive_sampling_report.json`
- `vlm_selection_comparison.json`
- `vlm_selection_comparison.md`
- `vlm_selection_comparison.csv`
- `comparison_contact_sheet.jpg`
- `comparison_summary.md`

The comparison now separates tender-demo strip coverage into:

- `current_panel_covered`: the target is close to the strip's CURRENT/focal panel
- `context_panel_covered`: the target is only close to PREVIOUS or NEXT
- `missing`: the target is not close to any panel

This matters because the Qwen prompt mainly analyzes the CURRENT panel and uses PREVIOUS/NEXT as context.

Tender coverage is computed globally across all Step 15 strips:

- nearest `CURRENT` panel across all strips
- nearest `PREVIOUS` panel across all strips
- nearest `NEXT` panel across all strips
- nearest `ANY` panel across all strips

That avoids a false `context_panel_covered` result when one strip has a close `NEXT` panel but another strip has the target as its `CURRENT` focal panel.

## Runner env vars

- `ADAPTIVE_COMPARE_VIDEO_PATH`
- `ADAPTIVE_COMPARE_TENDER_RUN_DIR`
- `ADAPTIVE_COMPARE_TARGET_TIMESTAMPS`
- `ADAPTIVE_COMPARE_TARGET_LABELS`
- `ADAPTIVE_COMPARE_BASE_INTERVAL_SECONDS`
- `ADAPTIVE_COMPARE_MOTION_THRESHOLD`
- `ADAPTIVE_COMPARE_HIST_THRESHOLD`
- `ADAPTIVE_COMPARE_SIMILARITY_THRESHOLD`
- `ADAPTIVE_COMPARE_MAX_FRAME_GAP_SECONDS`
- `ADAPTIVE_COMPARE_COVERAGE_THRESHOLD_SECONDS`

## Example

```powershell
$env:ADAPTIVE_COMPARE_VIDEO_PATH="C:\Users\Vinfocom\Downloads\robbery_5mins.mp4"
$env:ADAPTIVE_COMPARE_TENDER_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\tender_demo_case\debug_runs\robbery_5mins_20260703_131949"
$env:ADAPTIVE_COMPARE_TARGET_TIMESTAMPS="00:21,01:00,01:12,02:18"
$env:ADAPTIVE_COMPARE_TARGET_LABELS="normal_customer,weapon_stage,grabbing_stage,display_theft_stage"
$env:ADAPTIVE_COMPARE_COVERAGE_THRESHOLD_SECONDS="3.0"
.\.venv\Scripts\python.exe tests/adaptive_sampling_comparison/run_adaptive_sampling_comparison.py
```
