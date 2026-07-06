# VLM Summary Case

Isolated testcase to generate a plain human-readable summary from tender-demo Step 16 VLM outputs only.

This testcase is meant to answer:

- what does the VLM itself say is visible?
- what scene/activity/object summary can we build without Step 16B incident hardcoding?

It does not use:

- Step 16B incident recheck output
- Step 17 hardcoded incident category logic
- production code

## Inputs

You can pass either:

- a tender-demo `run_dir`
- or a direct `16_topk_vlm_outputs.json` path

Optional:

- `15_topk_vlm_inputs.json`
- `01_video_info.json`

## Run

From `C:\Mukul K\vinfo1\video-search-engine`:

```powershell
& ".\.venv\Scripts\python.exe" ".\tests\vlm_summary_case\run_vlm_summary_case.py" `
  --run-dir ".\tests\tender_demo_case\debug_runs\localcam2_20260704_181112"
```

Or:

```powershell
& ".\.venv\Scripts\python.exe" ".\tests\vlm_summary_case\run_vlm_summary_case.py" `
  --vlm-outputs ".\tests\tender_demo_case\debug_runs\localcam2_20260704_181112\16_topk_vlm_outputs.json"
```

## Outputs

A new folder is created under:

- `tests/vlm_summary_case/debug_runs/`

Files:

- `01_input_snapshot.json`
- `02_vlm_first_summary.json`
- `03_vlm_first_summary.md`
