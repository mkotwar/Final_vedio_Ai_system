# Tender Demo Test Cases

## Step 16 / Step 17 quality case

Use this testcase when you want to validate:

- Step 16 JSON parse quality
- prompt behavior on temporal strips
- Step 17 summary wording quality
- whether the final summary matches what the VLM actually described

Script:

`tests/tender_demo_case/run_step16_step17_quality_case.py`

### Run on a video

```powershell
& ".\video-search-engine\.venv\Scripts\python.exe" `
  ".\video-search-engine\tests\tender_demo_case\run_step16_step17_quality_case.py" `
  --video "C:\Users\Vinfocom\Downloads\localcam1.mp4" `
  --prompt-file ".\video-search-engine\tests\tender_demo_case\prompts\temporal_strip_observation_prompt.txt" `
  --sample-every-seconds 1.0 `
  --top-k-clips 6 `
  --motion-threshold 0.15 `
  --max-new-tokens 512
```

### Evaluate an existing run only

```powershell
& ".\video-search-engine\.venv\Scripts\python.exe" `
  ".\video-search-engine\tests\tender_demo_case\run_step16_step17_quality_case.py" `
  --run-dir ".\video-search-engine\tests\tender_demo_case\debug_runs\localcam1_20260704_192901"
```

### Output files

The testcase writes these into the chosen tender-demo run folder:

- `21_step16_step17_quality_case.json`
- `21_step16_step17_quality_case.md`

These summarize:

- Step 16 parse success / fallback counts
- Step 17 main summary
- scene overview
- clip-by-clip descriptions using Step 16 and Step 17 outputs

### Recommended use

Use this testcase when:

- a summary sounds hardcoded or wrong
- Step 16 parses but Step 17 wording still feels weak
- you want to try a new Step 16 prompt safely before changing broader demo behavior
