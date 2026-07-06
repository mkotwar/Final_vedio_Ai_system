# VLM Model Comparison Case

Isolated testcase to compare image-prompt outputs and speed between:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

This testcase does not modify production code or tender-demo code.

## What it does

- accepts one or more image paths
- accepts a prompt string or prompt file
- runs both VLMs on the same images
- records per-image raw outputs
- records per-image timing and output length
- writes a side-by-side JSON and Markdown report

## Run

From `C:\Mukul K\vinfo1\video-search-engine`:

```powershell
& ".\.venv\Scripts\python.exe" ".\tests\vlm_model_comparison_case\run_vlm_model_comparison.py" `
  --image "C:\path\to\image1.jpg" `
  --image "C:\path\to\image2.jpg" `
  --prompt "Describe exactly what is visible in this image."
```

Or use a prompt file:

```powershell
& ".\.venv\Scripts\python.exe" ".\tests\vlm_model_comparison_case\run_vlm_model_comparison.py" `
  --image "C:\path\to\image1.jpg" `
  --prompt-file ".\tests\vlm_model_comparison_case\sample_prompt.txt"
```

## Outputs

A new run folder is created under:

- `tests/vlm_model_comparison_case/debug_runs/`

Files include:

- `01_inputs.json`
- `02_comparison_results.json`
- `03_comparison_report.md`

