# Parallel HF Scaling Summary

- Model: `qwen2.5vl:7b`
- Sampled frames: `12`
- One-time warm load: `11.90s`

## Results

- `warm_batch4_uncached`: 2.24s/frame (5/12 succeeded)
- `warm_batch4_ocr_cached`: 2.06s/frame (5/12 succeeded)
- `warm_batch8_ocr_cached`: 2.05s/frame (5/12 succeeded)
- `warm_batch12_ocr_cached`: 2.02s/frame (5/12 succeeded)

## Answer

- Baseline warm batch-4 uncached cost is 2.24s/frame.
- Best observed configuration is warm_batch12_ocr_cached at 2.02s/frame.
- That is a 9.6% per-frame improvement versus the warm uncached batch-4 baseline.
- On this benchmark slice, keeping the model warm and moving OCR off the critical path helped, and batch scaling changed effective per-frame cost.