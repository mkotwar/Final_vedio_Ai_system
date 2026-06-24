# HF Stage Profile Summary

- Model: `qwen2.5vl:7b`
- Device: `cuda`
- One-time model load: `12.43s`
- Warm 1-frame runtime: `8.77s/frame`
- Warm 2-frame runtime: `7.85s/frame`
- Dominant steady-state stage: `generate`

## Answer

- This profile measures warm one-frame and two-frame HF runs separately from model load.
- Model load is a one-time cost of 12.43s and should not be confused with steady-state frame latency.
- Warm 1-frame average runtime is 8.77s/frame.
- Inside that 1-frame run: generate=7.27s, vision+tensor=0.07s, ocr=1.42s.
- The dominant steady-state stage is generate.
- So the ~5s/frame behavior is mainly model generation time, not OCR and not image preprocessing.
- Batching improves per-frame cost, which means there is some shared overhead amortization across frames.