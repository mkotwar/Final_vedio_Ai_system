# Event Candidate Benchmark Summary

- Input video: `C:\Users\Vinfocom\Downloads\2026-06-01_18-33-54.mp4`
- Video duration: `121.55s`
- Total frames extracted: `122`
- Candidate events: `8`

## Modes

### candidate_only
- Variant `strip_tokens150_batch4`
  - Selected keyframes: `27`
  - Frames sent to Qwen: `8`
  - Batch size: `4`
  - Average frames per event: `3.38`
  - Average output tokens: `255.25`
  - Successful responses: `8`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 8}`
  - Wall-clock runtime: `63.88s`
  - Realtime ratio: `0.526x`

## Comparison

- Baseline HF Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Current Dynamic Selection Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=8, tokens=255.25, latency=63.88388100001612, success=8, failed=0