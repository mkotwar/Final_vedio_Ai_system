# Event Candidate Benchmark Summary

- Input video: `C:\Users\Vinfocom\Downloads\jwellery_theft.mp4`
- Video duration: `331.37s`
- Total frames extracted: `332`
- Candidate events: `21`

## Modes

### candidate_only
- Variant `strip_tokens150_batch4`
  - Selected keyframes: `65`
  - Frames sent to Qwen: `21`
  - Batch size: `4`
  - Average frames per event: `3.10`
  - Average output tokens: `256.00`
  - Successful responses: `21`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 21}`
  - Wall-clock runtime: `167.32s`
  - Realtime ratio: `0.505x`

## Comparison

- Baseline HF Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Current Dynamic Selection Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=21, tokens=256.0, latency=167.3153293999785, success=21, failed=0