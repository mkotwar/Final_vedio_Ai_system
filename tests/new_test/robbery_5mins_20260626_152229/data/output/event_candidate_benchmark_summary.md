# Event Candidate Benchmark Summary

- Input video: `C:\Mukul K\test_video\robbery_5mins.mp4`
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
  - Wall-clock runtime: `144.47s`
  - Realtime ratio: `0.436x`

## Comparison

- Baseline HF Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Current Dynamic Selection Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=21, tokens=256.0, latency=144.47355729999254, success=21, failed=0