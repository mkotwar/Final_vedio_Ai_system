# Event Candidate Benchmark Summary

- Input video: `C:\Mukul K\test_video\robbery_5mins.mp4`
- Video duration: `331.37s`
- Total frames extracted: `332`
- Candidate events: `21`

## Modes

### candidate_only
- Variant `strip_eventsfirst_tokens512_batch4`
  - Selected keyframes: `65`
  - Frames sent to Qwen: `21`
  - Batch size: `4`
  - Average frames per event: `3.10`
  - Average output tokens: `256.00`
  - Successful responses: `21`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 21}`
  - Wall-clock runtime: `160.96s`
  - Realtime ratio: `0.486x`

## Comparison

- Baseline HF Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Current Dynamic Selection Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Event-Candidate Reasoning (candidate_only/strip_eventsfirst_tokens512_batch4): frames=21, tokens=256.0, latency=160.9585118999821, success=21, failed=0