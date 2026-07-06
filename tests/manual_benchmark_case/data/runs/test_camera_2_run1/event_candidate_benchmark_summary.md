# Event Candidate Benchmark Summary

- Input video: `C:\Mukul K\test_video\test_camera_2.mp4`
- Video duration: `249.90s`
- Total frames extracted: `250`
- Candidate events: `11`

## Modes

### candidate_only
- Variant `strip_tokens150_batch4`
  - Selected keyframes: `30`
  - Frames sent to Qwen: `11`
  - Batch size: `4`
  - Average frames per event: `2.73`
  - Average output tokens: `149.18`
  - Successful responses: `11`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 11}`
  - Wall-clock runtime: `54.93s`
  - Realtime ratio: `0.220x`

## Comparison

- Baseline HF Pipeline: frames=46, tokens=None, latency=194.56963429999814, success=13, failed=33
- Current Dynamic Selection Pipeline: frames=59, tokens=None, latency=293.46596149999823, success=59, failed=0
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=11, tokens=149.1818181818182, latency=54.92899909999687, success=11, failed=0