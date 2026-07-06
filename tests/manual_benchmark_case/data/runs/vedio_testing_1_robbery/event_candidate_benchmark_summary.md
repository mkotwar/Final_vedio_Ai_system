# Event Candidate Benchmark Summary

- Input video: `C:\Users\Vinfocom\Downloads\vedio_testing_1_robbery.mp4`
- Video duration: `1045.04s`
- Total frames extracted: `1046`
- Candidate events: `66`

## Modes

### candidate_only
- Variant `strip_tokens150_batch4`
  - Selected keyframes: `186`
  - Frames sent to Qwen: `66`
  - Batch size: `4`
  - Average frames per event: `2.82`
  - Average output tokens: `149.41`
  - Successful responses: `66`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 66}`
  - Wall-clock runtime: `282.14s`
  - Realtime ratio: `0.270x`

## Comparison

- Baseline HF Pipeline: frames=46, tokens=None, latency=194.56963429999814, success=13, failed=33
- Current Dynamic Selection Pipeline: frames=59, tokens=None, latency=293.46596149999823, success=59, failed=0
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=66, tokens=149.4090909090909, latency=282.1395298000134, success=66, failed=0