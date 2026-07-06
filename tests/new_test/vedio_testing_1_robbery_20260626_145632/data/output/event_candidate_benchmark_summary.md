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
  - Average output tokens: `149.20`
  - Successful responses: `66`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 66}`
  - Wall-clock runtime: `273.97s`
  - Realtime ratio: `0.262x`

## Comparison

- Baseline HF Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Current Dynamic Selection Pipeline: frames=None, tokens=None, latency=None, success=None, failed=None
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=66, tokens=149.1969696969697, latency=273.97283340001013, success=66, failed=0