# Event Candidate Benchmark Summary

- Input video: `C:\Mukul K\test_video\V_ai_test_2min.mp4`
- Video duration: `138.57s`
- Total frames extracted: `139`
- Candidate events: `5`

## Modes

### candidate_only
- Variant `strip_tokens150_batch1`
  - Selected keyframes: `18`
  - Frames sent to Qwen: `5`
  - Batch size: `1`
  - Average frames per event: `3.60`
  - Average output tokens: `48.00`
  - Successful responses: `5`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 5}`
  - Wall-clock runtime: `25.21s`
  - Realtime ratio: `0.182x`

- Variant `strip_tokens150_batch4`
  - Selected keyframes: `18`
  - Frames sent to Qwen: `5`
  - Batch size: `4`
  - Average frames per event: `3.60`
  - Average output tokens: `19.80`
  - Successful responses: `2`
  - Failed responses: `3`
  - Failure breakdown: `{'json_success': 2, 'non_json_garbage': 1, 'empty_response': 1, 'non_english_garbage': 1}`
  - Wall-clock runtime: `5.95s`
  - Realtime ratio: `0.043x`

- Variant `single_peak_tokens150_batch1`
  - Selected keyframes: `5`
  - Frames sent to Qwen: `5`
  - Batch size: `1`
  - Average frames per event: `1.00`
  - Average output tokens: `48.00`
  - Successful responses: `5`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 5}`
  - Wall-clock runtime: `11.26s`
  - Realtime ratio: `0.081x`

- Variant `single_peak_tokens150_batch4`
  - Selected keyframes: `5`
  - Frames sent to Qwen: `5`
  - Batch size: `4`
  - Average frames per event: `1.00`
  - Average output tokens: `20.20`
  - Successful responses: `2`
  - Failed responses: `3`
  - Failure breakdown: `{'json_success': 2, 'non_json_garbage': 3}`
  - Wall-clock runtime: `5.38s`
  - Realtime ratio: `0.039x`

### candidate_plus_periodic10s
- Variant `strip_tokens150_batch1`
  - Selected keyframes: `18`
  - Frames sent to Qwen: `19`
  - Batch size: `1`
  - Average frames per event: `3.60`
  - Average output tokens: `49.95`
  - Successful responses: `19`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 19}`
  - Wall-clock runtime: `47.95s`
  - Realtime ratio: `0.346x`

- Variant `strip_tokens150_batch4`
  - Selected keyframes: `18`
  - Frames sent to Qwen: `19`
  - Batch size: `4`
  - Average frames per event: `3.60`
  - Average output tokens: `19.84`
  - Successful responses: `7`
  - Failed responses: `12`
  - Failure breakdown: `{'json_success': 7, 'non_json_garbage': 10, 'empty_response': 1, 'non_english_garbage': 1}`
  - Wall-clock runtime: `18.67s`
  - Realtime ratio: `0.135x`

- Variant `single_peak_tokens150_batch1`
  - Selected keyframes: `5`
  - Frames sent to Qwen: `19`
  - Batch size: `1`
  - Average frames per event: `1.00`
  - Average output tokens: `49.89`
  - Successful responses: `19`
  - Failed responses: `0`
  - Failure breakdown: `{'json_success': 19}`
  - Wall-clock runtime: `45.50s`
  - Realtime ratio: `0.328x`

- Variant `single_peak_tokens150_batch4`
  - Selected keyframes: `5`
  - Frames sent to Qwen: `19`
  - Batch size: `4`
  - Average frames per event: `1.00`
  - Average output tokens: `20.00`
  - Successful responses: `7`
  - Failed responses: `12`
  - Failure breakdown: `{'json_success': 7, 'non_json_garbage': 12}`
  - Wall-clock runtime: `17.43s`
  - Realtime ratio: `0.126x`

## Comparison

- Baseline HF Pipeline: frames=46, tokens=None, latency=194.56963429999814, success=13, failed=33
- Current Dynamic Selection Pipeline: frames=59, tokens=None, latency=293.46596149999823, success=59, failed=0
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch1): frames=5, tokens=48.0, latency=25.208116999994672, success=5, failed=0
- Event-Candidate Reasoning (candidate_only/strip_tokens150_batch4): frames=5, tokens=19.8, latency=5.9457116999983555, success=2, failed=3
- Event-Candidate Reasoning (candidate_only/single_peak_tokens150_batch1): frames=5, tokens=48.0, latency=11.255448199997772, success=5, failed=0
- Event-Candidate Reasoning (candidate_only/single_peak_tokens150_batch4): frames=5, tokens=20.2, latency=5.377710599997954, success=2, failed=3
- Event-Candidate Reasoning (candidate_plus_periodic10s/strip_tokens150_batch1): frames=19, tokens=49.94736842105263, latency=47.946188599999005, success=19, failed=0
- Event-Candidate Reasoning (candidate_plus_periodic10s/strip_tokens150_batch4): frames=19, tokens=19.842105263157894, latency=18.669558000001416, success=7, failed=12
- Event-Candidate Reasoning (candidate_plus_periodic10s/single_peak_tokens150_batch1): frames=19, tokens=49.89473684210526, latency=45.50168059999851, success=19, failed=0
- Event-Candidate Reasoning (candidate_plus_periodic10s/single_peak_tokens150_batch4): frames=19, tokens=20.0, latency=17.42509959999734, success=7, failed=12