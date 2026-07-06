# VLM Routing Map

- Input video: `C:\Mukul K\test_video\V_ai_test_2min.mp4`
- Primary mode/variant: `candidate_only / strip_tokens150_batch1`
- Extracted frames: `139`
- Candidate events: `5`
- Raw selected frames used to build VLM context: `18`
- Actual Qwen image inputs: `5`
- Raw selected frame video: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\vlm_raw_selected_frames.mp4`
- Reasoning input video: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\vlm_reasoning_inputs_video.mp4`

## Event Order

### cand_evt_001
- Window: `00:00:19 -> 00:00:34`
- Heuristic type: `group_activity`
- Selected frame ids: `['11111111-2222-4333-8444-777777777777_f0020', '11111111-2222-4333-8444-777777777777_f0023', '11111111-2222-4333-8444-777777777777_f0034', '11111111-2222-4333-8444-777777777777_f0035']`
- Qwen input image: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\reasoning_inputs\candidate_only_strip_tokens150_batch1_cand_evt_001.jpg`

### cand_evt_002
- Window: `00:00:40 -> 00:00:45`
- Heuristic type: `movement_activity`
- Selected frame ids: `['11111111-2222-4333-8444-777777777777_f0041', '11111111-2222-4333-8444-777777777777_f0045', '11111111-2222-4333-8444-777777777777_f0046']`
- Qwen input image: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\reasoning_inputs\candidate_only_strip_tokens150_batch1_cand_evt_002.jpg`

### cand_evt_003
- Window: `00:00:53 -> 00:01:13`
- Heuristic type: `group_activity`
- Selected frame ids: `['11111111-2222-4333-8444-777777777777_f0054', '11111111-2222-4333-8444-777777777777_f0069', '11111111-2222-4333-8444-777777777777_f0073', '11111111-2222-4333-8444-777777777777_f0074']`
- Qwen input image: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\reasoning_inputs\candidate_only_strip_tokens150_batch1_cand_evt_003.jpg`

### cand_evt_004
- Window: `00:01:22 -> 00:01:44`
- Heuristic type: `group_activity`
- Selected frame ids: `['11111111-2222-4333-8444-777777777777_f0083', '11111111-2222-4333-8444-777777777777_f0096', '11111111-2222-4333-8444-777777777777_f0104', '11111111-2222-4333-8444-777777777777_f0105']`
- Qwen input image: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\reasoning_inputs\candidate_only_strip_tokens150_batch1_cand_evt_004.jpg`

### cand_evt_005
- Window: `00:02:01 -> 00:02:03`
- Heuristic type: `movement_activity`
- Selected frame ids: `['11111111-2222-4333-8444-777777777777_f0122', '11111111-2222-4333-8444-777777777777_f0123', '11111111-2222-4333-8444-777777777777_f0124']`
- Qwen input image: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\reasoning_inputs\candidate_only_strip_tokens150_batch1_cand_evt_005.jpg`
