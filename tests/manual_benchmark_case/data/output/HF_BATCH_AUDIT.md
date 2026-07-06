# HF Batch Generation Audit for Qwen2.5-VL

## Scope

- Input set: `candidate_only/strip_tokens150_batch4` saved benchmark prompts and images
- Candidate prompt/image pairs audited: `5`
- Source summary: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\event_candidate_benchmark_summary.json`

## 1. Tokenizer Audit

- `padding_side`: `right`
- `pad_token`: `<|endoftext|>`
- `pad_token_id`: `151643`
- `eos_token`: `<|im_end|>`
- `eos_token_id`: `151645`
- Assignment in project code: `No assignment found in project code; inherited from AutoProcessor.from_pretrained().`
- Later mutation detected in project code: `False`

## 2. Processor Audit

- Processor init site: `app/services/qwen_vlm_hf.py:147`
- Processor call site: `app/services/qwen_vlm_hf.py:226`
- Processor call args: `{'text': 'texts', 'images': 'image_inputs', 'videos': 'video_inputs', 'padding': True, 'return_tensors': 'pt'}`
- Returned keys in audited runs: `['attention_mask', 'image_grid_thw', 'input_ids', 'mm_token_type_ids', 'pixel_values']`
- Attention mask present: `True`

## 3. Generate Audit

- Generate site: `app/services/qwen_vlm_hf.py:244`
- Generate receives `input_ids`: `True`
- Generate receives `attention_mask`: `True`
- Generate receives `pixel_values`: `True`
- Generate receives `image_grid_thw`: `True`
- Extra kwargs shape source: ``**inputs` from processor output`

## 4. Batching Audit

- Prompt list construction site: `app/services/qwen_vlm_hf.py:186`
- Image/prompt zip site: `app/services/qwen_vlm_hf.py:191`
- Output zip site: `Outputs are returned from generate_batch() in the same row order as image_paths/prompt_list; benchmark harness then consumes them in order.`
- Ordering issue detected: `False`
- Batch4 execution groups: `[['cand_evt_001', 'cand_evt_002', 'cand_evt_003', 'cand_evt_004'], ['cand_evt_005']]`

## 5. Decode Audit

- Decode site: `app/services/qwen_vlm_hf.py:265`
- `skip_special_tokens`: `True`
- `clean_up_tokenization_spaces`: `False`
- Decode-only bug detected: `False`

## 6. Output Slicing Audit

- Current trim site: `app/services/qwen_vlm_hf.py:261`
- Current trim basis: ``len(in_ids)` per row after processor padding`
- Alternative trim compared: ``attention_mask.sum()` per row`
- Current-vs-attention decode mismatch detected: `False`
- Assessment: `No issue detected.`

## 7. Vision Processing Audit

- `process_vision_info` source: `C:\Mukul K\vinfo1\video-search-engine\.venv\Lib\site-packages\qwen_vl_utils\vision_process.py`
- Vision call site: `app/services/qwen_vlm_hf.py:215`
- Variable image grid across samples: `True`
- Assessment: `Variable image_grid_thw values are present across samples, but no standalone vision batching bug was isolated in this audit.`

## 8. Batch1 vs Batch4 Comparison

### batch1

- Run 1: batch_size=1, events=['cand_evt_001'], generate_seconds=5.58
  - cand_evt_001: prompt_tokens=202, chat_tokens=224, padded_len=703, valid_len=703, gen_tokens_current=49, decoded_len=175, json_success=True
- Run 2: batch_size=1, events=['cand_evt_002'], generate_seconds=4.70
  - cand_evt_002: prompt_tokens=198, chat_tokens=220, padded_len=651, valid_len=651, gen_tokens_current=49, decoded_len=179, json_success=True
- Run 3: batch_size=1, events=['cand_evt_003'], generate_seconds=4.78
  - cand_evt_003: prompt_tokens=178, chat_tokens=200, padded_len=679, valid_len=679, gen_tokens_current=49, decoded_len=175, json_success=True
- Run 4: batch_size=1, events=['cand_evt_004'], generate_seconds=2.26
  - cand_evt_004: prompt_tokens=170, chat_tokens=192, padded_len=671, valid_len=671, gen_tokens_current=49, decoded_len=175, json_success=True
- Run 5: batch_size=1, events=['cand_evt_005'], generate_seconds=2.19
  - cand_evt_005: prompt_tokens=175, chat_tokens=197, padded_len=628, valid_len=628, gen_tokens_current=49, decoded_len=180, json_success=True

### batch4

- Run 1: batch_size=4, events=['cand_evt_001', 'cand_evt_002', 'cand_evt_003', 'cand_evt_004'], generate_seconds=5.85
  - cand_evt_001: prompt_tokens=202, chat_tokens=224, padded_len=703, valid_len=703, gen_tokens_current=49, decoded_len=175, json_success=True
  - cand_evt_002: prompt_tokens=198, chat_tokens=220, padded_len=703, valid_len=651, gen_tokens_current=49, decoded_len=14, json_success=False
  - cand_evt_003: prompt_tokens=178, chat_tokens=200, padded_len=703, valid_len=679, gen_tokens_current=49, decoded_len=0, json_success=False
  - cand_evt_004: prompt_tokens=170, chat_tokens=192, padded_len=703, valid_len=671, gen_tokens_current=49, decoded_len=4, json_success=False
- Run 2: batch_size=4, events=['cand_evt_005'], generate_seconds=4.78
  - cand_evt_005: prompt_tokens=175, chat_tokens=197, padded_len=628, valid_len=628, gen_tokens_current=49, decoded_len=180, json_success=True

## 9. Identified Issues Ranked by Confidence

- Right-padding in batched decoder-only generation is the top root-cause candidate.
  - Confidence: `high`
  - Why: Tokenizer padding_side is right, the HF warning is triggered only on batch sizes above 1, batch_size=1 succeeds on all 5 samples while batch_size=4 succeeds on only 2/5 for the same prompt/image pairs.
  - Proposed fix: Initialize the tokenizer for left padding before batched generate().
- Attention mask is present and forwarded into generate().
  - Confidence: `high`
  - Why: Processor returns attention_mask and model.generate receives it via **inputs; no dropped-mask bug was detected.
  - Proposed fix: No fix needed here.
- Prompt/image ordering is preserved through batching.
  - Confidence: `high`
  - Why: The batch is built by zipping image_paths and prompt_list in order, and outputs are consumed with zip(batch, raw_outputs).
  - Proposed fix: No fix needed here.
- Current output slicing is probably not the main failure cause.
  - Confidence: `medium`
  - Why: Current trim and attention-sum trim decode identically across the audited batch-4 runs.
  - Proposed fix: Revisit trim logic after the padding fix only if failures remain.
- The fifth candidate in batch_size=4 is effectively a single-sample run.
  - Confidence: `high`
  - Why: With 5 jobs and batch_size=4, the benchmark executes one batch of 4 and one batch of 1; the trailing single-item batch succeeds, reinforcing that corruption is tied to multi-sample batching.
  - Proposed fix: No fix needed; this is diagnostic evidence.

## 10. Exact Code Locations Requiring Changes

- [qwen_vlm_hf.py](/C:\Mukul K\vinfo1\video-search-engine\app\services\qwen_vlm_hf.py:147): tokenizer/processor initialization point
- [qwen_vlm_hf.py](/C:\Mukul K\vinfo1\video-search-engine\app\services\qwen_vlm_hf.py:226): processor batch construction with `padding=True`
- [qwen_vlm_hf.py](/C:\Mukul K\vinfo1\video-search-engine\app\services\qwen_vlm_hf.py:244): `model.generate(**inputs, ...)` batched call
- [qwen_vlm_hf.py](/C:\Mukul K\vinfo1\video-search-engine\app\services\qwen_vlm_hf.py:261): generated token slicing
- [utils.py](/C:\Mukul K\vinfo1\video-search-engine\.venv\Lib\site-packages\transformers\generation\utils.py:2457): HF right-padding warning trigger

If no issue is found in a section above, it is explicitly marked as no issue detected.