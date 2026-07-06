# LEFT_PADDING_BENCHMARK

## Configuration

- Tokenizer padding before: `right`
- Tokenizer padding after: `left`
- Pad token before: `<|endoftext|>`
- Pad token after: `<|endoftext|>`
- EOS token: `<|im_end|>`
- `max_new_tokens`: `150`
- Changed variable only: `processor.tokenizer.padding_side = "left"`

## Conclusion

- Did changing only tokenizer left padding eliminate the batch-generation corruption? `True`
- Short answer: Yes. Left padding alone removed the batch-generation corruption in this benchmark.

## Before vs After

### candidate_only

- batch_size=1
  - success: `1 -> 1`
  - failed: `0 -> 0`
  - latency: `13.01s -> 2.67s`
  - warnings: `None -> 0`
  - failures before: `{'json_success': 1}`
  - failures after: `{'json_success': 1}`
- batch_size=4
  - success: `1 -> 1`
  - failed: `0 -> 0`
  - latency: `2.45s -> 2.25s`
  - warnings: `None -> 0`
  - failures before: `{'json_success': 1}`
  - failures after: `{'json_success': 1}`

### candidate_plus_periodic10s

- batch_size=1
  - success: `7 -> 7`
  - failed: `0 -> 0`
  - latency: `17.13s -> 15.41s`
  - warnings: `None -> 0`
  - failures before: `{'json_success': 7}`
  - failures after: `{'json_success': 7}`
- batch_size=4
  - success: `3 -> 7`
  - failed: `4 -> 0`
  - latency: `7.26s -> 6.64s`
  - warnings: `None -> 0`
  - failures before: `{'json_success': 3, 'non_json_garbage': 4}`
  - failures after: `{'json_success': 7}`

## Remaining Failures

- None.

## Warning Comparison

- candidate_only batch_size=1: warning_count=`0`, warnings=`[]`
- candidate_only batch_size=4: warning_count=`0`, warnings=`[]`
- candidate_plus_periodic10s batch_size=1: warning_count=`0`, warnings=`[]`
- candidate_plus_periodic10s batch_size=4: warning_count=`0`, warnings=`[]`