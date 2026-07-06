# Tender Demo Latency And Performance

## Main performance story
The isolated tender-demo pipeline is mostly cheap until it reaches Qwen. In both fast and incident-sensitive runs, VLM reasoning dominates total runtime by a large margin.

## Observed run examples

### Fast road-scene run
Observed folder: `debug_runs/localcam2_20260702_163729`

- video duration: `63.633s`
- total runtime: `81.647s`
- runtime/video ratio: `1.283x`
- Step 16 runtime: `74.14s`

Older fast-run metrics from this folder show:

- Steps 7, 8, 9, and 12 skipped
- Step 16 as the clear bottleneck

### Newer fast road-scene quick run
Observed folder: `debug_runs/localcam2_20260702_183043`

- video duration: `63.633s`
- total runtime: `87.732s`
- runtime/video ratio: `1.379x`
- Step 16 runtime: `78.543s`
- settings included `sample_every_seconds=4.0`, `top_k=3`, `qwen_max_new_tokens=192`

### Incident-sensitive shop run
Observed folder: `debug_runs/robbery_5mins_20260703_110934`

- video duration: `331.367s`
- total runtime: `416.785s`
- runtime/video ratio: `1.258x`
- Step 16 runtime: `161.506s`
- Step 16B runtime: `214.898s`

This run shows that incident recheck can cost more than the primary Top-K Qwen pass.

## Why the fast pipeline works well

### Cost shifted away from full-video VLM
The old expensive path was:

- create strips for every expanded clip
- run Qwen on every strip

The fast path instead:

- uses motion and YOLO first
- ranks clips
- sends only Top-K clips to Qwen

### Parallel branches reduce idle time
The fast pipeline overlaps:

- clip grouping and expansion
- YOLO detection, scoring, and motion-state estimation

This helps keep non-VLM work out of the critical path.

## Where the time goes

### Usually cheap

- Step 1 video metadata
- Step 3 motion scoring
- Step 4 motion candidate selection
- Step 13 ranking
- Step 14 selection
- Step 17 summary generation
- Step 19 HTML generation

### Moderately expensive

- Step 2 frame sampling on longer videos
- Step 10 YOLO
- Step 11 YOLO scoring
- Step 15 strip generation
- Step 18 video export and compile

### Usually dominant

- Step 16 Qwen on Top-K clips
- Step 16B incident recheck when enabled

## Biggest runtime levers

### `TENDER_DEMO_TOP_K_CLIPS`
This directly controls how many strips reach Step 16, and often Step 16B too.

### `TENDER_DEMO_QWEN_MAX_NEW_TOKENS`
More tokens can improve richer reasoning but increase generation time and risk long outputs.

### `TENDER_DEMO_SAMPLE_EVERY_SECONDS`
Lower intervals produce:

- more sampled frames
- more motion candidates
- more clip windows
- potentially more YOLO work
- potentially a different Top-K set

### Incident recheck flags

- `TENDER_DEMO_ENABLE_INCIDENT_RECHECK`
- `TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK`

These can significantly increase runtime on longer videos.

### YOLO image size
Higher `TENDER_DEMO_YOLO_IMGSZ` improves small-object sensitivity but increases YOLO cost.

## Accuracy versus speed tradeoffs

### Faster settings

- higher sample interval
- smaller Top-K
- fewer Qwen tokens
- no incident recheck
- smaller YOLO image size

Likely result:

- faster runs
- lower recall for subtle events
- less descriptive summaries

### More sensitive settings

- lower sample interval
- larger Top-K
- larger token budget
- incident recheck enabled
- lower motion threshold

Likely result:

- better chance of catching subtle suspicious behavior
- significantly slower runtime

## Parse quality and performance
Step 16 quality is not only about model intelligence. It is also about whether the chosen schema fits into the token budget.

The current compact flat schema improves performance indirectly by:

- making generation shorter
- reducing JSON truncation risk
- reducing parser failures

That means a shorter schema can produce a better end-to-end summary even when the model itself is unchanged.

## Compiled video export cost
Step 18 cost depends on:

- number of selected clips
- panel count
- title-card duration
- frame duration
- whether FFmpeg is available
- whether AVI fallback conversion to browser MP4 is needed

Direct FFmpeg MP4 export is typically cleaner for playback. AVI fallback is reliable for local artifact preservation but may need an extra conversion pass for browser playback.

## Current performance limitations

- Qwen remains serial and expensive at `batch_size=1`.
- Incident recheck can double the reasoning cost.
- The pipeline is local-process oriented and does not distribute work.
- Standard pipeline still retains the older full-VLM path, which is much less efficient.

## Practical recommendations

### Best default for demos
Use `Fast demo` or `Balanced`.

### Best default for suspicious-incident review
Use `Sensitive Incident Review`, but expect much longer runtime.

### Best lever when runtime is too high
Reduce `Top-K` first, then reduce `Qwen max new tokens`, then increase the sample interval.
