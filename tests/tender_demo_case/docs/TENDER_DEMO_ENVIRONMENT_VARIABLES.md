# Tender Demo Environment Variables

## Overview
The isolated tender-demo pipeline is configured almost entirely through environment variables. The Streamlit UI sets the most important ones automatically before it launches a pipeline script.

## Core pipeline control

| Variable | Used by | Meaning |
| --- | --- | --- |
| `TENDER_DEMO_INPUT_VIDEO` | standard, fast, UI | Absolute input video path. Required for actual runs. |
| `TENDER_DEMO_PIPELINE_ENGINE` | UI, fast metrics | Labels the selected pipeline family. |
| `TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE` | UI, Step 17, fast metrics | Human-readable preset label such as `Fast demo` or `Sensitive Incident Review`. |
| `TENDER_DEMO_MAX_VIDEO_SECONDS` | fast pipeline, UI | Optional cap for partial processing in fast mode. |

## Sampling, motion, and clip construction

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_SAMPLE_EVERY_SECONDS` | `1.0` standard, `3.0` fast defaults | Higher values reduce runtime and recall. Lower values increase coverage and cost. |
| `TENDER_DEMO_MOTION_THRESHOLD` | `0.20` | Higher values keep fewer frames. Lower values admit more motion candidates. |
| `TENDER_DEMO_MAX_GAP_SECONDS` | `2.0` | Maximum separation to group motion candidates into one clip. |
| `TENDER_DEMO_MAX_CLIP_SECONDS` | `12.0` | Maximum raw clip duration before splitting. |
| `TENDER_DEMO_CLIP_OVERLAP_SECONDS` | `2.0` | Overlap used when splitting long motion segments. |
| `TENDER_DEMO_CONTEXT_BEFORE_SECONDS` | `2.0` | Context added before each selected clip. |
| `TENDER_DEMO_CONTEXT_AFTER_SECONDS` | `2.0` | Context added after each selected clip. |
| `TENDER_DEMO_MIN_EXPANDED_CLIP_SECONDS` | `4.0` | Minimum expanded clip duration. |

## YOLO controls

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_RUN_YOLO` | `true` | Enables or skips Step 10 and downstream YOLO-driven evidence. |
| `TENDER_DEMO_YOLO_MODEL` | `yolov8n.pt` | Ultralytics model name or path. |
| `TENDER_DEMO_YOLO_CONF` | `0.25` | Detection confidence threshold. Lower values increase detections and noise. |
| `TENDER_DEMO_YOLO_IMGSZ` | `640` | Inference image size. Larger sizes help small objects but cost more time. |
| `TENDER_DEMO_YOLO_INPUT_SCOPE` | `motion_candidates` | Chooses whether YOLO runs on `04_motion_candidates.json` or `02_sampled_frames.json`. |

## Top-K selection controls

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_TOP_K_CLIPS` | `5` in Step 14, preset-specific in UI | Requested clip count for Qwen review. |
| `TENDER_DEMO_TOP_K_MAX_CLIPS` | `25` | Hard upper cap on effective Top-K. |
| `TENDER_DEMO_ENABLE_SELECTION_GUARDRAILS` | `true` | Enables guardrail-based clip insertion. |
| `TENDER_DEMO_HIGH_MOTION_GUARDRAIL_COUNT` | `3` | Number of high-motion clips to force in when guardrails apply. |
| `TENDER_DEMO_MIN_SELECTED_CLIPS` | `5` | Minimum selection fill target. |

## Qwen adapter and generation

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_QWEN_MODEL_ID` | adapter default alias | Hugging Face model ID or alias. |
| `TENDER_DEMO_QWEN_DEVICE` | auto | Optional explicit compute device. |
| `TENDER_DEMO_QWEN_BATCH_SIZE` | `1` | Batch size for generation. |
| `TENDER_DEMO_QWEN_MAX_NEW_TOKENS` | adapter default, preset-specific in UI | Major accuracy and latency lever. Too small may truncate output. |
| `TENDER_DEMO_QWEN_LOCAL_FILES_ONLY` | `false` | Restricts model loading to local cache only. |
| `TENDER_DEMO_FAST_COMPACT_QWEN_SCHEMA` | `true` in Step 16 fast path | Uses the flatter compact schema that is easier to parse. |
| `TENDER_DEMO_TEST_IMAGE` | none | Smoke-test image used by the adapter CLI test path. |

## Fast-pipeline orchestration

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_FAST_PARALLEL_BRANCHES` | `true` | Enables parallel clip branch and YOLO branch in the fast pipeline. |
| `TENDER_DEMO_QUICK_RESULT_MODE` | UI-controlled | Enables the most aggressive speed-oriented UI preset behavior. |

## Incident recheck and fallback pass

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_ENABLE_INCIDENT_RECHECK` | `false` | Enables Step 16B. |
| `TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK` | `false` | Rechecks all Top-K clips instead of a narrower subset. |
| `TENDER_DEMO_INCIDENT_FALLBACK_PASS` | `false` | Allows the fast pipeline to rerun Steps 14 to 19 with a larger Top-K when no suspicious or review clips were found. |
| `TENDER_DEMO_INCIDENT_FALLBACK_PASS_USED` | runtime-set | Indicates whether that fallback rerun actually occurred. |
| `TENDER_DEMO_INCIDENT_FALLBACK_REASON` | runtime-set | Human-readable explanation recorded into Step 17 summary. |

## Export and compiled-video settings

| Variable | Default | Effect |
| --- | --- | --- |
| `TENDER_DEMO_EXPORT_PRIORITY_CLIPS` | `true` | Export individual priority clips. |
| `TENDER_DEMO_EXPORT_REVIEW_CLIPS` | `true` | Export individual review clips. |
| `TENDER_DEMO_EXPORT_NORMAL_CLIPS` | `false` | Export individual normal clips. |
| `TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS` | `true` in fast defaults | If no suspicious clips exist, Step 18 can compile normal clips into the review video. |
| `TENDER_DEMO_EXPORT_FPS` | `5` | FPS for individual exported clip videos. |
| `TENDER_DEMO_EXPORT_FORMAT` | `mp4` | Target format for individual clip exports. |
| `TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO` | `true` | Enables compiled review video creation. |
| `TENDER_DEMO_COMPILED_VIDEO_FPS` | `5` | Playback FPS for the compiled review video. |
| `TENDER_DEMO_SECONDS_PER_FRAME` | `1.0` | How long each panel frame is shown in the compiled review video. |
| `TENDER_DEMO_SECONDS_PER_TITLE_CARD` | `1.5` | Title-card duration for each compiled clip entry. |
| `TENDER_DEMO_STRIP_PANEL_COUNT` | `3` | Number of panels expected per strip. |
| `TENDER_DEMO_COMPILED_FRAME_WIDTH` | `1280` | Output width for compiled review video frames. |
| `TENDER_DEMO_COMPILED_FRAME_HEIGHT` | `720` | Output height for compiled review video frames. |

## UI presets and what they imply

### Quick scan

- highest speed bias
- sparse sampling
- very small Top-K
- lower Qwen token budget
- no incident recheck

### Fast demo

- optimized demo default
- moderate sampling
- small Top-K
- no incident recheck

### Balanced

- middle ground between speed and coverage

### Sensitive Incident Review

- dense sampling
- Top-K 20
- incident recheck enabled for all Top-K clips
- fallback pass enabled

### High accuracy review

- densest sampling
- highest clip count
- highest cost among built-in presets

## Practical tuning guidance

### For speed

- increase `TENDER_DEMO_SAMPLE_EVERY_SECONDS`
- decrease `TENDER_DEMO_TOP_K_CLIPS`
- decrease `TENDER_DEMO_QWEN_MAX_NEW_TOKENS`
- use smaller `TENDER_DEMO_YOLO_IMGSZ`
- disable `TENDER_DEMO_ENABLE_INCIDENT_RECHECK`

### For better suspicious-incident recall

- decrease `TENDER_DEMO_SAMPLE_EVERY_SECONDS`
- lower `TENDER_DEMO_MOTION_THRESHOLD`
- increase `TENDER_DEMO_TOP_K_CLIPS`
- increase `TENDER_DEMO_QWEN_MAX_NEW_TOKENS`
- enable `TENDER_DEMO_ENABLE_INCIDENT_RECHECK`
- enable `TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK`
- enable `TENDER_DEMO_INCIDENT_FALLBACK_PASS`

## Variables most likely to affect runtime

Ordered roughly from highest impact to lower impact:

1. `TENDER_DEMO_ENABLE_INCIDENT_RECHECK`
2. `TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK`
3. `TENDER_DEMO_TOP_K_CLIPS`
4. `TENDER_DEMO_QWEN_MAX_NEW_TOKENS`
5. `TENDER_DEMO_SAMPLE_EVERY_SECONDS`
6. `TENDER_DEMO_YOLO_IMGSZ`
7. `TENDER_DEMO_RUN_YOLO`

## Variables most likely to affect parse stability

- `TENDER_DEMO_FAST_COMPACT_QWEN_SCHEMA`
- `TENDER_DEMO_QWEN_MAX_NEW_TOKENS`
- `TENDER_DEMO_TOP_K_CLIPS`

The current Step 16 design intentionally favors the compact schema because it is easier for the model to complete within low token budgets.
