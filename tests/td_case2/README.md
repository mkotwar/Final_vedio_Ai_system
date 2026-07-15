# td_case2

This testcase is an isolated tender demo sandbox under `tests/td_case2`.

It started with the early foundation stages:

1. Read video metadata with OpenCV.
2. Sample full-scene frames at a fixed time interval.

It now also contains later isolated step runners, including Step 08 query search validation on top of the Step 07 vehicle search index output.
It also includes Step 09 search result packaging, which converts Step 07 and Step 08 outputs into UI/demo-ready result cards.
It also includes Step 10 live search demo running, which behaves like a small API-style query layer on top of the existing JSON outputs.
It also includes Step 11 full-scene event candidate generation, which creates rule-based candidate events from the existing motion, detection, and tracking outputs.
It also includes Step 12 event candidate ranking, which reduces the larger Step 11 candidate set into a smaller Top-K list for later VLM review.
It also includes Step 13 VLM input generation, which turns the selected event candidates into full-scene temporal strip packages for later review.

It does not modify production code or `tests/tender_demo_case`.

## Local development setup

Use Python 3.11 and pip. From the repository root:

```powershell
py -3.11 -m venv tests\td_case2\.venv
tests\td_case2\.venv\Scripts\python.exe -m pip install -r tests\td_case2\requirements-cuda.txt
tests\td_case2\.venv\Scripts\python.exe -m pip install -r tests\td_case2\requirements.txt
Copy-Item tests\td_case2\.env.example tests\td_case2\.env
```

Edit `tests/td_case2/.env` and set `TD_CASE2_INPUT_VIDEO` to an absolute local
video path. The checked-in `yolo11m.pt` is configured as the portable combined
YOLO fallback. The complete search-ready pipeline additionally needs
`TD_CASE2_FLORENCE_MODEL_PATH` to point to a local Florence-2 model directory.
Relative YOLO model paths in this file are resolved from `tests/td_case2`, so
the default `../../yolo11m.pt` resolves to the repository-root model.
For interactive commands launched from the repository root, existing paths such
as `yolo11m.pt` and `OCR_MUKUL/license_plate_weights.pt` are also accepted.
The plate detector is configured separately through
`TD_CASE2_PLATE_DETECTOR_MODEL_PATH`; Florence OCR runs on its detected plate
crops as well as on selected vehicle crops.

`td_case2` is now GPU-first by default. If `torch.cuda.is_available()` is true,
YOLO, the plate detector, Florence, and both local Qwen stages automatically
use CUDA without any manual device environment variables. CPU is used only when
CUDA is unavailable, when a stage is CPU-only by design, or when you explicitly
override the default with `TD_CASE2_DEVICE=cpu` or a per-stage device override.

Each run writes `gpu_utilization_report.json`, which records the resolved device
for every GPU-capable component plus the stages that remain CPU-bound. Motion
sampling, deterministic tracking association, JSON enrichment, search packaging,
and result ranking remain CPU work in the current td_case2 implementation.

The default `TD_CASE2_VLM_BACKEND=disabled` runs later stages without a Qwen
model or API key. Set it to `local_qwen` and provide the local Qwen model paths,
or set it to `api_qwen` and provide `TD_CASE2_QWEN_API_KEY`, when VLM review is
needed.

The `local_qwen` backend uses strict BitsAndBytes 4-bit NF4 loading for both
the Step 11.5 3B filter and Step 14 7B reviewer. CUDA and `bitsandbytes` are
required; these stages do not fall back to FP16, BF16, or CPU inference.
The default local checkpoints are the pre-quantized Hugging Face repositories
downloaded under `models/Qwen2.5-VL-3B-Instruct-bnb-4bit` and
`models/Qwen2.5-VL-7B-Instruct-bnb-4bit`.

Start the Streamlit UI with:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m streamlit run tests\td_case2\td_case2_traffic_search_ui.py
```

For the integrated all-in-one workbench UI, use:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m streamlit run tests\td_case2\td_case2_workbench_ui.py
```

`td_case2_results_ui.py` now opens the integrated workbench by default, with the VLM Summary section selected first.

`td_case2` also exposes Streamlit sidebar pages under `tests/td_case2/pages/`, so you can switch between:

- `Workbench`
- `Search And Clips`
- `VLM Summary`

## What it does

The runner reads one local video path from an environment variable, creates a new debug run folder, writes `01_video_info.json`, saves sampled full-frame images into `02_sampled_frames/`, writes `02_sampled_frames.json`, and updates `00_stage_gate_report.json`.

Supported video extensions include:

`.AVI`, `.MKV`, `.MPEG4`, `.MOV`, `.WMV`, `.DVR`, `.ASF`, `.RT4`, `.DIVX`, `.264`, `.H264`, `.H265`, `.GE5`, `.TS`, `.3GP`, and `.mp4`.

## How to run it

PowerShell example:

```powershell
cd "C:\Mukul K\vinfo1\video-search-engine"

$env:TD_CASE2_INPUT_VIDEO="C:\Users\Vinfocom\Downloads\anpr_test_5min.mp4"
$env:TD_CASE2_SAMPLE_EVERY_SECONDS="1.0"

python tests\td_case2\run_td_case2_step01_02.py
```

Complete search-ready pipeline:

```powershell
tests\td_case2\.venv\Scripts\python.exe tests\td_case2\run_td_case2_search_ready_pipeline.py
```

This runner creates a fresh run directory and sets `TD_CASE2_RUN_DIR` for its
downstream steps internally. Any stale `TD_CASE2_RUN_DIR` value in the shell or
local `.env` is replaced with the newly created directory.

If Step 03 has already completed, resume at Step 04A without repeating sampling
or YOLO detection:

```powershell
tests\td_case2\.venv\Scripts\python.exe tests\td_case2\run_td_case2_search_ready_pipeline.py --resume-run-dir "debug_runs\<existing-run>"
```

When an inherited `TD_CASE2_FLORENCE_MODEL_PATH` points to a missing placeholder
path, Steps 04A and 06 use a valid model path from `tests/td_case2/.env`.

Complete VLM + evidence pipeline on an existing search-ready run:

```powershell
$env:TD_CASE2_RUN_DIR="C:\path\to\debug_runs\<existing-run>"
tests\td_case2\.venv\Scripts\python.exe tests\td_case2\run_td_case2_vlm_event_pipeline.py
```

Run Step 16 evidence video generation by itself on an existing VLM-ready run:

```powershell
$env:TD_CASE2_RUN_DIR="C:\path\to\debug_runs\<existing-run>"
tests\td_case2\.venv\Scripts\python.exe tests\td_case2\run_td_case2_step16_evidence_video.py
```

Optional custom output root:

```powershell
$env:TD_CASE2_OUTPUT_ROOT="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs"
```

Step 08 query validation example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"
$env:TD_CASE2_STEP08_ALLOW_WEAK_OCR_SEARCH="true"
$env:TD_CASE2_STEP08_TIME_TOLERANCE_SECONDS="3.0"
$env:TD_CASE2_STEP08_TOP_K="20"

python tests\td_case2\run_td_case2_step08_query_validation.py
```

Step 09 result packaging example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"
$env:TD_CASE2_STEP09_TOP_K="10"

python tests\td_case2\run_td_case2_step09_result_packaging.py
```

Step 10 live search demo example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"

python tests\td_case2\run_td_case2_step10_search_demo.py --query "DL12CL4316"
```

Step 11 full-scene event candidate generation example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"

python tests\td_case2\run_td_case2_step11_event_candidates.py
```

Step 12 event candidate ranking example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"

python tests\td_case2\run_td_case2_step12_event_ranking.py
```

Step 13 VLM input generation example:

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"

python tests\td_case2\run_td_case2_step13_vlm_inputs.py
```

## Environment variables

- `TD_CASE2_INPUT_VIDEO`
  Required. Full local path to the input video.
- `TD_CASE2_SAMPLE_EVERY_SECONDS`
  Optional. Default is `1.0`.
  `1.0` means one frame every second.
  `0.5` means two frames every second.
  `2.0` means one frame every two seconds.
- `TD_CASE2_OUTPUT_ROOT`
  Optional. Default is `tests/td_case2/debug_runs`.
- `TD_CASE2_RUN_DIR`
  Required for later step runners such as Step 08. It must point to an existing `tests/td_case2/debug_runs/<run_folder>` directory.
- `TD_CASE2_STEP08_ALLOW_WEAK_OCR_SEARCH`
  Optional. Default is `true`. Enables weak OCR candidate matching during Step 08 validation.
- `TD_CASE2_STEP08_TIME_TOLERANCE_SECONDS`
  Optional. Default is `3.0`. Controls timestamp query tolerance in Step 08.
- `TD_CASE2_STEP08_TOP_K`
  Optional. Default is `20`. Limits returned matches per Step 08 query.
- `TD_CASE2_STEP08_FAIL_ON_CRITICAL_TEST_FAILURE`
  Optional. Default is `false`. When `true`, Step 08 returns `failed` if critical validation groups fail.
- `TD_CASE2_STEP08_SAVE_MATCH_PREVIEW`
  Optional. Default is `true`. Reserved Step 08 flag for match preview output handling.
- `TD_CASE2_STEP09_TOP_K`
  Optional. Default is `10`. Limits cards returned per demo query package.
- `TD_CASE2_STEP09_INCLUDE_WEAK_OCR`
  Optional. Default is `true`. Includes possible / weak OCR metadata in the packaged result cards.
- `TD_CASE2_STEP09_INCLUDE_INVALID_DEBUG_FIELDS`
  Optional. Default is `false`. When enabled, invalid OCR values are exposed only inside the debug block.
- `TD_CASE2_STEP09_INCLUDE_DEBUG_PATHS`
  Optional. Default is `true`. Keeps debug media path fields in card output.
- `TD_CASE2_STEP09_VALIDATE_PATH_STRINGS`
  Optional. Default is `true`. Checks whether packaged media paths exist under the run directory.
- `TD_CASE2_STEP09_BUILD_DEMO_QUERIES`
  Optional. Default is `true`. Builds demo query packages from Step 08 query matches.
- `TD_CASE2_STEP09_RESULT_CARD_VERSION`
  Optional. Default is `v1`. Sets the packaged result-card schema version.
- `TD_CASE2_SEARCH_QUERY`
  Optional. Single live query string for Step 10.
- `TD_CASE2_SEARCH_QUERIES`
  Optional. JSON list of query strings for Step 10 batch mode.
- `TD_CASE2_SEARCH_MODE`
  Optional. Default is `auto`. Allowed values are `auto`, `exact_plate`, `color_class`, `class_only`, `color_only`, `timestamp`, `weak_ocr`, and `combined`.
- `TD_CASE2_SEARCH_TOP_K`
  Optional. Default is `10`. Limits returned cards per query in Step 10.
- `TD_CASE2_SEARCH_ALLOW_WEAK_OCR`
  Optional. Default is `true`. Allows weak OCR fallback/search in Step 10.
- `TD_CASE2_SEARCH_TIME_TOLERANCE_SECONDS`
  Optional. Default is `3.0`. Used for timestamp and combined Step 10 search.
- `TD_CASE2_SEARCH_INCLUDE_FALLBACK`
  Optional. Default is `true`. Includes fallback tracks in Step 10 search results.
- `TD_CASE2_SEARCH_REQUIRE_IMAGE_PATHS`
  Optional. Default is `true`. Filters out cards missing crop or full-frame media paths.
- `TD_CASE2_SEARCH_SAVE_DEBUG`
  Optional. Default is `true`. Keeps debug metadata on returned Step 10 cards.
- `TD_CASE2_STEP11_WINDOW_SECONDS`
  Optional. Default is `2.0`. Sliding scene window size for event candidate generation.
- `TD_CASE2_STEP11_WINDOW_STRIDE_SECONDS`
  Optional. Default is `1.0`. Sliding scene window stride for Step 11.
- `TD_CASE2_STEP11_MERGE_GAP_SECONDS`
  Optional. Default is `3.0`. Merge gap for nearby raw triggers.
- `TD_CASE2_STEP11_MAX_EVENT_SECONDS`
  Optional. Default is `12.0`. Max merged candidate event duration.
- `TD_CASE2_STEP11_CONTEXT_BEFORE_SECONDS`
  Optional. Default is `3.0`. Extra context before a candidate event.
- `TD_CASE2_STEP11_CONTEXT_AFTER_SECONDS`
  Optional. Default is `3.0`. Extra context after a candidate event.
- `TD_CASE2_STEP11_MIN_CANDIDATE_SCORE`
  Optional. Default is `0.35`. Minimum candidate score to keep.
- `TD_CASE2_STEP11_TOP_K_PREVIEW`
  Optional. Default is `50`. Number of top candidates to summarize in the Step 11 report.
- `TD_CASE2_STEP11_SAVE_FLAT`
  Optional. Default is `true`. Writes flat candidate rows for Step 11.
- `TD_CASE2_STEP11_INCLUDE_SEARCH_METADATA`
  Optional. Default is `true`. Adds Step 07 search metadata to involved objects when available.
- `TD_CASE2_STEP12_TOP_K`
  Optional. Default is `10`. Limits the final Step 12 selected event candidates.
- `TD_CASE2_STEP12_MIN_RANKING_SCORE`
  Optional. Default is `0.40`. Minimum ranking score required for selection.
- `TD_CASE2_STEP12_MIN_TEMPORAL_GAP_SECONDS`
  Optional. Default is `8.0`. Used for temporal clustering and near-duplicate suppression.
- `TD_CASE2_STEP12_MAX_PER_EVENT_TYPE`
  Optional. Default is `4`. Caps how many selected candidates can share the same event type.
- `TD_CASE2_STEP12_MAX_PER_TIME_CLUSTER`
  Optional. Default is `2`. Caps how many selected candidates can come from the same local time cluster.
- `TD_CASE2_STEP12_PREFER_TRAFFIC_SAFETY`
  Optional. Default is `true`. Gives preference to traffic-safety-style candidate events.
- `TD_CASE2_STEP12_INCLUDE_LOW_CONFIDENCE`
  Optional. Default is `true`. Allows low-confidence Step 11 candidates to remain eligible for ranking/selection.
- `TD_CASE2_STEP12_SAVE_FLAT`
  Optional. Default is `true`. Writes flat Step 12 selected-candidate output.
- `TD_CASE2_STEP12_REQUIRE_FULL_FRAME_PATH`
  Optional. Default is `true`. Requires full-scene frame evidence for final selection.
- `TD_CASE2_STEP13_MERGE_NEARBY_SELECTED`
  Optional. Default is `true`. Merges nearby Step 12 selected candidates into one VLM input group.
- `TD_CASE2_STEP13_MERGE_GAP_SECONDS`
  Optional. Default is `8.0`. Controls temporal merge distance for selected candidates.
- `TD_CASE2_STEP13_MAX_GROUP_DURATION_SECONDS`
  Optional. Default is `14.0`. Caps the merged VLM group duration.
- `TD_CASE2_STEP13_CONTEXT_BEFORE_SECONDS`
  Optional. Default is `3.0`. Extra context before the merged event center.
- `TD_CASE2_STEP13_CONTEXT_AFTER_SECONDS`
  Optional. Default is `3.0`. Extra context after the merged event center.
- `TD_CASE2_STEP13_STRIP_MODE`
  Optional. Default is `three_panel`. Allowed values are `three_panel` and `five_panel`.
- `TD_CASE2_STEP13_STRIP_WIDTH`
  Optional. Default is `1440`. Output strip width in pixels.
- `TD_CASE2_STEP13_STRIP_PANEL_HEIGHT`
  Optional. Default is `540`. Output strip panel height in pixels.
- `TD_CASE2_STEP13_ADD_LABELS`
  Optional. Default is `true`. Draws panel labels and timestamps onto strip images.
- `TD_CASE2_STEP13_SAVE_CONTACT_SHEET`
  Optional. Default is `true`. Saves one contact sheet per VLM input group.
- `TD_CASE2_STEP13_REQUIRE_FULL_FRAME_EXISTS`
  Optional. Default is `true`. Skips inputs that cannot resolve required full-scene frames.
- `TD_CASE2_STEP13_MAX_INPUTS`
  Optional. Default is `10`. Caps how many selected candidates are considered for VLM packaging.

## Files produced

Each run creates a new folder like:

`tests/td_case2/debug_runs/<video_stem>_<timestamp>/`

Inside that run folder:

- `00_stage_gate_report.json`
- `01_video_info.json`
- `02_sampled_frames/`
- `02_sampled_frames.json`
- `08_query_validation_results.json`
- `08_query_validation_matches.json`
- `08_query_validation_report.json`
- `09_search_result_cards.json`
- `09_search_result_cards_flat.json`
- `09_demo_query_result_packages.json`
- `09_search_result_card_schema.json`
- `09_search_result_packaging_report.json`
- `10_search_demo_response.json`
- `10_search_demo_results_flat.json`
- `10_search_demo_report.json`
- `10_search_demo_query_log.json`
- `11_full_scene_event_candidates.json`
- `11_full_scene_event_candidates_flat.json`
- `11_full_scene_event_candidate_report.json`
- `12_ranked_event_candidates.json`
- `12_selected_top_event_candidates.json`
- `12_event_candidate_ranking_report.json`
- `12_selected_event_candidates_flat.json`
- `13_vlm_event_inputs/`
- `13_vlm_event_inputs.json`
- `13_vlm_event_inputs_flat.json`
- `13_vlm_event_input_report.json`
- `16_evidence_video_report.json`
- `evidence_video.mp4`
- `evidence_video_index.json`

## What to check after running

- Confirm `01_video_info.json` shows readable FPS, frame count, duration, width, and height.
- Confirm `02_sampled_frames/` contains full-scene frame images named like `frame_000030.jpg`.
- Confirm `02_sampled_frames.json` lists the same frames with timestamps and relative image paths.
- Confirm `00_stage_gate_report.json` shows both steps as `success`.
- Confirm the console ends by printing the run directory path.
- For Step 08, confirm the report shows strong results for `exact_plate`, `color_class`, `invalid_ocr_blocking`, and `path_availability`.
- For Step 08, confirm returned match paths point to real best crop and full-frame files from earlier stages.
- For Step 09, confirm card titles, subtitles, and descriptions look clean for verified plate, weak OCR, and color/class-only cases.
- For Step 09, confirm media paths stay run-relative and the packaging report shows expected crop/full-frame availability counts.
- For Step 10, confirm exact plate, color/class, combined, weak OCR, and blocked invalid OCR queries produce sensible API-style responses.
- For Step 10, confirm `10_search_demo_report.json` shows correct query counts and path availability for returned cards.
- For Step 11, confirm event candidate wording stays at the candidate/possible level and does not imply confirmed truth.
- For Step 11, confirm representative frames and `full_frame_paths` point to full-scene images rather than crop images.
- For Step 12, confirm only a small Top-K set is selected and that the chosen candidates keep `needs_vlm_review` / candidate-only wording.
- For Step 12, confirm temporal suppression and event-type diversity look reasonable in `12_event_candidate_ranking_report.json`.
- For Step 13, confirm merged groups make sense for nearby selected events and that only full-scene frames are used in strips/contact sheets.
- For Step 13, confirm each VLM input package remains candidate-only and includes prompt context without calling any VLM yet.
- For Step 16, confirm `evidence_video.mp4` now includes three sections in one file: searchable event clips, a deduplicated gallery of unique full-scene object-detection frames, and the Step 13 VLM input media.
- For Step 16, confirm `evidence_video_index.json` maps each clip back to the original video time and indicates whether the clip came from an event, the object-frame gallery, or the VLM-input gallery.
