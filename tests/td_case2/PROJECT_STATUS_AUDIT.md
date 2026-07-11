# TD Case2 Project Status Audit

## 1. Git Status

- Repo path audited: `C:\Mukul K\vinfo1\video-search-engine`
- Branch: `main`
- Upstream tracking: `origin/main`
- Ahead/behind status at audit start: `0 ahead / 0 behind`
- Modified files at audit start: none
- Untracked files at audit start: none
- Safety to continue coding: yes, the repo was clean and synced before this audit file was created

Notes:

- I did not reset, pull, checkout, commit, or delete anything.
- The only file created by this audit is `tests/td_case2/PROJECT_STATUS_AUDIT.md`.

## 2. Implemented Pipeline Steps

### Runner files found

- `run_td_case2_step01_02.py`
- `run_td_case2_step01_02_02a.py`
- `run_td_case2_step03_yolo.py`
- `run_td_case2_step04a_florence_audit.py`
- `run_td_case2_step04b_tracking.py`
- `run_td_case2_step05_best_frames.py`
- `run_td_case2_step06_ocr_color.py`
- `run_td_case2_step07_search_index.py`
- `run_td_case2_step08_query_validation.py`
- `run_td_case2_step09_result_packaging.py`
- `run_td_case2_step10_search_demo.py`
- `run_td_case2_step11_event_candidates.py`
- `run_td_case2_step12_event_ranking.py`
- `run_td_case2_step13_vlm_inputs.py`

### Step/helper/module files found

- `step_02a_motion_adaptive_sampling.py`
- `step_03a_yolo_model_audit.py`
- `step_03b_yolo_detection.py`
- `step_04a_florence_model_audit.py`
- `step_04b_tracking.py`
- `step_05_best_track_frame_selector.py`
- `step_06_ocr_color_enrichment.py`
- `step_07_search_index_enrichment.py`
- `step_08_query_search_validation.py`
- `step_09_search_result_packaging.py`
- `step_10_search_demo_runner.py`
- `step_11_full_scene_event_candidates.py`
- `step_12_event_candidate_ranking.py`
- `step_13_vlm_input_generation.py`

### Config / utilities / docs used by td_case2

- `config.py`
- `stage_checks.py`
- `README.md`

### What appears implemented

- Step 01: video info
- Step 02: base frame sampling
- Step 02A: adaptive/motion-based frame filtering
- Step 03A: YOLO model audit
- Step 03B: YOLO detections
- Step 04A: Florence audit
- Step 04B: tracking
- Step 05: best track frames / crops / full frames / contact sheets
- Step 06: OCR and color enrichment
- Step 07: search index enrichment
- Step 08: query validation
- Step 09: search result packaging
- Step 10: search demo runner
- Step 11: full-scene event candidate generation
- Step 12: event candidate ranking / Top-K selection
- Step 13: VLM input generation

### What appears missing or intentionally not implemented

- Step 11.5 lightweight VLM filter: not implemented yet, as expected
- Step 14 VLM review/inference stage: not present in `tests/td_case2`

### Important structure note

- There are no separate `step_01_*.py` or `step_02_*.py` helper modules inside `tests/td_case2`.
- Early pipeline logic is mainly inside the Step 01/02 runners, while later steps have dedicated helper modules.

## 3. Latest Debug Run Found

- Latest run directory:
  `tests/td_case2/debug_runs/anpr_test_5min_20260707_122758`
- Last write time observed:
  `2026-07-07 18:32:26`
- Stage gate overall status:
  `success`

## 4. Available Output Artifacts

### Key JSON outputs present in latest run

- `00_stage_gate_report.json`
- `01_video_info.json`
- `02_sampled_frames.json`
- `02A_adaptive_filter_report.json`
- `02A_adaptive_frames.json`
- `03A_yolo_model_audit.json`
- `03_yolo_detection_report.json`
- `03_yolo_detections.json`
- `04A_florence_audit_results.json`
- `04A_florence_model_audit.json`
- `04B_detection_track_assignments.json`
- `04B_tracking_quality_report.json`
- `04B_tracking_report.json`
- `04B_tracks.json`
- `05_best_track_frames.json`
- `05_best_track_frames_report.json`
- `06_ocr_color_report.json`
- `06_ocr_color_report_cleaned.json`
- `06_ocr_color_report_verified.json`
- `06_ocr_color_results.json`
- `06_ocr_color_results_cleaned.json`
- `06_ocr_color_results_verified.json`
- `07_vehicle_search_index.json`
- `07_vehicle_search_index_flat.json`
- `07_vehicle_search_index_report.json`
- `08_query_validation_matches.json`
- `08_query_validation_report.json`
- `08_query_validation_results.json`
- `09_demo_query_result_packages.json`
- `09_search_result_card_schema.json`
- `09_search_result_cards.json`
- `09_search_result_cards_flat.json`
- `09_search_result_packaging_report.json`
- `10_search_demo_query_log.json`
- `10_search_demo_report.json`
- `10_search_demo_response.json`
- `10_search_demo_results_flat.json`
- `11_full_scene_event_candidate_report.json`
- `11_full_scene_event_candidates.json`
- `11_full_scene_event_candidates_flat.json`
- `12_event_candidate_ranking_report.json`
- `12_ranked_event_candidates.json`
- `12_selected_event_candidates_flat.json`
- `12_selected_top_event_candidates.json`
- `13_vlm_event_input_report.json`
- `13_vlm_event_inputs.json`
- `13_vlm_event_inputs_flat.json`

### Output folders present in latest run

- `02_sampled_frames`
- `02A_adaptive_preview_frames`
- `03_yolo_annotated_frames`
- `03_yolo_object_crops`
- `03A_yolo_audit_annotated_frames`
- `04A_florence_audit_inputs`
- `04A_plate_audit_crops`
- `04B_tracking_preview_frames`
- `05_selected_track_crops`
- `05_selected_track_full_frames`
- `05_track_contact_sheets`
- `06_debug_images`
- `06_plate_crops`
- `13_vlm_event_inputs`

## 5. Step-by-Step Health Summary

### Video metadata

- Duration: `377.867` seconds
- FPS: `30.0`
- Frame count: `11336`
- Resolution: `1280 x 720`

### Sampling / adaptive selection

- Base sampled frames: `1890`
- Step 02A frames selected for YOLO: `816`
- Step 02A skipped frames: `1074`
- Step 02A selection ratio: `0.431746`

Health:

- Step 01, Step 02, and Step 02A are working.
- Adaptive filtering is reducing the scan set by about 57%, which is useful for the isolated pipeline.

### YOLO detections

- Frames processed: `816`
- Frames with detections: `656`
- Total detections: `1120`
- Main class counts:
  `car 310`, `person 236`, `motorcycle 191`, `truck 177`, `bus 79`

Health:

- Step 03B is working and producing usable detections.
- There is some class noise in the output because many unrelated COCO classes are also present.

### Tracking

- Detections considered for tracking: `926`
- Tracks created: `379`
- Track type counts:
  `vehicle 273`, `person 106`
- Track quality counts:
  `good 27`, `fragmented 150`, `single_frame 201`, `weak 1`
- Usable vehicle tracks for OCR/color: `22`

Health:

- Step 04B is working.
- The biggest weakness in the current pipeline is track quality. Most tracks are fragmented or single-frame.

### Best track frames / crops

- Vehicle track count total: `273`
- Primary vehicle tracks: `22`
- Fallback vehicle tracks: `251`
- Selected track count: `273`
- Total selected detections: `304`
- Selected crops saved: `304`
- Selected full frames saved: `304`
- Contact sheets saved: `273`

Health:

- Step 05 is working.
- It preserves full-frame evidence correctly, which later steps depend on.

### OCR and color enrichment

- Processed crops: `304`
- Successful crops: `304`
- Plate crops found: `188`
- Tracks with raw plate text: `272`
- Tracks with valid plate text: `125`
- Tracks with verified license plate: `66`
- Verified unique license plates: `54`
- Tracks with vehicle color: `226`

Health:

- Step 06 is working.
- OCR/color output is substantial and feeds Step 07 correctly.
- There is still a large pool of invalid or weak OCR, so query/search logic must remain defensive.

### Search index enrichment

- Total vehicle records: `273`
- Searchable records: `273`
- Primary records: `22`
- Fallback records: `251`
- Records with verified plate: `66`
- Records with color: `226`
- Records with full frame: `273`
- Records missing full frame: `0`

Health:

- Step 07 is working.
- Full-frame coverage is strong, which is helpful for later event/VLM stages.

### Query validation

- Total tests: `22`
- Passed tests: `22`
- Pass rate: `100%`

Health:

- Step 08 is working cleanly in this run.

### Search result packaging

- Total query packages: `22`
- Total cards created: `100`
- Cards with verified plate: `59`
- Cards with full frame: `100`
- Cards with crop: `100`

Health:

- Step 09 is working.

### Search demo runner

- Queries run: `1`
- Total cards returned: `0`
- Queries with results: `0`
- Queries blocked invalid OCR: `1`

Health:

- Step 10 ran successfully, but this latest run only proves the invalid-OCR blocking path.
- It does not prove a positive-result demo query in this specific latest run.

### Step 11 event candidate generation

- Windows created: `377`
- Frames used: `816`
- Tracks used: `379`
- YOLO detections loaded: `1120`
- Raw triggers created: `515`
- Candidate events created: `187`
- Confidence mix:
  `0 high`, `45 medium`, `142 low`
- Event type counts:
  `possible_collision_or_near_miss 101`
  `sudden_stop 4`
  `stationary_vehicle 0`
  `traffic_congestion_or_dense_vehicle_activity 0`
  `vehicle_person_interaction 0`
  `unusual_motion_spike 36`
  `object_density_spike 4`
  `track_start_stop_activity 42`

Health:

- Step 11 is definitely producing candidates.
- It is producing a large number of low-confidence candidates.
- The output is heavily dominated by `possible_collision_or_near_miss`, which is the main false-positive risk area.

### Step 12 event ranking / Top-K selection

- Input candidate count: `187`
- Ranked candidate count: `187`
- Selected Top-K count: `10`
- Temporal clusters: `7`
- Selected event type counts:
  `possible_collision_or_near_miss 4`
  `sudden_stop 2`
  `unusual_motion_spike 4`
- Suppression summary:
  `100` suppressed by temporal cluster
  `21` suppressed by event type cap
  `0` below min ranking score

Health:

- Step 12 is working.
- It is acting as a useful compression layer on top of noisy Step 11 output.

### Step 13 VLM input generation

- Selected candidates loaded: `10`
- Merged groups created: `6`
- VLM inputs created: `6`
- Temporal strips created: `6`
- Contact sheets created: `6`
- Inputs skipped: `0`
- Inputs ready for VLM: `6`

Health:

- Step 13 is working.
- It is producing full-scene event review packages, not crop-only packages.

## 6. Step 11 Current Logic Summary

### Files Step 11 reads

Required:

- `01_video_info.json`
- one accepted Step 02A file such as `02A_adaptive_frames.json`
- `03_yolo_detections.json`
- `04B_tracks.json`

Optional enrichment:

- `04B_tracking_report.json`
- `05_best_track_frames.json`
- `07_vehicle_search_index.json`

### How Step 11 creates windows

- It builds sliding scene windows across the full video using:
  - `window_seconds` default `2.0`
  - `window_stride_seconds` default `1.0`
- For each window it aggregates:
  - motion score
  - motion pixel ratio
  - histogram change
  - object count
  - vehicle count
  - person count
  - active track ids
- It also picks a representative full-scene frame nearest the window center.

### Event types currently supported

- `possible_collision_or_near_miss`
- `sudden_stop`
- `stationary_vehicle`
- `traffic_congestion_or_dense_vehicle_activity`
- `vehicle_person_interaction`
- `unusual_motion_spike`
- `object_density_spike`
- `track_start_stop_activity`

### Main scoring behavior

- Step 11 first builds a `score_base` from generic scene evidence:
  - motion spike
  - high motion pixels
  - high histogram change
  - object density
  - vehicle density
- It then adds event-specific score bumps.
- Example:
  - close vehicle pair starts at `0.25`
  - extra motion evidence can add `0.15`
  - sudden stop adds `0.20`
  - bbox overlap adds `0.10`
- A raw trigger is kept when it reaches `min_candidate_score`, default `0.35`.
- Compatible nearby triggers are merged into one candidate event.
- Merging can add up to `0.10` more score.

### How final candidate events are formed

- Raw triggers are merged when they are:
  - close in time
  - in compatible event groups
  - or share involved tracks
- Final candidate event fields include:
  - timestamps
  - candidate score
  - confidence/severity labels
  - trigger reasons
  - involved track ids/classes
  - optional search metadata
  - representative full-scene frame
  - `full_frame_paths`
  - `needs_vlm_review = true`
  - `final_event_truth = unknown_candidate_only`

### Output files Step 11 writes

- `11_full_scene_event_candidates.json`
- `11_full_scene_event_candidates_flat.json`
- `11_full_scene_event_candidate_report.json`

### Config / environment variables controlling Step 11

- `TD_CASE2_RUN_DIR`
- `TD_CASE2_STEP11_WINDOW_SECONDS`
- `TD_CASE2_STEP11_WINDOW_STRIDE_SECONDS`
- `TD_CASE2_STEP11_MERGE_GAP_SECONDS`
- `TD_CASE2_STEP11_MAX_EVENT_SECONDS`
- `TD_CASE2_STEP11_CONTEXT_BEFORE_SECONDS`
- `TD_CASE2_STEP11_CONTEXT_AFTER_SECONDS`
- `TD_CASE2_STEP11_MIN_CANDIDATE_SCORE`
- `TD_CASE2_STEP11_TOP_K_PREVIEW`
- `TD_CASE2_STEP11_SAVE_FLAT`
- `TD_CASE2_STEP11_INCLUDE_SEARCH_METADATA`

Defaults from `config.py`:

- window seconds: `2.0`
- stride seconds: `1.0`
- merge gap seconds: `3.0`
- max event seconds: `12.0`
- context before: `3.0`
- context after: `3.0`
- min candidate score: `0.35`
- top preview: `50`
- save flat: `true`
- include search metadata: `true`

### Where false positives can come from

1. Generic motion and density evidence is enough to create candidates even when nothing important is happening.
2. `track_start_stop_activity` can trigger from crowded normal activity, not just meaningful events.
3. `possible_collision_or_near_miss` uses proximity plus optional overlap plus optional motion boosts. In dense traffic this can overfire.
4. Step 11 uses all Step 04B tracks, including fragmented and single-frame tracks. That increases noisy interactions.
5. Merge logic can boost candidate score when several weak triggers happen near each other.
6. The minimum score threshold is low enough that many weak windows survive.

## 7. Step 12 / Step 13 Compatibility

### Step 12 inputs and coupling

- Step 12 directly reads:
  `11_full_scene_event_candidates.json`
- It expects candidate fields such as:
  - `candidate_event_id`
  - `event_type`
  - `candidate_score`
  - `representative_frame`
  - `full_frame_paths`
  - `needs_vlm_review`
  - `final_event_truth`

### Step 12 filename dependency

- Yes, Step 12 currently depends on the exact filename `11_full_scene_event_candidates.json`.
- It also writes `source_file` metadata with that exact name.

### Step 12 compatibility with future Step 11.5

- Easy to modify later if Step 11.5 writes the same schema.
- Main updates would be:
  - change the input filename in the Step 12 runner / loader
  - update `source_file` metadata strings
- If Step 11.5 preserves candidate ids and key fields, the change is small.

### Step 13 inputs and coupling

- Step 13 directly reads:
  - `12_selected_top_event_candidates.json`
  - `12_event_candidate_ranking_report.json`
  - `12_ranked_event_candidates.json` if present
  - `11_full_scene_event_candidates.json`
  - `01_video_info.json`
  - `02_sampled_frames.json`
- Step 13 uses Step 11 again to enrich selected Step 12 candidates before packaging.

### Step 13 filename dependency

- Yes, Step 13 currently depends on:
  - `12_selected_top_event_candidates.json`
  - `11_full_scene_event_candidates.json`
  - `02_sampled_frames.json`

### Step 13 compatibility with future Step 11.5

- Mostly easy if Step 12 already switches to the filtered Step 11.5 candidate file and keeps the same candidate schema.
- One extra detail:
  Step 13 still enriches from Step 11. If Step 11.5 becomes the real source of truth, Step 13 should later enrich from the filtered candidate file too, not from the original noisy Step 11 file.

### Does Step 13 use full-scene frames or crops?

- It uses full-scene frames.
- Evidence:
  - Step 12 selected payload contains `representative_frame_path` and `full_frame_paths` like `02_sampled_frames/frame_001110.jpg`
  - Step 13 loads `02_sampled_frames.json`
  - Step 13 resolves frame paths against sampled full-scene images
  - Step 13 writes temporal strips and contact sheets from those full-scene frames

Conclusion:

- Step 13 is giving full-scene VLM inputs, not cropped object images.

## 8. Missing or Broken Pieces

1. Step 11.5 filter is not implemented yet.
2. Step 14 VLM review/inference is not implemented in this isolated folder.
3. Step 10 latest run does not demonstrate a positive search-return case, only an invalid-OCR blocked case.
4. Track quality is weak overall, with many fragmented and single-frame tracks.
5. Step 11 candidate volume is high and mostly low-confidence.
6. Some supported Step 11 event types are not appearing at all in this run:
   `stationary_vehicle`, `traffic_congestion_or_dense_vehicle_activity`, `vehicle_person_interaction`
7. Step 12 and Step 13 are tightly bound to exact input filenames, so later Step 11.5 integration should be done carefully.

## 9. Risks / False Positive Sources

1. Dense traffic can look like collision risk because Step 11 uses proximity and overlap heuristics.
2. Motion spikes and histogram changes can reflect normal road flow, camera shake, or scene clutter.
3. Fragmented tracking can create fake short interactions between unrelated objects.
4. `track_start_stop_activity` is very generic and can label normal busy windows as meaningful.
5. Step 11 currently outputs many low-confidence candidates, so raw candidate count alone is not trustworthy.
6. Because Step 12 still ranks some low-confidence Step 11 candidates, weak Step 11 logic can still leak into final VLM inputs.

## 10. Recommended Next Actions Before Step 11.5

1. Do not build Step 11.5 yet.
2. First tighten and instrument Step 11 heuristics so we understand why false positives are being created.
3. Focus on reducing false positives in `possible_collision_or_near_miss` and `track_start_stop_activity`.
4. Add richer Step 11 debug/report fields before adding a filter layer:
   - counts by trigger reason
   - counts by raw trigger type before merge
   - counts by track quality involved in selected candidates
   - collision candidates with center-distance and IoU summaries
5. Consider gating Step 11 interaction events by track quality, so fragmented or single-frame tracks contribute less.
6. Review whether Step 11 should ignore some generic windows unless there is stronger motion plus stronger track evidence.
7. After Step 11 becomes less noisy, then add Step 11.5 as a lightweight filtered stage between Step 11 and Step 12.

## 11. Exact Commands I Should Run Next

These are safe next commands for this isolated pipeline.

### Re-check repo state

```powershell
cd "C:\Mukul K\vinfo1\video-search-engine"
git status --branch --short
```

### Reuse the latest isolated run

```powershell
$env:TD_CASE2_RUN_DIR="C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"
```

### Re-run lightweight event/ranking/VLM packaging stages after future code edits

```powershell
python tests\td_case2\run_td_case2_step11_event_candidates.py
python tests\td_case2\run_td_case2_step12_event_ranking.py
python tests\td_case2\run_td_case2_step13_vlm_inputs.py
```

### Inspect the current latest outputs quickly

```powershell
Get-Content .\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758\11_full_scene_event_candidate_report.json
Get-Content .\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758\12_event_candidate_ranking_report.json
Get-Content .\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758\13_vlm_event_input_report.json
```

## Bottom Line

- The isolated `td_case2` pipeline is implemented through Step 13.
- The latest run completed successfully through Step 13.
- The main current problem is not missing code. It is Step 11 over-triggering candidate events from normal activity.
- The best next coding task is:
  tighten Step 11 heuristics and add better Step 11 diagnostics before implementing Step 11.5.
