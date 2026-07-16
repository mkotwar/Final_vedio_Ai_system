# TD Case 2 Stage Input / Output Map

## Table 1: Complete Stage Input/Output Map

| Step | Stage | Receives | Processing | Produces | Consumed by | Runner | Logic file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Video information | input video path, output root, sample interval | OpenCV metadata read | `01_video_info.json` | 02, 02A, 03, 11, 13, 14, 16 | `run_td_case2_step01_02.py` or `run_td_case2_step01_02_02a.py` | same runner helpers |
| 02 | Fixed frame sampling | `01_video_info.json`, source video | OpenCV decode every N seconds | `02_sampled_frames.json`, `02_sampled_frames/` | 02A, 03, 13, 16 fallback frame catalog | `run_td_case2_step01_02.py` or `run_td_case2_step01_02_02a.py` | same runner helpers |
| 02A | Adaptive frame sampling | `01_video_info.json`, `02_sampled_frames.json`, sampled JPEGs | motion score, blob ratio, histogram change, quality filters | `02A_adaptive_frames.json`, `02A_adaptive_filter_report.json`, preview dir | 03, 11, 16 preferred frame catalog | `run_td_case2_step01_02_02a.py` | `step_02a_motion_adaptive_sampling.py` |
| 03A | YOLO model audit | `02A_adaptive_frames.json`, YOLO model paths | sample-frame model load and smoke inference | `03A_yolo_model_audit.json` | audit only | `run_td_case2_step03_yolo.py` | `step_03a_yolo_model_audit.py` |
| 03B | YOLO detection | `01_video_info.json`, `02A_adaptive_frames.json`, `02_sampled_frames/`, YOLO config | YOLO inference on selected adaptive frames | `03_yolo_detections.json`, `03_yolo_detection_report.json`, crops, annotated frames | 04A, 04B, 07B, 11 | `run_td_case2_step03_yolo.py` | `step_03b_yolo_detection.py` |
| 04A | Florence model audit | `03_yolo_detections.json`, Florence model path, optional plate detector | audit-only OCR/caption checks on selected crops | `04A_florence_model_audit.json`, `04A_florence_audit_results.json` | audit only | `run_td_case2_step04a_florence_audit.py` | `step_04a_florence_model_audit.py` |
| 04B | Tracking | `01_video_info.json`, `03_yolo_detections.json`, tracking config | custom deterministic association over YOLO detections | `04B_tracks.json`, `04B_detection_track_assignments.json`, two reports | 05, 07B, 11, 16 | `run_td_case2_step04b_tracking.py` | `step_04b_tracking.py` |
| 05 | Best track-frame selection | `04B_tracks.json`, `04B_tracking_report.json`, crop/full-frame paths in track detections | rank primary and fallback evidence frames per track | `05_best_track_frames.json`, report, selected crop dir, full-frame dir, contact sheets | 06, 07B optional, 11 optional | `run_td_case2_step05_best_frames.py` | `step_05_best_track_frame_selector.py` |
| 06 | OCR and colour enrichment | `05_best_track_frames.json`, `05_best_track_frames_report.json`, Florence model path, optional plate detector | Florence OCR/caption, plate YOLO, OCR cleaning, color extraction | `06_ocr_color_results.json`, verified outputs, reports, plate/debug dirs | 07 legacy, 07B | `run_td_case2_step06_ocr_color.py` | `step_06_ocr_color_enrichment.py` |
| 07B | Traffic/object search index | `03_yolo_detections.json`, `04B_tracks.json`, `05_best_track_frames.json`, `01_video_info.json`, optional Step 06 outputs | merge track evidence with OCR/color; add detection fallbacks | `07B_traffic_object_search_index.json`, flat file, report | 08B, 09B, 10B, 16 | `run_td_case2_step07b_traffic_object_search_index.py` | `traffic_search_common.py` |
| 08B | Dynamic search validation | `07B` payload + flat + report | deterministic query tests over active search index | `08B_dynamic_search_validation_results.json`, matches, report | 09B, UI | `run_td_case2_step08b_dynamic_search_validation.py` | runner-local logic |
| 09B | Search-result cards | `07B` payload + report, `08B` results | package universal UI cards and schema | `09B_universal_search_cards.json`, flat file, schema, report | UI | `run_td_case2_step09b_universal_search_cards.py` | runner-local logic |
| 10B | Universal search demo | `07B` payload + report | build demo response directly from active search index | `10B_universal_search_demo_response.json`, report | UI/demo terminal output | `run_td_case2_step10b_universal_search_demo.py` | runner-local logic |
| 11 | Full-scene event candidates | `01_video_info.json`, `02A_adaptive_frames.json`, `03_yolo_detections.json`, `04B_tracks.json`, optional `04B_tracking_report.json`, optional `05_best_track_frames.json`, optional legacy `07_vehicle_search_index.json` | rule-based temporal event proposal from full-scene detections/tracks | `11_full_scene_event_candidates.json`, flat file, report, diagnostics | 11.5, 12 | `run_td_case2_step11_event_candidates.py` | `step_11_full_scene_event_candidates.py` |
| 11.5 | Lightweight VLM filter | `11_full_scene_event_candidates.json`, `01_video_info.json`, backend config | one-image Qwen filter or deterministic fallback | `11_5_vlm_filtered_event_candidates.json`, flat file, report, cache dir | 12 | `run_td_case2_step11_5_vlm_filter.py` | `step_11_5_lightweight_vlm_filter.py` |
| 12 | Event-candidate ranking | `11_full_scene_event_candidate_report.json`, plus Step 11.5 filtered file when valid or else raw Step 11 files | deterministic ranking, clustering, top-k selection | `12_ranked_event_candidates.json`, `12_selected_top_event_candidates.json`, flat file, report | 13, 14 optional summary context, 16 fallback scene events | `run_td_case2_step12_event_ranking.py` | `step_12_event_candidate_ranking.py` |
| 13 | VLM input generation | Step 12 selected/ranked outputs, `01_video_info.json`, `02_sampled_frames/`, Step 11 or Step 11.5 candidate file | build temporal strips, contact sheets, primary-frame packages | `13_vlm_event_inputs.json`, flat file, report, `13_vlm_event_inputs/` | 14, 16 VLM gallery | `run_td_case2_step13_vlm_inputs.py` | `step_13_vlm_input_generation.py` |
| 14 | Main VLM review | `13_vlm_event_inputs.json`, `13_vlm_event_input_report.json`, `12_selected_top_event_candidates.json`, `01_video_info.json`, backend config | Qwen local/API review of Step 13 media; normalize JSON decision | `14_vlm_event_reviews.json`, flat file, `14_final_video_summary.json`, report | 15, 16 fallback scene-event synthesis, UI | `run_td_case2_step14_vlm_review.py` | `step_14_vlm_event_review.py` |
| 15 | Searchable reviewed events | Step 11, Step 12, Step 14 review + summary files | persist only visible reviewed scene events as searchable records | `15_searchable_events.json`, flat file, report | 16 | `run_td_case2_step15_searchable_events.py` | `step_15_searchable_event_generation.py` |
| 16 | Evidence-video generation | `01_video_info.json`, `04B_tracks.json`, `07B_traffic_object_search_index.json`, `11_full_scene_event_candidates.json`, `12_selected_top_event_candidates.json`, optional Step 13/14/15 | reuse existing frames and JSON artifacts to encode final MP4 | `evidence_video.mp4`, `evidence_video_index.json`, `16_evidence_video_report.json` | UI/final output | `run_td_case2_step16_evidence_video.py` | `step_16_evidence_video_generation.py` |

## Detailed Stage Notes

### Step 01 `CONFIRMED`

- Runner: `run_td_case2_step01_02.py`
- Main symbols: `extract_video_info`, `main`
- Inputs:
  - `TD_CASE2_INPUT_VIDEO`
  - `TD_CASE2_OUTPUT_ROOT`
  - `TD_CASE2_SAMPLE_EVERY_SECONDS`
- Output fields in `01_video_info.json`:
  - `input_video_path`
  - `video_name`
  - `fps`
  - `frame_count`
  - `duration_seconds`
  - `duration_text`
  - `width`
  - `height`
- Downstream: mandatory for Steps 02, 02A, 03, 11, 13, 14, 16.
- Source: `tests/td_case2/run_td_case2_step01_02.py:215-245`

### Step 02 `CONFIRMED`

- Runner: `run_td_case2_step01_02.py`
- Output fields in `02_sampled_frames.json`:
  - `sample_every_seconds`
  - `fps`
  - `frame_count`
  - `expected_sample_count`
  - `actual_sample_count`
  - `sampled_frames_folder`
  - `sampled_frames[]` with `frame_id`, `timestamp_seconds`, `timestamp_text`, `image_path`
- Downstream: mandatory for Step 02A and Step 03; Step 13/16 use the image directory.
- Source: `tests/td_case2/run_td_case2_step01_02.py:200-211, 242-255`

### Step 02A `CONFIRMED`

- Logic: motion adaptive filtering on the Step 02 sample set only.
- Important processing:
  - ROI crop
  - frame quality checks
  - motion score
  - motion-pixel/blob ratios
  - histogram change
  - heartbeat/min-gap rules
- Output fields in `02A_adaptive_frames.json`:
  - `selected_frames[]`
  - `input_sampled_frames`
  - `selected_for_yolo`
  - `selection_ratio`
  - per-frame reasons and motion metrics
- Downstream: Step 03 and Step 11; Step 16 prefers it for frame catalog.
- Source: `tests/td_case2/step_02a_motion_adaptive_sampling.py:358-460, 696-697`

### Step 03A `OPTIONAL audit`

- Inputs:
  - `02A_adaptive_frames.json`
  - person/object/combined YOLO model paths
- Processing: loads configured YOLO models and runs audit-frame inference only.
- Output fields in `03A_yolo_model_audit.json`:
  - `models[]`
  - `overall_ready_for_detection`
  - `class_names`
  - device and memory fields
- Downstream: none.
- Source: `tests/td_case2/run_td_case2_step03_yolo.py:148-179, 194-215`; `tests/td_case2/step_03a_yolo_model_audit.py:73-125`

### Step 03B `CONFIRMED`

- Inputs:
  - `01_video_info.json`
  - `02A_adaptive_frames.json`
  - `02_sampled_frames/`
  - YOLO thresholds and device config
- Processing:
  - loads YOLO models
  - runs inference on adaptive-selected frames
  - can save object crops and annotated full frames
- Output fields in `03_yolo_detections.json`:
  - frame-level detection entries
  - `detection_id`
  - `frame_id`
  - `class_name`
  - `confidence`
  - `bbox_xyxy`
  - `image_path`
  - `crop_path`
- Downstream:
  - mandatory: 04A, 04B, 07B, 11
- Source: `tests/td_case2/run_td_case2_step03_yolo.py:139-179`; `tests/td_case2/step_03b_yolo_detection.py:86-107, 359-360`

### Step 04A `OPTIONAL audit`

- Inputs:
  - `03_yolo_detections.json`
  - Florence model path
  - optional adapter path
  - optional plate detector path
- Processing:
  - sample crop selection from Step 03 detections
  - Florence OCR/caption smoke tests
  - optional plate-detector probe
- Outputs:
  - `04A_florence_model_audit.json`
  - `04A_florence_audit_results.json`
- Downstream: none.
- Source: `tests/td_case2/run_td_case2_step04a_florence_audit.py:131-156, 239-344`; `tests/td_case2/step_04a_florence_model_audit.py:78-187, 391-655`

### Step 04B `CONFIRMED`

- Inputs:
  - `01_video_info.json`
  - `03_yolo_detections.json`
  - tracking thresholds from env
- Processing:
  - class filtering
  - IoU and center-distance association
  - area-change limits
  - time-gap constraints
  - dominant-class and best-detection tracking
- Output fields in `04B_tracks.json`:
  - `track_id`
  - `dominant_class_name`
  - `track_quality`
  - `start_timestamp_seconds`
  - `end_timestamp_seconds`
  - `duration_seconds`
  - `best_detection_id`
  - `best_crop_path`
  - `detections[]`
- Additional outputs:
  - `04B_detection_track_assignments.json`
  - `04B_tracking_report.json`
  - `04B_tracking_quality_report.json`
- Downstream: mandatory for 05, 07B, 11, 16.
- Source: `tests/td_case2/run_td_case2_step04b_tracking.py:108-137, 232-276`; `tests/td_case2/step_04b_tracking.py:329-430, 694-697`

### Step 05 `CONFIRMED`

- Inputs:
  - `04B_tracks.json`
  - `04B_tracking_report.json`
- Processing:
  - separates `primary` and `fallback` vehicle tracks
  - scores detections per track
  - copies selected crops/full frames
  - builds per-track contact sheets
- Output fields in `05_best_track_frames.json`:
  - `tracks[]`
  - `track_id`
  - `selection_group`
  - `selected_count`
  - `selected_detections[]`
  - `selected_crop_path`
  - `selected_full_frame_path`
  - `contact_sheet_path`
  - `final_selection_score`
- Downstream: Step 06 mandatory; Step 07B and Step 11 optional enrichment.
- Source: `tests/td_case2/run_td_case2_step05_best_frames.py:103-145, 230-278`; `tests/td_case2/step_05_best_track_frame_selector.py:344-548, 570-639`

### Step 06 `CONFIRMED`

- Inputs:
  - `05_best_track_frames.json`
  - `05_best_track_frames_report.json`
  - Florence model path
  - optional Florence adapter
  - optional plate-detector model path
  - process-group limits for `primary` and `fallback`
- Processing:
  - flattens Step 05 selections into a queue
  - reruns Florence OCR/caption per selected crop
  - runs license-plate detector when configured
  - cleans OCR, verifies plate format, extracts vehicle color
- Output fields:
  - `06_ocr_color_results.json`
  - `06_ocr_color_results_verified.json`
  - `06_ocr_color_report.json`
  - `06_ocr_color_report_verified.json`
- Important JSON fields:
  - `track_results[]`
  - `track_id`
  - `selection_group`
  - `best_license_plate_text`
  - `best_license_plate_valid`
  - `best_vehicle_color`
  - `vehicle_attributes`
  - `license_plate_attributes`
  - `scene_attributes`
- Downstream:
  - legacy Step 07 reads verified outputs only
  - active Step 07B prefers verified output, falls back to raw Step 06 if verified file missing
- Source: `tests/td_case2/run_td_case2_step06_ocr_color.py:169-204, 260-321`; `tests/td_case2/step_06_ocr_color_enrichment.py:1488-1577`

### Step 07B `CONFIRMED active`

- Inputs:
  - `03_yolo_detections.json`
  - `04B_tracks.json`
  - `05_best_track_frames.json`
  - `01_video_info.json`
  - optional `06_ocr_color_results_verified.json`
  - fallback `06_ocr_color_results.json`
- Processing:
  - builds track-backed searchable records
  - backfills single detections not present in tracks
  - assembles search text and normalized tokens
- Output fields in `07B_traffic_object_search_index.json`:
  - `records[]`
  - `object_record_id`
  - `source_type`
  - `track_id`
  - `detection_id`
  - `object_type`
  - `class_name`
  - `timestamp_seconds`
  - `full_frame_path`
  - `crop_path`
  - `contact_sheet_path`
  - `verified_vehicle_color`
  - `verified_license_plate`
  - `possible_plate_text`
  - `weak_ocr_text`
  - `quality`
  - `search_text`
  - `searchable_tokens`
- Downstream:
  - mandatory: 08B, 09B, 10B, 16
- Source: `tests/td_case2/traffic_search_common.py:455-728`; `tests/td_case2/run_td_case2_step07b_traffic_object_search_index.py:31-39, 90-118`

### Step 08B `CONFIRMED active`

- Inputs:
  - `07B_traffic_object_search_index.json`
  - `07B_traffic_object_search_index_flat.json`
  - `07B_traffic_object_search_index_report.json`
- Output fields:
  - `08B_dynamic_search_validation_results.json`
  - `08B_dynamic_search_validation_matches.json`
  - `08B_dynamic_search_validation_report.json`
- Downstream:
  - Step 09B requires `results`
  - UI can inspect `matches` and `report`
- Missing file behavior: runner fails fast.
- Source: `tests/td_case2/run_td_case2_step08b_dynamic_search_validation.py:31-38, 81-328`

### Step 09B `CONFIRMED active`

- Inputs:
  - `07B_traffic_object_search_index.json`
  - `07B_traffic_object_search_index_report.json`
  - `08B_dynamic_search_validation_results.json`
- Outputs:
  - `09B_universal_search_cards.json`
  - `09B_universal_search_cards_flat.json`
  - `09B_universal_search_card_schema.json`
  - `09B_universal_search_packaging_report.json`
- Downstream: UI only.
- Source: `tests/td_case2/run_td_case2_step09b_universal_search_cards.py:30-36, 70-164`

### Step 10B `CONFIRMED active`

- Inputs:
  - `07B_traffic_object_search_index.json`
  - `07B_traffic_object_search_index_report.json`
- Outputs:
  - `10B_universal_search_demo_response.json`
  - `10B_universal_search_demo_report.json`
- Downstream: UI/demo only.
- Source: `tests/td_case2/run_td_case2_step10b_universal_search_demo.py:29-30, 43-102`

### Step 11 `CONFIRMED with legacy optional enrichment`

- Inputs:
  - `01_video_info.json`
  - `02A_adaptive_frames.json` via tolerant resolver
  - `03_yolo_detections.json`
  - `04B_tracks.json`
  - optional `04B_tracking_report.json`
  - optional `05_best_track_frames.json`
  - optional legacy `07_vehicle_search_index.json`
- Processing:
  - rule-based temporal windows over full-scene detections and tracks
  - candidate scoring and merging
  - optional legacy search metadata lookup by track id
- Output fields in `11_full_scene_event_candidates.json`:
  - `candidate_events[]`
  - `candidate_event_id`
  - `event_type`
  - `candidate_score`
  - `best_timestamp_seconds`
  - `context_start_seconds`
  - `context_end_seconds`
  - `involved_track_ids`
  - `involved_classes`
  - `representative_frame.image_path`
  - `full_frame_paths`
- Downstream:
  - Step 11.5 mandatory in VLM pipeline
  - Step 12 fallback source
- Source: `tests/td_case2/step_11_full_scene_event_candidates.py:1156-1175, 1313-1319`

### Step 11.5 `CONFIRMED active`

- Inputs:
  - `11_full_scene_event_candidates.json`
  - `01_video_info.json`
  - `TD_CASE2_VLM_BACKEND`
  - model path or API provider/model
- Processing:
  - resolves one full-scene image per candidate
  - local Qwen image call, API Qwen image call, or deterministic fallback when disabled
  - can backfill normal-context candidates to meet minimum output count
- Output fields in `11_5_vlm_filtered_event_candidates.json`:
  - `candidate_events[]`
  - `vlm_filter.decision`
  - `vlm_filter.short_reason`
  - `vlm_filter.image_path_used`
  - `representative_frame_path`
- Downstream: Step 12.
- Source: `tests/td_case2/run_td_case2_step11_5_vlm_filter.py:71-89, 184-212`; `tests/td_case2/step_11_5_lightweight_vlm_filter.py:106-140, 349-365, 385-440, 452-728`

### Step 12 `CONFIRMED`

- Inputs:
  - mandatory `11_full_scene_event_candidate_report.json`
  - preferred `11_5_vlm_filtered_event_candidates.json`
  - fallback `11_full_scene_event_candidates.json` and flat file
- Processing:
  - chooses Step 11.5 payload when it exists and contains non-empty `candidate_events`
  - otherwise falls back to Step 11
  - ranking, clustering, per-type cap, per-cluster cap, forced accident preservation
- Outputs:
  - `12_ranked_event_candidates.json`
  - `12_selected_top_event_candidates.json`
  - `12_selected_event_candidates_flat.json`
  - `12_event_candidate_ranking_report.json`
- Downstream:
  - Step 13 mandatory
  - Step 16 can reconstruct scene events from Step 12+14 when Step 15 missing
- Source: `tests/td_case2/run_td_case2_step12_event_ranking.py:95-105`; `tests/td_case2/step_12_event_candidate_ranking.py:364-375, 426-533, 543-621`

### Step 13 `CONFIRMED`

- Inputs:
  - `12_selected_top_event_candidates.json`
  - `12_selected_event_candidates_flat.json`
  - `12_event_candidate_ranking_report.json`
  - `01_video_info.json`
  - `02_sampled_frames/`
  - Step 11 or Step 11.5 candidate source file
- Processing:
  - enriches selected candidates with Step 11 context
  - optionally merges nearby selected candidates
  - builds temporal strip, optional contact sheet, primary-frame reference
- Output fields in `13_vlm_event_inputs.json`:
  - `vlm_inputs[]`
  - `vlm_input_id`
  - `source_candidate_ids`
  - `source_event_types`
  - `best_timestamp_seconds`
  - `media.temporal_strip_path`
  - `media.contact_sheet_path`
  - `media.primary_frame_path`
- Downstream:
  - Step 14 mandatory
  - Step 16 gallery optional
- Source: `tests/td_case2/run_td_case2_step13_vlm_inputs.py:101-112`; `tests/td_case2/step_13_vlm_input_generation.py:478-505, 560-657, 698-734`

### Step 14 `CONFIRMED`

- Inputs:
  - `13_vlm_event_inputs.json`
  - `13_vlm_event_input_report.json`
  - `12_selected_top_event_candidates.json`
  - `01_video_info.json`
  - backend mode `local_qwen`, `api_qwen`, or `disabled`
- Processing:
  - prefers contact sheet when present, otherwise temporal strip/primary frame
  - Qwen review
  - response normalization and consistency correction
- Output fields:
  - `14_vlm_event_reviews.json`
  - `14_vlm_event_reviews_flat.json`
  - `14_final_video_summary.json`
  - `14_vlm_event_review_report.json`
- Important review fields:
  - `vlm_input_id`
  - `source_candidate_ids`
  - `model_review.review_decision`
  - `model_review.event_visible`
  - `model_review.event_type`
  - `model_review.risk_level`
  - `model_review.summary_caption`
  - `model_review.needs_human_review`
  - `model_review.confidence`
- Downstream:
  - Step 15 mandatory
  - Step 16 optional fallback scene-event synthesis
- Source: `tests/td_case2/run_td_case2_step14_vlm_review.py:75-83, 175-232`; `tests/td_case2/step_14_vlm_event_review.py:141-150, 224-291, 513-608, 691, 771-875`

### Step 15 `CONFIRMED`

- Inputs:
  - `11_full_scene_event_candidates.json`
  - `12_selected_top_event_candidates.json`
  - `14_vlm_event_reviews.json`
  - `14_final_video_summary.json`
- Processing:
  - keeps only visible reviewed moments
  - normalizes event type aliases
  - assigns priority rank
- Output fields in `15_searchable_events.json`:
  - `records[]`
  - `searchable_event_id`
  - `source_candidate_ids`
  - `source_vlm_input_id`
  - `event_type`
  - `title`
  - `summary`
  - `start_seconds`
  - `end_seconds`
  - `best_timestamp_seconds`
  - `confidence`
  - `risk_level`
  - `critical_event`
  - `track_ids`
  - `class_names`
  - `representative_frame_path`
  - `review_decision`
- Downstream: Step 16.
- Source: `tests/td_case2/run_td_case2_step15_searchable_events.py:26-57`; `tests/td_case2/step_15_searchable_event_generation.py:102-188`

### Step 16 `CONFIRMED`

- Inputs:
  - mandatory `01_video_info.json`
  - mandatory `04B_tracks.json`
  - mandatory `07B_traffic_object_search_index.json`
  - mandatory `11_full_scene_event_candidates.json`
  - mandatory `12_selected_top_event_candidates.json`
  - optional `15_searchable_events.json`
  - optional `14_vlm_event_reviews.json`
  - optional `13_vlm_event_inputs.json`
  - frame catalog from `02A_adaptive_frames.json` or fallback `02_sampled_frames.json`
- Processing:
  - prefers Step 15 scene events
  - falls back to Step 12 + Step 14 derived scene events
  - builds searchable object events from `07B`
  - deduplicates unique full-scene object gallery frames
  - appends VLM-input gallery
  - writes final MP4 and index
- Outputs:
  - `evidence_video.mp4`
  - `evidence_video_index.json`
  - `16_evidence_video_report.json`
- Downstream: UI/final output only.
- Source: `tests/td_case2/run_td_case2_step16_evidence_video.py:82-105, 128-152`; `tests/td_case2/step_16_evidence_video_generation.py:17-23, 80-88, 164-178, 507-528, 974-1370`

## Table 2: File Data Lineage

| File or directory | Created by | Important fields/content | Consumed by | Required/optional |
| --- | --- | --- | --- | --- |
| `01_video_info.json` | Step 01 | video path, fps, width, height, duration | 02, 02A, 03, 11, 13, 14, 16 | Required |
| `02_sampled_frames.json` | Step 02 | `sampled_frames[]`, sample interval | 02A, 03 | Required |
| `02_sampled_frames/` | Step 02 | JPEG frames | 02A, 03, 13, 16 fallback | Required |
| `02A_adaptive_frames.json` | Step 02A | `selected_frames[]`, keep reasons, motion metrics | 03, 11, 16 preferred | Required for active path |
| `03A_yolo_model_audit.json` | Step 03A | audit model status | nobody | Optional terminal audit |
| `03_yolo_detections.json` | Step 03B | frame detections, crop/full-frame paths | 04A, 04B, 07B, 11 | Required |
| `03_yolo_object_crops/` | Step 03B | cropped objects | 04A, 05 indirectly | Optional but usually produced |
| `04A_florence_model_audit.json` | Step 04A | audit status | nobody | Optional terminal audit |
| `04B_tracks.json` | Step 04B | track ids, detections, best detection, timing | 05, 07B, 11, 16 | Required |
| `04B_tracking_report.json` | Step 04B | counts and thresholds | 05, 11 optional | Required for 05 |
| `05_best_track_frames.json` | Step 05 | selected detections, groups, contact-sheet path | 06, 07B optional, 11 optional | Required for 06 |
| `05_selected_track_crops/` | Step 05 | chosen crops | 06 | Required for practical Step 06 |
| `06_ocr_color_results_verified.json` | Step 06 | verified OCR/color records | 07, 07B preferred | Optional fallback chain |
| `06_ocr_color_results.json` | Step 06 | raw OCR/color records | 07B fallback | Optional |
| `07_vehicle_search_index.json` | Legacy Step 07 | legacy vehicle search records | Step 11 optional enrichment | `LEGACY` |
| `07B_traffic_object_search_index.json` | Step 07B | active traffic/object records | 08B, 09B, 10B, 16 | Required |
| `08B_dynamic_search_validation_results.json` | Step 08B | dynamic validation results | 09B | Required for 09B |
| `09B_universal_search_cards.json` | Step 09B | UI cards | UI | Terminal artifact |
| `10B_universal_search_demo_response.json` | Step 10B | demo results | UI/demo | Terminal artifact |
| `11_full_scene_event_candidates.json` | Step 11 | `candidate_events[]` | 11.5, 12 fallback, 15, 16 | Required |
| `11_5_vlm_filtered_event_candidates.json` | Step 11.5 | filtered `candidate_events[]` | 12 preferred, 13 source selection via Step 12 source | Optional preferred |
| `12_selected_top_event_candidates.json` | Step 12 | chosen candidate list | 13, 14, 15, 16 fallback | Required |
| `12_ranked_event_candidates.json` | Step 12 | full ranked list | 13 optional enrichment | Optional but produced |
| `13_vlm_event_inputs.json` | Step 13 | VLM media packages | 14, 16 gallery | Required for 14 |
| `14_vlm_event_reviews.json` | Step 14 | reviewed moments | 15, 16 fallback scene events | Required for 15 |
| `14_final_video_summary.json` | Step 14 | aggregate summary | 15, UI | Required for 15 |
| `15_searchable_events.json` | Step 15 | reviewed visible scene events | 16 preferred scene events | Optional preferred |
| `evidence_video.mp4` | Step 16 | final investigator video | UI/final output | Terminal artifact |

## Table 3: Model And Algorithm Map

| Stage | Model/algorithm | Purpose | Input | Output | CPU/GPU/API | Config variable |
| --- | --- | --- | --- | --- | --- | --- |
| 01-02 | OpenCV `VideoCapture` + JPEG writes | metadata and fixed sampling | source video | `01`, `02` artifacts | CPU | `TD_CASE2_INPUT_VIDEO`, `TD_CASE2_SAMPLE_EVERY_SECONDS` |
| 02A | adaptive motion sampling | sparse full-scene frame selection | Step 02 frames | `02A_adaptive_frames.json` | CPU | `TD_CASE2_ADAPTIVE_*` |
| 03A/03B | YOLO | audit and object detection | adaptive frames | detections, crops, annotations | GPU when available via device manager | `TD_CASE2_*YOLO*` |
| 04B | custom deterministic tracker | associate Step 03 detections into tracks | `03_yolo_detections.json` | `04B_tracks.json` | CPU | `TD_CASE2_TRACKING_*` |
| 04A, 06 | Florence-2 OCR/caption | audit and OCR/color enrichment | object crops | audit outputs or Step 06 OCR results | GPU when available | `TD_CASE2_FLORENCE_*` |
| 04A, 06 | plate detector YOLO | plate crop detection | vehicle crops | plate crops/candidates | GPU when available | `TD_CASE2_PLATE_DETECTOR_MODEL_PATH` |
| 06 | OCR cleaning + Indian plate verification + color extraction | verify plate text and normalize color | Florence/plate outputs | verified OCR/color fields | CPU + model calls | `TD_CASE2_STEP06_*` |
| 07B | search-text/token assembly | object search index | Steps 03/04B/05/06 | `07B` search records | CPU | `TD_CASE2_STEP07_*` |
| 08B | deterministic search matching | active search validation | `07B` records | validation results | CPU | runner-local |
| 11 | temporal rules and candidate scoring | scene-event proposal | adaptive frames + detections + tracks | Step 11 candidates | CPU | `TD_CASE2_STEP11_*` |
| 11.5 | Qwen 3B local or Qwen API | one-image scene filter | Step 11 candidate image | filtered candidates | local GPU or API or disabled fallback | `TD_CASE2_VLM_BACKEND`, `TD_CASE2_STEP11_5_*`, `TD_CASE2_QWEN_API_*` |
| 12 | deterministic ranking/clustering | select top VLM-worthy events | Step 11/11.5 candidates | Step 12 selected/ranked files | CPU | `TD_CASE2_STEP12_*` |
| 13 | OpenCV strip/contact-sheet generation | prepare review media | selected scene events + sampled frames | VLM input images | CPU | `TD_CASE2_STEP13_*` |
| 14 | Qwen 7B local or Qwen API | final event review | Step 13 media | reviewed events and summary | local GPU or API or disabled | `TD_CASE2_VLM_BACKEND`, `TD_CASE2_STEP14_*`, `TD_CASE2_QWEN_API_*` |
| 16 | OpenCV MP4 encoding + artifact merge | evidence-video generation | Steps 07B/11/12/13/14/15 + frames | final MP4 and index | CPU | `TD_CASE2_STEP16_*` |

## Table 4: Active Vs Legacy Files

| File | Active or legacy | Replaced by | Still imported by | Notes |
| --- | --- | --- | --- | --- |
| `run_td_case2_step07_search_index.py` | Legacy | `run_td_case2_step07b_traffic_object_search_index.py` | nobody in active orchestrators | Builds `07_vehicle_search_index.json` |
| `run_td_case2_step08_query_validation.py` | Legacy | `run_td_case2_step08b_dynamic_search_validation.py` | nobody in active orchestrators | Uses legacy Step 07 files |
| `run_td_case2_step09_result_packaging.py` | Legacy | `run_td_case2_step09b_universal_search_cards.py` | nobody in active orchestrators | Uses legacy Step 07/08 files |
| `run_td_case2_step10_search_demo.py` | Legacy | `run_td_case2_step10b_universal_search_demo.py` | nobody in active orchestrators | Uses legacy Step 07 files |
| `step_07_search_index_enrichment.py` | Legacy | `traffic_search_common.py` | Step 11 optional reads its output only | Source of Step 11 mismatch |
| `step_08_query_search_validation.py` | Legacy | runner-local 08B logic | legacy runner only | Not used by active search-ready path |
| `step_09_search_result_packaging.py` | Legacy | runner-local 09B logic | Step 15 uses `write_json_any` helper only | Helper import survives |
| `step_10_search_demo_runner.py` | Legacy | runner-local 10B logic | nobody in active orchestrators | Standalone old demo |

## Broken Or Mismatched Connections

- `MISMATCH` Step 11 optional search enrichment expects `07_vehicle_search_index.json`, but active search-ready produces `07B_traffic_object_search_index.json`.
  Source: `tests/td_case2/step_11_full_scene_event_candidates.py:1170`; `tests/td_case2/traffic_search_common.py:725-728`
- `OPTIONAL` Step 16 requires `07B_traffic_object_search_index.json` even if the user only wants reviewed scene-event export.
  Source: `tests/td_case2/run_td_case2_step16_evidence_video.py:82-89`
