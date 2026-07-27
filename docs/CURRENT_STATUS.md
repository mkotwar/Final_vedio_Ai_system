# CURRENT_STATUS.md

# Project Status

**Project Name:** AI Video Review & Investigation System

**Last Updated:** 2026-07-27

**Overall Progress:** ~45%

**Current Focus:**
We have successfully implemented the **Event Abstraction Layer**, completing Phase 1B (Event Quality). The aggregator now uses Semantic State Tracking (Entity & Activity overlap) with a 1-frame tolerance window, significantly reducing over-segmentation. We are preparing to move into Phase 3 (Performance Optimization).

Recent UI adjustment: the multicamera vehicle search results page now suppresses plate-image thumbnails and keeps plate evidence visible only on the track/global-vehicle detail pages, reducing clutter on result cards while preserving detailed evidence review.
Recent search-filter adjustment: the multicamera vehicle search page now exposes an explicit `All runs` processing-run option for structured search instead of auto-forcing the latest run into the filter.
Recent confidence-filter adjustment: multicamera search, local-track listings, and global-vehicle listings now default to a `0.50` minimum confidence threshold so sub-50% vehicle objects are hidden unless the operator deliberately lowers the filter.

## Phase Progress
* **Phase 1A (Data Completeness):** 100% COMPLETE. (Dynamic fallback mechanism implemented via Activity Recovery layer).
* **Phase 1B (Event Quality):** 100% COMPLETE. (Event Abstraction layer validated; 1-second fragmentation eliminated using semantic state tracking).
* **Phase 1C (Investigation Narrative Engine):** 100% COMPLETE. (Replaced statistical overview with investigation-grade dynamic narrative synthesis; resolved file lock truncation bugs).
* **Phase 2 (Semantic Search Integration):** 100% COMPLETE. (Search endpoint isolated by video ID and cosine similarity normalized for intuitive UI filtering).
* **Phase 3 (Performance Optimization):** 0% COMPLETE. (Next focus).

---

# Completed Components

## Video Ingestion

* [x] Video Upload API
* [x] Video Storage Management
* [x] Video Metadata Tracking

## Frame Processing

* [x] OpenCV Frame Extraction
* [x] Timestamp Generation
* [x] Adaptive Frame Sampling
* [x] Frame Storage

## OCR Pipeline

* [x] EasyOCR Integration
* [x] OCR Text Extraction
* [x] OCR Metadata Integration

## Vision-Language Processing

* [x] Qwen2.5-VL Integration
* [x] 4-bit NF4 Quantization
* [x] Native HF backend selected as current runtime backend
* [x] Native vLLM backend retained for future high-throughput batching
* [x] Ollama VLM backend removed from active architecture
* [x] Rich Scene Understanding
* [x] Object Detection Metadata
* [x] Activity Metadata
* [x] Caption Generation
* [x] Search Text Generation

## Metadata Processing

* [x] Metadata Validation
* [x] JSON Repair
* [x] Metadata Storage

## Event Processing

* [x] Event Aggregation Service
* [x] Event Similarity Scoring
* [x] Event Timeline Generation
* [x] Event Catalog Generation
* [x] Event Persistence

## Summary Pipeline

* [x] Summary Service Foundation
* [x] Summary API Endpoint
* [x] Event Catalog Loading
* [x] Event Path Resolution Fix
* [x] Event Schema Alignment Fix

## Infrastructure

* [x] FastAPI Backend
* [x] Pydantic Models
* [x] Docker Support
* [x] Docker Compose Support
* [x] Structured Logging

## Performance Optimization

* [x] Runtime Profiling
* [x] Bottleneck Analysis
* [x] VLM Optimization
* [x] Batch Optimization
* [x] Resolution Optimization

## Semantic Search & Vector Database

* [x] BGE-M3 Dense Embedding Service
* [x] Qdrant Integration Client
* [x] In-Memory Qdrant Local Fallback
* [x] Event Auto-Indexing Pipeline
* [x] Isolated ANPR artifact browser UI added for the completed 10 FPS validation run. The Streamlit UI reads JSON/JSONL artifacts only, reuses deterministic Step 10 structured search, and stores manual plate-review decisions separately under each run's `ui_state` directory.
* [x] Isolated ANPR artifact browser UI now renders scene/object evidence with centralized fixed-size image containers. The UI attempts full-frame path resolution from selected crops, search records, crop bundles, lifecycle crop candidates, and track observations before showing `Full frame unavailable`; the completed `163012` run currently has 0/167 records with saved full-frame paths.
* [x] Isolated streaming tracking pipeline now carries explicit `object_group` metadata through detection, ByteTrack, lifecycle, and Step 9 searchable records. Person records are supported and bypass plate YOLO, plate OCR, and vehicle-colour OCR. A saved audit for the completed `163012` vehicle run confirms zero person records because `object/vehical_detection/best_old.pt` has no `person` class. The corrected dedicated person model `object/Person_detection.pt` now loads successfully with class `0: person`; a 400-frame person validation run wrote 38 person searchable records with crop evidence under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_evidence_video_ultralytics_bytetrack_20260718_184838/`.
* [x] Added a combined vehicle + person full-video runner for the isolated streaming tracking pipeline. `run_combined_vehicle_person_pipeline.py` runs vehicle YOLO and person YOLO on each ordered frame, merges normalized detections, feeds one ByteTrack/lifecycle/crop stream, routes vehicles to ANPR/vehicle-colour processing, routes persons to deterministic visible-clothing colour analysis, and writes one combined Step 9/10 search index for one Streamlit UI. Full-video validation on `anpr_test_5min.mp4` produced 205 combined records under `debug_runs/streaming_tracking_combined_anpr_test_5min_20260718_191939/`: 145 vehicles and 60 persons, with UI classes `3wheeler`, `bus`, `car`, `motorcycle`, `person`, and `truck`. All 205 records have matching full-frame evidence; all 60 person records bypass ANPR.
* [x] POST /api/v1/search API Endpoint
* [x] Video UUID Query Filtering

## TD Case2 Streaming Tracking Prototype

* [x] Step 1 shared schemas/config/serialization foundation.
* [x] Step 2 sequential source/detection/tracking contracts and mock validation.
* [x] Step 3 real OpenCV + YOLO + ByteTrack sequential validation.
* [x] Step 4 application-level track lifecycle records, events, generations, and completion reasons.
* [x] Step 5 deterministic per-track observation and crop-candidate collection.
* [x] Step 6 deterministic final best-crop selection with primary/fallback outputs.
* [x] Step 7 bounded sequential plate YOLO plus Florence-2 raw OCR/vehicle-colour enrichment from Step 6 selected crop jobs.
* [x] Step 7.5 bounded plate-detection diagnostics and retry over Step 6 selected crops.
* [x] Image-folder ANPR validation for local plate YOLO and Florence OCR.
* [x] Step 9 artifact-only searchable vehicle records from completed track generations and Step 8 statuses.
* [x] Step 10 artifact-only structured search over Step 9 vehicle records.
* [x] Step 11 artifact-only UI-ready result-card packaging over Step 10 search results.

Steps 5, 6, 7, 7.5, image ANPR validation, bounded ANPR video validation, Step 8 plate validation, Step 9 searchable-record export, Step 10 structured search, and Step 11 result-card packaging remain isolated under `tests/td_case2/streaming_tracking_pipeline/`. Step 7 writes raw ANPR/colour artifacts only; Step 7.5 and image validation write diagnostic plate/OCR artifacts only. Step 8 consumes saved Step 7/7.5 artifacts and validates raw OCR into track-generation-level plate statuses without rerunning detection, tracking, plate YOLO, Florence, ByteTrack, or OpenCV video processing. Step 9 consumes only saved lifecycle/crop/ANPR/Step 8 artifacts to emit JSONL records for future search indexing. Step 10 consumes only Step 9 JSONL records for deterministic structured filtering, text-token matching, and ranking. Step 11 consumes only saved Step 9 and Step 10 artifacts to package UI-ready result cards and a static preview. These isolated steps do not alter tracking, index production search, emit events, add ReID, queues, threads, multiprocessing, production UI/API, or production integration.

Validation on 2026-07-18:

* Unit suite: 204 streaming tracking tests passed.
* Synthetic Step 5 run: 10 observations, 10 crop candidates, 1 completed crop bundle, 3 retained candidates.
* Real Step 5 run on `evidence_video.mp4`: 60 processed frames, 22 observations, 21 crop candidates, 13 retained candidates, 6 completed crop bundles, 1 too-small crop rejection.
* Real Step 5 artifacts: `debug_runs/streaming_tracking_pipeline/streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138/`.
* Synthetic Step 6 run: 16 completed bundles, 12 primary crops, 9 fallback crops, 5 primary-selected tracks, 9 fallback-only tracks, 2 no-valid tracks.
* Existing Step 5 artifact replay: 6 completed bundles, 3 primary crops, 6 fallback crops, 2 tracks with primary crops, 4 fallback-only tracks, 0 no-valid tracks, 6 previews.
* Real bounded Step 1-6 run: 60 processed frames through YOLO/ByteTrack/lifecycle/crops/selection with the same Step 6 selection counts as artifact replay.
* Real Step 6 artifacts: `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_existing_step5_20260718_144055/` and `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_evidence_video_ultralytics_bytetrack_20260718_144104/`.
* Synthetic Step 7 run: 1 processed track, 1 plate candidate, 1 raw OCR result, 1 normalized vehicle colour, artifacts under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step7_synthetic_20260718_150206/`.
* Existing Step 6 artifact replay for Step 7: 6 bounded identities, 9 selected crop jobs, local Florence loaded and colour inference ran without errors, plate YOLO ran on all selected vehicle crops, 0 plate candidates in this bounded replay, artifacts under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step7_existing_step6_20260718_150352/`.
* Synthetic Step 7.5 diagnostics: 15 synthetic tracks, 18 selected crop attempts, raw-box preservation, threshold classification, class/geometry/size rejection, primary/fallback retry, OCR empty/non-empty behavior, 10 accepted plate candidates, 8 tracks with non-empty OCR, artifacts under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step75_synthetic_20260718_152215/`.
* Existing Step 6 artifact replay for Step 7.5: 9 selected crop jobs inspected from `streaming_tracking_step6_existing_step5_20260718_144055`, plate model `OCR_MUKUL/license_plate_weights.pt`, CPU, normal threshold `0.20`, diagnostic thresholds `0.25,0.15,0.10,0.05`, model classes `{0: License_Plate}`, 9 detector calls, 0 raw detector boxes, 0 boxes below normal threshold, 0 class/geometry/size rejections, 0 accepted plate candidates, 0 OCR calls, 6 tracks exhausted, 9 annotated vehicle crops, artifacts under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step75_existing_step6_20260718_152224/`. A known-good check on `debug_runs/ANPR1-D_20260715_175538/06_plate_crops/vehicle_track_0001_frame_000030_combined_001_plate.jpg` produced 1 raw/accepted box at confidence `0.776766`, so the replay result is truly zero boxes on the selected crops, not a loader failure.
* Image ANPR validation on `debug_runs/test pictures`: 7 images discovered/read, plate model `OCR_MUKUL/license_plate_weights.pt`, Florence base `C:\Mukul K\mk\models\Florence-2-base-ft`, adapter `OCR_MUKUL/adaptor_florance_baseFT`, CPU float32, normal threshold `0.25`, diagnostic thresholds `0.25,0.15,0.10,0.05`, 7 plate model calls, 13 raw detector boxes, 7 images with accepted plates, 1 box below normal threshold, 0 class/geometry/clipping/size rejections, 13 accepted crops saved, 7 detector-crop OCR calls with 7 non-empty raw outputs, 7 direct-input OCR calls with 7 non-empty raw outputs, artifacts under `debug_runs/streaming_tracking_pipeline/image_anpr_validation_20260718_153928/`.
* Bounded 10 FPS ANPR-video validation on `C:\Mukul K\mk\test_video\anpr_test_5min.mp4`: 600 selected frames at target 10 FPS over frames 0-1797 (`59.9` seconds), source FPS `30.0`, source frames `11336`, custom vehicle model `object/vehical_detection/best_old.pt` on CUDA with no fallback, plate model `OCR_MUKUL/license_plate_weights.pt` on CUDA with no fallback, Florence base `C:\Mukul K\mk\models\Florence-2-base-ft` plus adapter `OCR_MUKUL/adaptor_florance_baseFT` on CUDA float16 with no quantization and no fallback. The run produced 106 filtered detections (`bus=24`, `car=39`, `motorcycle=19`, `truck=24`), 20 raw tracker IDs, 61 track observations, 23 completed vehicle track generations, 61 crop candidates, 56 retained candidates, 18 selected fallback vehicle crops, 0 selected primary vehicle crops, 17 ANPR-eligible jobs, 1 crop excluded by the minimum vehicle-crop-size gate, 17 plate detector calls, 12 raw/accepted plate boxes, 12 non-empty raw OCR outputs, and 17 colour calls. Runtime improved from the previous CPU-Florence bounded run (`45.403463` seconds total, `32.13337` seconds Florence) to the auto-CUDA run (`19.936328` seconds total, `4.731093` seconds Florence). CUDA was available on `NVIDIA GeForce RTX 5070 Ti`, and peak allocated VRAM reported by PyTorch was `718.092 MB`. Artifacts were written under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/`.
* Step 8 plate validation ran against `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/` without rerunning inference. It read 12 raw OCR records, produced 12 normalized candidates, applied 2 controlled corrections (`I->1` and `1->I`), found 3 strict-format candidate matches, 0 relaxed matches, 6 partial candidates, and 3 not-plate-like candidates. Final statuses over 23 track generations were 3 verified, 6 weak, 3 invalid, and 11 no-plate-detected. Verified plate texts were `UP81CH4158`, corrected `UP81CW4150`, and `UP14CW4087`. There were 0 exact-agreement groups and 0 similarity-based agreements because this bounded run produced one OCR candidate per OCR-bearing track generation. Artifacts were written under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/08_plate_validation/`.
* Full-video Step 9 searchable-object export ran against `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/` without rerunning models or changing tracking/crop selection. It consumed 167 completed tracks, 167 selected crop sets, 145 ANPR/colour records, and 167 Step 8 final plate-validation records; emitted 167 generation-aware vehicle records; and preserved Step 8 statuses as 58 verified, 28 weak, 14 invalid, and 67 no-plate-detected. Query smoke checks returned 27 white cars, 58 verified plates, 1 exact `UP81CH4158` match, 2 `UP81` prefix matches, 17 red vehicles, 16 vehicles between 60 and 120 seconds, 28 weak-OCR records, and 67 no-plate records. Artifacts were written under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/09_searchable_objects/`.
* Step 10 structured search validation ran against `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/09_searchable_objects/searchable_vehicle_records.jsonl` without rerunning inference or mutating Step 9 records. It indexed 167 records and executed 11 deterministic validation queries using class, colour, plate text/prefix, plate status, track identity, and overlap-based time filters. Counts matched the expected Step 9 smoke checks: `white car=27`, `verified plates=58`, `UP81CH4158=1`, `UP81=2`, `red vehicle=17`, `vehicles between 60 and 120 seconds=16`, `weak OCR=28`, and `no plate=67`. Additional structured checks returned `motorcycle without plate=11`, `white car between 2 and 3 minutes=8`, and `truck with verified plate=3`. Artifacts were written under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/10_structured_search/`.
* Step 11 result-card packaging ran against `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/` without rerunning inference, search indexing, or validation. It consumed `09_searchable_objects/searchable_vehicle_records.jsonl` and `10_structured_search/validation_search_results.jsonl`, packaged 11 demo queries into 138 UI-ready cards, preserved Step 10 rank/score order, joined Step 9 confidence/duration fields, and wrote JSON artifacts plus a static HTML preview under `11_result_cards/`. The package contains 133 cards with vehicle images, 98 cards with plate images, 5 missing vehicle-image warnings, 40 missing plate-image warnings, 61 verified cards, 30 weak cards, 40 no-plate cards, and 7 invalid cards. The generated HTML has 231 image references and all referenced files resolve locally.
* Added an object-class and colour-correction layer for the isolated streaming tracking pipeline: central class normalization now preserves raw model class ID/name while exposing normalized classes (`person`, `car`, `motorcycle`, `bicycle`, `bus`, `truck`, `other_vehicle`, `other_object`), and `motorcycle`/`motorbike`/`scooter`/`two_wheeler` never normalize to `car`. The object folder inspection found `object/vehical_detection/best_old.pt` with raw classes `{0: 3Wheeler, 1: bus, 2: car, 3: motorcycle, 4: truck}` and separate normalized motorcycle support. The local `object/Person_detection.pt` file is present but cannot be loaded by Ultralytics/PyTorch because its zip archive contains a single root entry `Person_detection.pt`, causing `file in archive is not in a subdirectory`; real person inference is blocked until that checkpoint is repaired or re-exported.
* Added class-aware multi-model detection combination and duplicate suppression helpers, plus dominant foreground/body colour analysis with debug artifacts. Validation on 40 saved Step 9 crops wrote `12_object_class_colour_validation/colour_debug/` artifacts with original crops, analysed regions, exclusion masks, and dominant-colour reports. The analyser rejected small-region colour pollution in the sampled run; for example `anpr_test_5min:track_000019:gen_000` had raw Florence colour `red` but central-body dominant colour `gray` with coverage `0.648272`.

---

# Current Architecture

```text
Video Upload
    ↓
Frame Extraction
    ↓
Adaptive Sampling
    ↓
OCR
    ↓
Qwen2.5-VL
    ↓
Frame Metadata
    ↓
Event Aggregation
    ↓
Event Catalog
    ↓
Investigation Services
```

---

# Current Performance

## Hardware

GPU:
RTX 5070 Ti 16GB

RAM:
32GB

## Profiling Results

Qwen2.5-VL:
80–94% of total runtime

OCR:
5–19% of total runtime

Everything Else:
<1%

## Average Processing Speed

7–10 seconds per analyzed frame

## Current Bottleneck

Qwen2.5-VL inference

No further OCR or storage optimization is currently prioritized.

---

# In Progress

## Event Quality Improvements

* [ ] Event abstraction layer
* [ ] Event normalization
* [ ] Event classification improvements
* [ ] Better event descriptions
* [ ] Better event type detection

## Summary Generation

* [x] Statistics engine
* [x] Timeline generation
* [x] Peak activity detection
* [x] Notable event detection
* [x] Human-readable overview generation (Investigation Narrative Engine)
* [ ] Investigation report generation

## Validation

* [ ] Multi-video testing
* [ ] Long-duration video testing
* [ ] Summary accuracy testing

---

# Known Issues

## Event Descriptions

Current event descriptions are generated by concatenating frame captions.

Example:

```text
Blue car driving.
Blue car driving with pedestrian.
Blue car driving near building.
```

Desired:

```text
Blue car moved through monitored area.
```

---

## Event Similarity Threshold

Current threshold may be too strict.

Potential impact:

* Event fragmentation
* Over-segmentation
* Reduced summary quality

Needs tuning and validation.

---

## Event Typing

Current event typing is basic.

Future event categories should include:

* vehicle_entry
* vehicle_exit
* pedestrian_entry
* pedestrian_exit
* vehicle_movement
* pedestrian_crossing
* restricted_area_activity
* stationary_vehicle
* suspicious_activity

---

## Summary Quality

Summary quality is currently dependent on event quality.

Event abstraction must improve before introducing LLM-based summaries.

---

# Next Priorities

## Immediate

* [ ] Implement investigation report generation

## Short-Term

* [ ] Complete Phase 1
* [ ] Validate summary quality
* [ ] Create benchmark test videos

## Medium-Term

* [ ] Set up Docker containerization for Qdrant service
* [ ] Integrate Celery/Redis asynchronous task queue

---

# Upcoming Phases

## Phase 2 — Semantic Search

Status:
✅ Completed

Deliverables:

* [x] Event Embeddings (BGE-M3 local model)
* [x] Qdrant Integration (in-memory local client & remote server config)
* [x] Search Service (indexing with stable UUIDs & filtering)
* [x] Search API (POST /api/v1/search)
* [x] Natural Language Search

Examples:

* Find red motorcycle
* Find person carrying backpack
* Find white truck

---

## Phase 3 — Object Tracking

Status:
Not Started

Recommended Stack:

YOLO + ByteTrack

Deliverables:

* Persistent Track IDs
* Track Timelines
* Track Storage

---

## Phase 4 — Investigation Analytics

Status:
Not Started

Deliverables:

* Area Analytics
* Path Analytics
* Dwell Analytics
* Speed Analytics

---

## Phase 5 — Person & Vehicle ReID

Status:
Not Started

Deliverables:

* Person ReID
* Vehicle ReID
* Appearance Similarity Search

---

## Phase 6 — Multi-Camera Investigation

Status:
Not Started

Deliverables:

* Cross-Camera Correlation
* Entity Linking
* Investigation Timelines

---

## Phase 7 — Investigation Dashboard

Status:
Not Started

Recommended Stack:

React + FastAPI

Deliverables:

* Upload UI
* Search UI
* Timeline Viewer
* Analytics Dashboard
* Investigation Workspace

---

# Sprint Notes

## Sprint 2026-06-04

Completed:

* Investigated "no_events" summary issue
* Identified event catalog path mismatch
* Fixed event catalog generation
* Fixed summary service loading
* Fixed schema alignment
* Added detailed logging
* Implemented dynamic event catalog fallback and auto-rebuilding from individual event JSON files
* Implemented dynamic on-the-fly event generation fallback from raw frame metadata if event files are entirely missing

Outcome:

Summary pipeline can now successfully load consolidated events, with a robust two-layer auto-rebuilding fallback mechanism (restores consolidated index from individual events, or regenerates events from frame metadata on-the-fly).

Next Sprint Focus:

Event abstraction and summary quality improvements.

---

# Change Log

## 2026-07-15

* Configured the isolated `tests/td_case2` local Qwen loaders (Step 11.5 3B and Step 14 7B) for strict BitsAndBytes 4-bit NF4 inference with double quantization and BF16/FP16 compute. CUDA is required and no higher-precision fallback is used. Pre-quantized Unsloth checkpoints are stored under the workspace `models/` directory and are now the default local model paths.
* Fixed the `td_case2` search-ready orchestrator to propagate the freshly created Step 01-02A run directory to all downstream steps, instead of reusing a stale `TD_CASE2_RUN_DIR` value that could point at the input video.
* Fixed Step 03 YOLO configuration so relative model paths loaded from `tests/td_case2/.env` resolve from the testcase directory; the portable `../../yolo11m.pt` fallback now locates the repository-root model correctly.
* Added the missing `einops` runtime dependency required by the local Florence-2 custom model code used in td_case2 Steps 04A and 06.
* Forced eager attention when loading the local Florence-2 custom model, avoiding incompatible SDPA capability probing under the installed Transformers runtime.
* Added search-ready pipeline resume support after completed Step 03 outputs and a safe td_case2 `.env` fallback when an inherited Florence model path is invalid.
* Fixed Florence-2 generation under Transformers 4.57 by disabling the incompatible KV cache, enabled the available license-plate detector for td_case2, forced Step 06 to recompute unless reuse is explicitly requested, and made all-crop inference failure a blocking stage error instead of a false success.
* Validated the repair on `anpr_test_5min_20260715_140144`: Step 06 completed 157/157 crops, detected colors on 83 tracks, found 61 plate crops, produced 23 format-valid plate candidates and 3 verified plates; rebuilt Steps 07B-10B with all 42 dynamic search validations passing.
* Updated td_case2 model-path resolution to accept existing paths relative to the launch directory while retaining testcase-relative `.env` paths, preventing interactive `yolo11m.pt` and `OCR_MUKUL/license_plate_weights.pt` values from being misresolved.
* Configured all GPU-capable td_case2 models for explicit CUDA execution: combined YOLO, plate-detector YOLO, Florence OCR/color, Qwen 3B filtering, and Qwen 7B review. Plate detection now receives the resolved Florence/Step 06 device explicitly.
* Replaced the td_case2 CPU-only PyTorch build with the official Windows CUDA 12.8 stack (`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`) on the RTX 5070 Ti. Verified real CUDA inference for combined YOLO, plate YOLO, Florence, Qwen 3B NF4, and Qwen 7B NF4; Qwen smoke tests used approximately 2.34 GB and 6.58 GB allocated VRAM respectively.
* Refactored td_case2 into a GPU-first, self-configuring runtime. A centralized device manager now resolves CUDA vs CPU once and feeds YOLO, Florence, plate detection, and local Qwen stages. New runs write `gpu_utilization_report.json`, capturing the actual execution device, VRAM snapshot, and the CPU-only justification for non-accelerated stages.
* Validated the updated runtime on `debug_runs/anpr_test_5min_20260715_143548`: Step 03 YOLO inference executed on `cuda:0`; Step 04A Florence audit and plate detection executed on `cuda:0`; Step 06 OCR/color enrichment executed on `cuda:0`; Step 11.5 local Qwen 3B executed on `cuda:0` with approximately 2.35 GB allocated VRAM; Step 14 local Qwen 7B executed on `cuda:0` with approximately 8.27 GB allocated VRAM. CPU-only stages remain motion sampling, deterministic tracking, event/search ranking, and JSON/report assembly.
* Extended td_case2 Step 16 evidence-video generation so the final `evidence_video.mp4` can append two visual galleries in the same output: deduplicated unique full-scene object-detection frames from the searchable object index and the Step 13 VLM input media (temporal strips/contact sheets/primary frames). Validated on `debug_runs/ANPR1-D_20260715_175538`, which produced 1,200 object-gallery frames and 4 VLM-input gallery frames in the final MP4.
* Removed Florence `<DETAILED_CAPTION>` inference from td_case2 Steps 04A/06 while retaining the compatibility fields and using only `<OCR>` plus `<CAPTION>`. Step 06 now extracts conservative structured vehicle, plate, and scene attributes and carries them into both Step 07 search indexes.
* Improved td_case2 plate recovery with 0.20 detector confidence, 960-pixel inference, padded plate crops, an enhanced CLAHE OCR fallback, original-image preference when both OCR variants are valid, and real Indian state/UT-prefix checks before verification. Vehicle crops now receive 5% context padding in Step 03 for future full runs.
* Validated Step 06 on `anpr_test_5min_20260715_151259` using the same 256 selected crops: plate crops increased 113 to 125, format-valid tracks 49 to 52, verified tracks 13 to 19, and verified unique plates 12 to 17. Total crop processing time decreased from 145.14 to 96.49 seconds; Florence detailed-caption calls decreased from 256 to zero. Rebuilt Steps 07 and 07B successfully.
* Replaced td_case2's fixed Florence color-word scan with free-caption shade normalization plus a central-crop HSV fallback. Decorative shade names such as pearl white, navy, burgundy, champagne gold, bronze, and charcoal are stored as canonical search colors, while plate colors are no longer mistaken for vehicle colors. Revalidation populated a canonical color for all 242 vehicle tracks in `anpr_test_5min_20260715_151259`, with zero unknown/empty Step 06 or Step 07 vehicle colors.
* Added td_case2 event-preview clip generation for search results. The traffic search UI can now build and preview per-result MP4 clips from the processed event frame sequence, overlay the running video timestamp plus detected start/end window, hold on the best frame, and persist clip metadata under `10C_search_event_clips_manifest.json`.
* Added an integrated `td_case2` Streamlit workbench UI that combines video upload, end-to-end pipeline controls, editable stage parameters, object/event search with clip previews, event artifact inspection, and VLM summary review in one page.
* Added td_case2 Step 16 evidence video generation. The VLM pipeline now finishes by building `evidence_video.mp4`, `evidence_video_index.json`, and `16_evidence_video_report.json` from existing event/search artifacts only, without re-running AI models.
* Step 16 selects chronological, high-value searchable events, deduplicates near-identical plate/track evidence, adds investigator overlays plus title cards, and records clip-to-source traceability for future UI jump navigation.
* Surfaced Step 16 outputs inside the td_case2 review UIs so investigators can preview the final evidence MP4 and inspect the per-clip index directly from the workbench / summary dashboard.
* Fixed td_case2 Step 13 non-merged VLM input generation so separate event review strips can be produced when `TD_CASE2_STEP13_MERGE_NEARBY_SELECTED=false`; re-reviewed the accident short with 10 separate VLM inputs and detected 4 high-risk collision moments.

## 2026-07-16

* Fixed the isolated `tests/td_case2` ByteTrack LAP shim to use `scipy.optimize.linear_sum_assignment` instead of the removed `ultralytics.utils.ops.linear_sum_assignment` helper. This unblocks the dynamic YOLO tracking experiment and the dense ByteTrack experiment under the current Ultralytics build.
* Revalidated the isolated tracking tests with `--noconftest`: `tests/td_case2/experiments/dynamic_yolo_tracking/test_dynamic_fps_controller.py`, `tests/td_case2/experiments/step04b_bytetrack/test_tracking_experiment.py`, and `tests/td_case2/test_step03_yolo_config.py` now pass 33/33 in the workspace root `.venv`.
* Optimized the isolated td_case2 fragment-merging stage to short-circuit invalid geometric pairs before any crop I/O, cache up to three representative HSV descriptors per track, cap merge candidates per source track through `TD_CASE2_EXP_MERGE_MAX_CANDIDATES_PER_TRACK`, and emit merge-stage progress plus timing metrics without changing the active td_case2 pipeline.
* Added a merge-only resume mode to the isolated dynamic YOLO tracking experiment so existing `single_pass_dynamic_tracks_raw.json` outputs can be finalized into merged tracks, audit, report, and preview artifacts without rerunning decoding, motion selection, YOLO, or ByteTrack.
* Validated the optimized merge-only resume flow on `debug_runs/traffic moderate 3.5min_20260716_150146`: the interrupted run completed from raw tracks in 0.246 seconds of merge time, reduced 154 raw tracks to 140 merged tracks through 18 accepted merge operations, and preserved the preexisting frame-selection / detection / raw-track artifacts unchanged.
* Added an isolated active-pipeline preflight helper at `tests/td_case2/check_td_case2_readiness.py` plus an optional launcher `tests/td_case2/run_current_td_case2.ps1` for new-PC setup. The checker validates the current search-ready and VLM/event runners, package/runtime availability, CUDA visibility, writable output roots, model-path loadability, and offline local-model readiness without running video inference or changing pipeline outputs.
* Audited the active `tests/td_case2` pipeline on the July 16, 2026 Windows workstation. The validated search-ready configuration uses the workspace root `.venv`, `object/Person_detection (1)/Person_detection.pt`, `object/vehical_detection/best_old.pt`, `ocr_colour/license_plate_weights.pt`, `C:\Mukul K\models\Florence-2-base-ft`, and the repo-level `debug_runs` output root. The code default `object/vehical_detection` directory remains invalid for Ultralytics unless overridden to the actual `.pt` file.
* Updated the active isolated `td_case2` local Qwen loaders so Step 11.5 and Step 14 can load normal FP16/BF16 Hugging Face checkpoints directly into runtime BitsAndBytes 4-bit NF4 mode, while still accepting already prequantized NF4 checkpoints. The loaders now enforce strict CUDA-only 4-bit verification, offline processor preflight checks, GPU-memory/load-time reporting, and explicit model cleanup between the 3B and 7B stages.
* Added isolated runtime validation coverage for the new local Qwen loading path at `tests/td_case2/test_qwen_4bit.py` and `tests/td_case2/validate_qwen_runtime_4bit.py`.
* Revalidated local Qwen runtime loading on this workstation using the restored normal checkpoints at `C:\Mukul K\models\Qwen2.5-VL-3B-Instruct` and `C:\Users\Vinfocom\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5`. Runtime NF4 validation passed for both models with BF16 compute; the 3B load used approximately 2304.50 MB allocated GPU memory and the 7B load used approximately 5690.11 MB allocated GPU memory.

* Added td_case2 Step 15 searchable reviewed-event generation so Step 14 `collision` / `near_miss` decisions are persisted as standalone searchable scene events instead of being reconstructed later from weaker Step 11 labels.
* Hardened Step 12 and Step 13 against silent loss of accident evidence: Step 11.5 `yes` detections with critical visible-event types now receive forced preservation through ranking, and Step 13 no longer merges critical accident candidates into ordinary nearby traffic groups.
* Updated Step 14 and Step 16 final summaries to explicitly surface collision evidence, and updated Step 16 to consume the new Step 15 reviewed-event file so evidence export preserves reviewed collision truth, priority, and timestamps end-to-end.

## 2026-07-17

* Added an isolated `tests/td_case2/hybrid_tracking_test` Streamlit manual-review workflow for the `post_tracking_v2` artifacts. Reviewers can inspect reconciled local objects, accepted merges, and possible merges; autosave per-decision judgments; generate cached short review clips; export manual ground-truth groups; and write comparison summaries under `post_tracking_v2/manual_review/` without modifying production code or Step 04B outputs.
* Added an isolated `tests/td_case2/continuous_mot_hybrid` experiment that processes a video sequentially at 10 FPS with adaptive YOLO scheduling, ByteTrack/BoT-SORT backend support, optional short-gap visual bridging, post-tracking integrity checks, fragment reconciliation, representative-frame selection, local identity packaging, and a three-pipeline comparison against the previous `td_case2` vs KCF-hybrid run. The first validation run on `Untitled design.mp4` completed end-to-end without touching production code, but current integrity and crop gating are conservative enough that the generated identity packages all fall into rejected/manual-review territory and need iteration before this experiment is a stronger OCR/search base.
* Added a second isolated validation flow at `tests/td_case2/continuous_mot_hybrid/run_fixed_5fps_bytetrack_validation.py` for a simpler continuous-tracking audit: 10 FPS processing, fixed 5 FPS YOLO, ByteTrack-only lifecycle handling, prediction-only skipped-frame handling, no visual tracker, and no reconciliation/crop/OCR stages. The July 17, 2026 validation run on `Untitled design.mp4` produced 151 raw track IDs with 51 confirmed tracks, reduced the sub-0.5-second fragments from 259 to 98 versus the earlier adaptive continuous run, and recorded `tracks_lost_due_to_skipped_detector_frame = 0`. The fixed-flow validation passed its explicit skipped-frame safety checks, but the run still produced 100 tentative tracks, 151 total IDs, only 3 successful reactivations versus 37 failed reactivation candidates, and 98 `removed_from_tracker` terminations, so ByteTrack lifecycle quality still needs further iteration before this isolated path is strong enough to replace the earlier KCF hybrid as the best search/OCR base.
* Added a third isolated fixed-flow validation at `tests/td_case2/continuous_mot_hybrid/run_fixed_5fps_reactivation_validation.py` with a stable local-object identity layer, recoverable-track snapshots, candidate indexing, recovery scoring, and one-to-one remap assignment above raw ByteTrack IDs. The July 17, 2026 validation run on `Untitled design.mp4` preserved the skipped-frame safety result (`tracks_lost_due_to_skipped_detector_frame = 0`) and emitted the new recovery/debug artifacts under `debug_runs/fixed_5fps_reactivation_Untitled design_20260717_170641`, but it did not yet reduce fragmentation: the run still ended with 151 stable local-object IDs for 151 raw tracker IDs, 0 tracker-ID remaps, 0 successful or possible recoveries, and 25 rejected recovery attempts with mean score `0.731957`, so the application-layer recovery thresholds and candidate coverage still need iteration before this path can practically outperform the simpler fixed ByteTrack baseline.
* Added an isolated tracker-backend comparison flow under `tests/td_case2/continuous_mot_hybrid/` that runs one shared 10 FPS / fixed 5 FPS YOLO detection pass, replays the exact same cached detections into ByteTrack, BoT-SORT without ReID, and BoT-SORT with requested ReID, and emits per-backend metrics, visual review cases, config diffs, and review-only identity-switch candidates under `debug_runs/tracker_backend_comparison_<video>_<timestamp>/`.
* Validated the comparison flow on `Untitled design.mp4` using Ultralytics `8.4.95` and shared detection-cache checksum `674fd2401234fb31144c52e021108c6a2bd442aaa6ec983dea1efa6f70972099`. All three replays preserved skipped-frame safety with `tracks_lost_due_to_skipped_detector_frame = 0` and consumed the same 794 detector frames from one YOLO cache.
* The July 17, 2026 comparison result favored the existing fixed-flow ByteTrack baseline over both BoT-SORT variants for this cached-replay isolation test. ByteTrack finished in 13.61 seconds with 151 raw IDs, 51 confirmed tracks, 98 sub-0.5-second fragments, 3 successful reactivations, and 98 removals. BoT-SORT without ReID finished in 31.44 seconds with 161 raw IDs, 38 confirmed tracks, 121 sub-0.5-second fragments, 0 successful reactivations, and 121 removals. The BoT-SORT `with_reid=true` replay finished in 29.96 seconds but matched the no-ReID metrics exactly, so it did not improve continuity in this configuration.
* Recorded that BoT-SORT `model: auto` ReID was not actually active during cached replay on this workstation. The runtime verification file reported `requested_with_reid = true`, `actual_with_reid = false`, `encoder_initialized = false`, `feature_vector_count = 0`, and fallback reason `cached_detection_replay_has_no_native_detector_features_for_model_auto`. We therefore mark the current cached-replay BoT-SORT + ReID path as `reid_not_available` rather than claiming an appearance-assisted comparison.
* Added a second isolated ReID-validation flow under `tests/td_case2/continuous_mot_hybrid/` that audits Ultralytics ReID behavior, inventories local ReID/classification checkpoints without downloading anything, captures native detector features from live `yolo11m.pt` detector frames at the fixed 5 FPS schedule, caches those features under `01_shared/reid_features.npz`, and replays both ByteTrack and BoT-SORT from the same 10 FPS / 5 FPS fixed schedule.
* Validated the verified ReID flow on July 17, 2026 in `debug_runs/verified_botsort_reid_Untitled design_20260717_191135`. The audit confirmed Ultralytics `8.4.95` uses a Detect-layer pre-hook in `track.py` for `model:auto`, that cached box-only replay cannot activate native ReID, and that `yolo11m.pt` is a standard non-end2end Detect model suitable for native feature capture. No local fallback `yolo11*-cls.pt`, `yolo26*-cls.pt`, or `yolo26*-reid.onnx` checkpoints were found in the repo or checked local caches, so the verified run used the native-feature path rather than an external encoder model.
* The verified BoT-SORT ReID run was genuinely active: `requested_with_reid = true`, `actual_with_reid = true`, `encoder_initialized = true`, `feature_vector_count = 383`, `feature_dimension = 256`, and `appearance_comparison_count = 398`, with 23 appearance-threshold acceptances and 375 rejections. The shared fixed-schedule detector cache checksum was `c309bc27df659a7b1b7e89442136e42cfa33ef21043928345097e352a707b065`, and skipped-frame safety remained intact with `tracks_lost_due_to_skipped_detector_frame = 0`.
* Despite genuine ReID activation, ByteTrack still outperformed BoT-SORT ReID on this single-camera traffic clip. ByteTrack finished in 12.99 seconds with 151 raw IDs, 51 confirmed tracks, 100 tentative tracks, 98 sub-0.5-second fragments, 3 successful reactivations, and 98 removals. Verified BoT-SORT ReID finished in 29.94 seconds with 160 raw IDs, 38 confirmed tracks, 122 tentative tracks, 120 sub-0.5-second fragments, 0 successful reactivations, and 120 removals. The only practical target it improved versus the ByteTrack baseline was a small interior new-ID reduction from 84 to 82; the other fragmentation and continuity targets remained worse, so ByteTrack remains the stronger isolated baseline for this video.

## 2026-07-18

* Added an isolated Step 1 foundation for the future `td_case2` streaming tracking pipeline under `tests/td_case2/streaming_tracking_pipeline/`. The new package defines shared dataclass schemas, constrained lifecycle/enrichment enums, JSON-safe serialization helpers, reusable validators, nested configuration dataclasses, environment-variable overrides with the `TD_CASE2_STREAM_` prefix, and a README documenting the architecture boundary.
* Kept the new streaming foundation fully isolated from existing `td_case2` production-oriented stages. Step 1 intentionally does not implement video decoding, YOLO inference, ByteTrack execution, Florence inference, plate detection, crop scoring, queues, workers, event detection, search indexing, or production integration.
* Validated the new foundation with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 30 tests passed. `pytest` is not installed in the available Python environments, so the focused tests were run through `unittest`.
* Completed Step 2 for the isolated streaming tracking pipeline: added sequential source/stage contracts, a deterministic synthetic frame source, exact timestamp-based frame-selection planning, model-free mock detection/tracking stages, tracker-ID normalization with `source_track_id` preservation, current Step 03/04B compatibility adapters, in-memory/JSONL packet sinks, a one-frame-at-a-time sequential pipeline, and a CPU-only contract-validation runner.
* Validated Step 2 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 58 tests passed. Also ran `tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step2_contract_validation`, which processed 7 selected synthetic frames, wrote `frame_packets.jsonl`, `detection_packets.jsonl`, `tracked_frame_packets.jsonl`, and `step2_contract_validation_report.json` under `debug_runs/streaming_tracking_pipeline/step2_contract_validation/`, and passed ordering/count checks.
* Known Step 2 limitations: no real video decoding, RTSP, YOLO, ByteTrack instantiation, lifecycle completion, crop selection, OCR/color enrichment, queueing, threading, search indexing, event detection, or production integration. Step 3 should connect the real frame/video source, real YOLO detector, and real ByteTrack backend in a sequential reference runner before any queue-based streaming work.
* Completed Step 3 for the isolated streaming tracking pipeline: added `OpenCvVideoSource`, a real Ultralytics YOLO detection stage, Ultralytics and Supervision ByteTrack adapters, a local LAP shim for Ultralytics tracker imports, tracking-stability summary metrics, a real one-frame-at-a-time runner, Step 3 artifact sink, optional annotated-video output, and focused unit coverage. The implementation remains isolated under `tests/td_case2/streaming_tracking_pipeline/` and does not add queues, workers, crop collection, OCR, plate detection, Florence, events, search, or production pipeline changes.
* Validated Step 3 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 70 tests passed. Also ran `py_compile` on the new Step 3 modules and confirmed the validation runner help command works.
* Real Step 3 validation results: `ultralytics_bytetrack` passed on `data/videos/00000000-0000-0000-0000-000000000001.mp4` at 5 FPS / 15 processed frames with `yolo11n.pt`, producing zero detections on that tiny sample but proving the real video -> YOLO -> ByteTrack -> JSONL/report path. `supervision_bytetrack` failed clearly in backend-comparison mode because `supervision` is not installed.
* Additional Step 3 real-object validation passed on `debug_runs\vidssave.com Woman crashes into lamppost, flips car during driving test 720P_20260716_112408\evidence_video.mp4` at 4 FPS / 60 processed frames with `yolo11n.pt`, low detector/tracker thresholds, and annotated-video output. The run emitted 43 filtered detections, 22 track observations, 2 unique track IDs, and wrote artifacts under `debug_runs/streaming_tracking_pipeline/step3_real_evidence_video_ultralytics_bytetrack_20260718_134920/`.
* Configured vehicle-model smoke validation passed on `debug_runs\test_anpr_day_10min_20260715_155615\10C_search_event_clips\obj_track_vehicle_track_0001_event_preview.mp4` using `object\vehical_detection\best_old.pt`, but the 2 FPS / 4-frame clip produced zero detections. This is recorded as an environment/model smoke result, not a tracking-quality benchmark.
* Known Step 3 limitations: Supervision ByteTrack is implemented but unavailable until the `supervision` package is installed; the strongest object-tracking validation used relaxed YOLO/ByteTrack thresholds because the available short local evidence clip has sparse detections; lifecycle completion, crop selection, OCR/color enrichment, search indexing, event creation, and production integration remain deferred. Step 4 should build lifecycle/crop-candidate logic on top of the validated sequential tracked-frame stream.
* Completed Step 4 for the isolated streaming tracking pipeline: added a reusable deterministic application-level `TrackLifecycleManager`, lifecycle event schemas, lifecycle configuration, lifecycle metrics, a sequential lifecycle pipeline, artifact sink, and `run_step4_lifecycle_validation.py`. The implementation consumes `TrackedFramePacket` objects after ByteTrack and does not modify ByteTrack internals or existing `td_case2` Step 04B behavior.
* Step 4 lifecycle policy now supports tentative, confirmed, temporarily-lost, and completed states; processed-frame missed counts; optional timestamp expiry; same-ID recovery before expiry; end-of-stream flush; class-vote updates; and completed-ID reuse through backward-compatible `track_generation` values on `TrackRecord`.
* Validated Step 4 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 82 tests passed. Also ran `py_compile` on `lifecycle.py`, `lifecycle_metrics.py`, `lifecycle_pipeline.py`, `run_step4_lifecycle_validation.py`, `config.py`, and `schemas.py`.
* Synthetic Step 4 validation passed with `tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step4_lifecycle_validation --mode synthetic_lifecycle`, covering tentative-to-confirmed, confirmed-to-temporarily-lost, same-ID recovered, lost-buffer completed, tentative invalid completion, video-end flush, completed-ID generation increment, and class-vote changes. Artifacts were written under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step4_synthetic_20260718_140338/`.
* Real Step 4 validation passed with `yolo11n.pt`, `ultralytics_bytetrack`, the Step 3 evidence video, 4 FPS, 60 processed frames, and the same low detector/tracker thresholds used for Step 3. The run emitted 43 filtered detections and 22 raw track observations over 2 raw tracker IDs; the lifecycle layer created 6 track generations, confirmed 2 tracks, emitted 2 temporarily-lost transitions, emitted 0 recoveries, and completed 6 tracks with completion reasons `invalid_track=3`, `lost_buffer_expired=2`, and `video_ended=1`. Artifacts were written under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step4_evidence_video_ultralytics_bytetrack_20260718_140346/`.
* Known Step 4 limitations: lifecycle management does not claim to fix ByteTrack fragmentation or ID switches; the real validation did not naturally produce same-ID recovery before expiry, so recovery is validated synthetically; no crop collection, OCR/color enrichment, plate detection, Florence, search indexing, event creation, ReID, appearance matching, track merging, speed analytics, queueing, or production integration was added. Step 5 should collect per-track observations/crop candidates from lifecycle-managed tracks only.
* Completed Step 5 for the isolated streaming tracking pipeline: added generation-aware `TrackObservation` collection, padded crop extraction, preliminary quality metrics, bounded crop candidate retention, completed crop bundles, Step 5 artifacts, and synthetic/real validation runners. Real validation processed 60 frames, emitted 22 observations, created 21 crop candidates, retained 13 candidates, and completed 6 crop bundles under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138/`.
* Completed Step 6 for the isolated streaming tracking pipeline: added `BestCropScoreConfig`, `BestCropSelectionConfig`, final score breakdowns, primary/fallback selected crop schemas, OCR-ready job descriptions without OCR execution, deterministic temporal/bbox diversity, explicit rejection reasons, selection metrics, artifact replay, optional previews, and bounded real validation. The Step 6 implementation remains isolated and does not run plate YOLO, Florence, OCR, colour detection, search, events, ReID, queues, threads, multiprocessing, or production integration.
* Validated Step 6 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 118 tests passed. `py_compile` passed for Step 6 modules and touched config.
* Synthetic Step 6 validation passed with 16 completed bundles, 12 primary crops, 9 fallback crops, 5 primary-selected tracks, 9 fallback-only tracks, 2 no-valid tracks, and disabled-selector coverage. Existing Step 5 artifact replay passed with 6 completed bundles, 3 primary crops, 6 fallback crops, 2 primary-selected tracks, 4 fallback-only tracks, 0 no-valid tracks, and 6 previews under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_existing_step5_20260718_144055/`. Bounded real Step 1-6 validation passed under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_evidence_video_ultralytics_bytetrack_20260718_144104/`.
* Known Step 6 limitations: final score is selection quality only and does not claim OCR readiness or OCR quality; plate visibility remains a zero-weight placeholder component until Step 7 actually detects plates; Step 7 should consume `SelectedTrackCropSet` / `SelectedCropJob` records and then add plate YOLO, Florence OCR, and vehicle colour detection sequentially.
* Completed Step 7 for the isolated streaming tracking pipeline: added Step 7 config caps, raw ANPR schemas, a crop-local Ultralytics plate detector, local-only Florence-2 inference with optional PEFT adapter, sequential crop-job orchestration, raw JSONL artifacts, metrics, validation runner, and fake-model unit tests. The implementation consumes Step 6 `SelectedCropJob` records and remains isolated from production code.
* Validated Step 7 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 130 tests passed. `py_compile` passed for all Step 7 modules and the Step 7 runner.
* Synthetic Step 7 validation passed with 1 processed track, 1 plate candidate, 1 raw OCR result, and 1 normalized vehicle colour under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step7_synthetic_20260718_150206/`. Existing Step 6 artifact replay passed structurally on 6 bounded identities and 9 selected crop jobs under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step7_existing_step6_20260718_150352/`; Florence loaded locally and colour inference ran without errors, plate YOLO ran on all selected vehicle crops, and the bounded crop set produced 0 plate candidates, so OCR attempts correctly remained 0.
* Completed Step 7.5 for the isolated streaming tracking pipeline: added plate diagnostic config, raw detector box diagnostics, constrained attempt/disposition statuses, crop-local threshold probing, vehicle-crop suitability metadata, annotated vehicle crop output, accepted/rejected diagnostic crop output, bounded primary/fallback retry, optional OCR-on-accepted-candidate policy, JSONL artifacts, metrics, validation runner, and fake-model tests.
* Validated Step 7.5 with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 139 tests passed. `py_compile` passed for Step 7.5 modules plus touched config and detector modules.
* Synthetic Step 7.5 validation passed with 15 synthetic tracks, 18 crop attempts, 15 raw detector boxes, 10 accepted plate candidates, 1 class rejection, 1 invalid-geometry rejection, 1 empty-after-clipping rejection, 2 size rejections, 1 below-normal-threshold diagnostic acceptance, 9 OCR calls, 8 non-empty OCR outputs, and 1 empty OCR output under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step75_synthetic_20260718_152215/`.
* Real Step 7.5 artifact replay passed on `debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_existing_step5_20260718_144055/`: 9 selected crop jobs, 9 plate detector calls, normal threshold `0.20`, diagnostic thresholds `0.25,0.15,0.10,0.05`, model classes `{0: License_Plate}`, 0 raw detector boxes, 0 boxes below normal threshold, 0 class/geometry/size rejections, 0 accepted candidates, 0 OCR calls, and 6 exhausted tracks under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step75_existing_step6_20260718_152224/`. The same detector produced 1 raw/accepted box at confidence `0.776766` on known existing plate crop `debug_runs/ANPR1-D_20260715_175538/06_plate_crops/vehicle_track_0001_frame_000030_combined_001_plate.jpg`, so the Step 6 replay issue is truly zero detector boxes on selected crops, not model load failure or downstream filtering.
* Added image-folder ANPR validation for local model checks without running video/tracking/lifecycle/crop-selection. The runner discovers supported images, runs raw plate YOLO diagnostics, saves annotations and accepted/rejected crops, optionally runs Florence OCR on accepted detector crops, optionally runs direct-input OCR, and writes manifest/JSONL/report artifacts. Validated with `tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"`: 146 tests passed, and `py_compile` passed for the two new image validation modules.
* Real image-folder validation passed on `debug_runs/test pictures`: 7 JPG images discovered/read, 13 raw detector boxes, 13 accepted plate candidates, 1 box below normal threshold, 0 rejected boxes by class/geometry/clipping/size, 13 accepted crops saved, 7 detector-crop OCR calls with 7 non-empty raw outputs, and 7 direct-input OCR calls with 7 non-empty raw outputs. Artifacts were written under `debug_runs/streaming_tracking_pipeline/image_anpr_validation_20260718_153928/`. Raw OCR remains unverified.
* Added bounded 10 FPS ANPR-video orchestration with an explicit vehicle-only crop eligibility gate, optional start/end offsets, automatic CUDA/CPU model-device selection, CUDA memory fallback reporting, peak VRAM reporting, and combined Step 5/6/7.5/OCR/colour reporting. Fixed Florence CUDA half-precision inference by casting floating processor inputs to the configured CUDA dtype. Validated on `C:\Mukul K\mk\test_video\anpr_test_5min.mp4` for 600 selected frames: 106 filtered detections, 20 raw tracker IDs, 23 completed vehicle track generations, 17 ANPR-eligible jobs, 12 raw/accepted plate boxes, 12 non-empty raw OCR outputs, and 17 colour calls under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/`.
* Completed Step 8 for the isolated streaming tracking pipeline: added artifact-only plate normalization, bounded OCR character correction, Indian plate-format validation, agreement scoring, final track-generation status selection, colour joins, metrics, JSONL/report artifacts, and tests. Validated on the `160910` bounded run with 12 raw OCR records, 12 normalized candidates, 2 corrected candidates, 3 verified results, 6 weak results, 3 invalid results, and 11 no-plate results under `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/08_plate_validation/`.
* Known Step 8 limitations: this isolated validation does not emit production object records, index search, create events, run ReID, merge tracker fragments, change crop selection, rerun models, add queues/threads/multiprocessing, or integrate with production APIs. Agreement remains unproven on the bounded run because each OCR-bearing track generation currently has only one OCR candidate.
* Completed Step 9 for the isolated streaming tracking pipeline: added artifact-only searchable vehicle record schemas, generation-aware record IDs, lifecycle/crop/ANPR/Step 8 joins, metrics, validation-query smoke checks, JSONL/flat-JSON/status-split artifacts, and tests. Validated on the full-video run `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/`: 167 records created, 58 verified plate records, 28 weak plate records, 14 invalid plate records, 67 no-plate records, no missing input artifacts, no join failures, and no duplicate record IDs under `09_searchable_objects/`.
* Known Step 9 limitations: this isolated export is not a production search index, vector embedding stage, event generator, ReID/fragment-merge stage, or API integration. It preserves weak plate text only as weak-search evidence and does not treat invalid raw OCR as searchable verified plate text.
* Completed Step 10 for the isolated streaming tracking pipeline: added deterministic query schemas/parser, in-memory structured search index, score-component ranker, search artifacts, metrics, validation runner, and tests. Validated on the full-video Step 9 artifact with 167 records indexed, 11 validation queries executed, zero expected-count mismatches, zero parser warnings, zero duplicate result IDs within query responses, and artifacts under `10_structured_search/`.
* Known Step 10 limitations: this is not an embedding search, FAISS/vector DB, API, UI, LLM query planner, production index, event generator, or result-card renderer. It does not merge track generations, repair identities, or treat invalid OCR as a valid plate.
* Completed Step 11 for the isolated streaming tracking pipeline: added UI-ready result-card schemas, deterministic card builder, artifact writer, metrics, validation runner, focused tests, and optional static HTML preview generation. Validated on the full-video Step 10 artifacts with 11 packaged queries, 138 cards, no duplicate cards within any query package, and artifacts under `11_result_cards/`.
* Known Step 11 limitations: this is not a production UI/API, queue/worker system, Step 07b integration, event/VLM integration, embedding search, vector index, or web framework. The HTML preview is static and local-file oriented only.
* Added focused object-class/colour correction infrastructure: `class_normalization.py`, `multi_model_detection.py`, `dominant_colour_analysis.py`, and `run_object_class_colour_validation.py`, plus tests. Search now parses `person`, `bicycle`, `scooter`, and `two wheeler`; structured search prefers `normalized_class_name` and validated `dominant_colour` when those fields are present. Current full-video artifacts still contain no person records because the discovered person checkpoint is unloadable.

## 2026-06-08

* Implemented Investigation Narrative Engine (Phase 1C) to replace statistical overviews with dynamic, deterministic narrative synthesis.
* Enriched events with intelligence fields: `scene_context`, `real_world_time`, `participants`, `behavioral_flags`, and `disposition`.
* Fixed Uvicorn file lock truncation bugs by routing output to `_events.json` and removing legacy fallbacks.
* Implemented semantic state-tracking algorithm in Event Abstraction Layer.
* Validated and completely resolved 1-second event fragmentation issue. Phase 1B complete.

## 2026-07-24

* Reorganized the multicamera vehicle-tracking pipeline's local model assets under the repository-root `models/` folder while keeping the runtime Python package under `tests/td_case2/multicamera_vehicle_tracking_pipeline/models/`.
* Added portable model-path defaults, a readiness-check workflow, and setup documentation so new-PC validation no longer depends on scattered legacy model folders or old machine-specific paths.
* Added a read-only `verify_enrichment_run.py` audit CLI for completed multicamera enrichment runs in the `analytics` schema, with JSON export, strict consistency checks, and mocked unit coverage for Supabase query behavior.
* Extended the same read-only verifier to support run, camera, and single-track drill-downs, including evidence diagnostics, missing-enrichment diagnostics, and plate-based cross-camera candidate hints without changing persistence or enrichment behavior.
* Added a first production-safe standalone cross-camera matching stage for the multicamera vehicle-tracking pipeline. The new `build_global_vehicle_objects.py` command performs deterministic post-run candidate generation, conservative scoring, auditable `cross_camera_match` persistence, and stable `global_vehicle` / `global_vehicle_track` creation without modifying camera-level `vehicle_track` rows.
* Added additive analytics-schema migrations `028_global_vehicle_matching_extensions.sql` and `029_global_vehicle_matching_indexes.sql`, plus updated `database/supabase/analytics_full_schema.sql`, so global objects now store `processing_run_id`, stable object codes, creation method, camera/track counts, and persisted match-to-object links.
* Added a read-only `verify_global_vehicle_objects.py` audit CLI plus focused unit coverage for global-match config, immutable models, scoring, service orchestration, repository idempotency, build CLI behavior, and verifier behavior.
* Added a first read-only FastAPI backend for the multicamera vehicle-tracking pipeline under `tests/td_case2/multicamera_vehicle_tracking_pipeline/api/`. It reuses the existing analytics client, adds a read-only query repository for the `analytics` schema, exposes safe JSON endpoints for runs, cameras, tracks, media references, cross-camera matches, and global vehicles, and strips sensitive metadata keys from responses.
* Added focused API tests covering health, pagination, filters, 404 behavior, media-reference safety, error masking, and OpenAPI credential hygiene. The API test slice passed on July 24, 2026 with 29 tests green; the full multicamera suite remained at the same two unrelated detection-class normalization failures (`automobile`, `3wheeler`) and introduced no new failures.

## 2026-07-25

* Added a read-only React + TypeScript + Vite frontend for `tests/td_case2/multicamera_vehicle_tracking_pipeline` under `tests/td_case2/multicamera_vehicle_tracking_pipeline/frontend`.
* The frontend is wired to the existing FastAPI backend only and does not call Supabase directly or expose service-role credentials in source.
* Added typed API modules for runs, tracks, global vehicles, cross-camera matches, and media references, plus reusable layout, state, table, filter, and evidence components.
* Added route-based screens for dashboard, runs, run detail, tracks, track detail, global vehicles, global vehicle detail, cross-camera matches, and 404 handling.
* Added focused frontend coverage for API base URL handling, paginated response parsing, backend error parsing, run/track/global-vehicle/match rendering, reference-only media behavior, loading and empty states, and source checks that block direct Supabase usage.
* Frontend validation passed locally on July 25, 2026 with `15` Vitest tests green and a successful production build from `tests/td_case2/multicamera_vehicle_tracking_pipeline/frontend`.
* Updated the multicamera camera-config loader to accept both legacy flat camera entries (`camera_name`, `source_path`) and the newer nested source shape used by the worker validation config (`source.path`). Missing `camera_name` now falls back to `camera_code`, preserving downstream metadata requirements without forcing duplicate naming in YAML.
* Fixed worker-pipeline file-source timestamp propagation for multicamera analytics persistence. File cameras without an explicit `start_time` now inherit one timezone-aware run-level source timestamp, and the same anchor is reused for frame packets, completed-track `first_seen_at` / `last_seen_at`, and analytics `video_source.source_start_at`.
* Added focused regression coverage for the run-level timestamp override in analytics persistence and for worker completed tracks from file camera sources without configured start times.
* Revalidated the two-camera 100-frame worker pipeline in dry-run mode on July 25, 2026: `RUN_20260725_131449` processed 200 frames, 314 detections, 286 observations, and 8 completed tracks. All 8 tracks reached analytics persistence dry-run validation, 286 observations validated, 8 media records validated, vehicle-colour worker processed 8 jobs, and ANPR worker processed 8 jobs with 2 verified plates.
* Added safe evidence image delivery for the multicamera FastAPI backend. The API now resolves approved local `track_media` paths under project evidence roots, serves browser-safe metadata through `GET /api/v1/media/{media_id}`, streams local image files through `GET /api/v1/media/{media_id}/content`, and preserves placeholder-safe availability states for missing, reference-only, unsafe, and unsupported media.
* Updated track and global-vehicle responses so the React frontend can render vehicle and plate crops without exposing absolute filesystem paths or any Supabase service-role credentials.
* Added focused backend coverage for media root parsing, safe path resolution, traversal rejection, symlink escape rejection, MIME handling, CORS on media content, and `FileResponse` delivery, bringing the focused API slice to 27 passing tests.
* Added frontend evidence rendering coverage for live image cards, placeholder fallback, and global-vehicle evidence rendering. On July 27, 2026 the frontend suite passed with 22 Vitest tests green and the production build succeeded.
* Live validation on July 27, 2026 confirmed that `RUN_20260725_131944:CAM_002:TRACK_4` serves `BEST_VEHICLE_CROP` media `bd043d85-8be5-464e-96e9-108f19499f87` and `PLATE_CROP` media `8971f0a1-28e4-47aa-9a2f-487b92a52753` as `LOCAL_FILE` with `200 image/jpeg` responses and `Access-Control-Allow-Origin: http://127.0.0.1:5173`. The confirmed global vehicle `GVO:RUN_20260725_131944:FA3FCF9E3ABC` rendered evidence from both `CAM_001 TRACK_4` and `CAM_002 TRACK_4` in the live React UI.
* Completed the follow-up Track Detail membership fix on July 27, 2026. The API now maps persisted `global_vehicle_track` memberships into a structured `global_membership` object that includes `linked`, `global_vehicle_id`, `global_vehicle_code`, confidence, status, and member-track count, and the Track Detail page now renders the linked global vehicle code plus `Open Global Vehicle` instead of incorrectly showing `Not linked`.
* Standardized multicamera frontend evidence cards on July 27, 2026 to use a shared `240px` preview viewport with `object-fit: contain` and modal preview behavior so vehicle crops and narrow plate crops remain fully visible with consistent card sizing.
* Final live UI validation on July 27, 2026 confirmed that `RUN_20260725_131944:CAM_002:TRACK_4` shows `GVO:RUN_20260725_131944:FA3FCF9E3ABC` in Track Detail, and the global vehicle page for `GVO:RUN_20260725_131944:FA3FCF9E3ABC` renders four consistent evidence cards across `CAM_001` and `CAM_002` vehicle and plate crops.

## 2026-07-27

* Added read-only structured vehicle search for the multicamera vehicle-tracking pipeline. The FastAPI backend now exposes `GET /api/v1/search/vehicles` with validated filters for run code, scope, class, colour, normalized plate text, camera selection, date/time overlap, confidence, multi-camera-only, verified-plate-only, pagination, and deterministic sorting.
* Added backend search orchestration that reuses the existing analytics read repository and media decoration path instead of exposing raw Supabase rows or direct browser access to Supabase.
* Added the React `/search` route, sidebar navigation entry, URL-backed filter form, structured result cards, and detail-route drill-down for both local tracks and global vehicles.
* Added focused frontend coverage for the new search page and result cards. On Monday, July 27, 2026, the frontend suite passed with `37` Vitest tests green and the production build succeeded.
* Added repository-level backend search coverage runnable under the local test environment. The API-level and service-level backend search tests were authored, but full execution remains blocked in the current local Python environment because the available interpreters do not currently provide `fastapi` and `pydantic`.
* Added read-only natural-language vehicle search on top of the existing structured search service. The FastAPI backend now exposes `POST /api/v1/search/natural-language` and `POST /api/v1/search/natural-language/parse`, validates parsed intent through Pydantic, applies explicit run/scope context safely, validates camera codes against the selected run, and reuses the existing `VehicleSearchService` instead of generating SQL or allowing direct LLM/database access.
* Added a deterministic natural-language fallback parser for common plate, class, colour, camera, multi-camera, verified-plate, and time phrases so basic operator queries still work when the configured provider is unavailable or returns invalid JSON.
* Updated the React `/search` page to include a natural-language search bar, interpreted-filters panel, fallback/provider diagnostics, clarification handling, and `Apply to filters` handoff back into the existing structured search form while continuing to reuse the same result cards and detail routes.
* On Monday, July 27, 2026, focused multicamera backend search tests passed with `30` pytest tests green, the frontend suite passed with `45` Vitest tests green, and the frontend production build succeeded after the natural-language search layer was added.
* Standardized multicamera vehicle evidence presentation around a reusable grouped `VehicleIdentityCard`. Track detail, global vehicle detail, cross-camera matches, search results, run-detail track/global tabs, and list summaries now keep each representative vehicle crop and its plate crop together instead of rendering flat unrelated evidence cards.
* Added lightweight `primary_vehicle_media` and `primary_plate_media` fields to the read-only API contracts used by tracks, global vehicles, search results, and match summaries, allowing thumbnail-rich list pages without fetching every stored media record.
* On Monday, July 27, 2026, focused grouped-evidence validation passed with `11` backend pytest tests green for the new read-side media/search contract plus `45` frontend Vitest tests green and a successful frontend production build.
* Enhanced multicamera ANPR recall for small and edge-positioned plates without weakening the verified `DL8CBF6268` regression case. The pipeline now evaluates multiple saved evidence roles, runs plate detection on original plus padded vehicle-crop variants, falls back to class-aware heuristic regions for classes such as `3WHEELER`, and aggregates OCR evidence across bounded preprocessing variants before final classification.
* Added a richer dry-run ANPR validator for existing artifact runs. `validate_anpr_on_existing_run.py` now accepts `--run-code`, `--track-uuid`, `--persist`, and dry-run reporting, emits per-attempt OCR diagnostics, and keeps analytics persistence backward compatible by mapping runtime `VERIFIED` / `PARTIAL` / `UNREADABLE` / `CONFLICTING_CANDIDATES` results into the existing persisted plate-reading schema only when required.
* On Monday, July 27, 2026, focused enhanced-ANPR tests passed with `18` pytest tests green. Dry-run validation on `RUN_20260725_131944:CAM_001:TRACK_4` and `RUN_20260725_131944:CAM_002:TRACK_4` still returned verified `DL8CBF6268`, while the previously failed `RUN_20260727_112517:CAM_003:TRACK_2` no longer stopped at `NO_PLATE_DETECTED` and now returns a classified heuristic-result report with saved OCR diagnostics under `debug_runs/multicamera_vehicle_tracking_pipeline/anpr_track2_validation.json`.
* Added worker-pipeline YOLO debug artifact export for multicamera validation runs. When `--save-sample-frames` is enabled, the tracking worker now writes per-camera `yolo_detections/sample_*.jpg` annotated detector previews, matching `sample_*.json` bbox payloads, and `tracking_samples/sample_*.jpg` tracking previews under the selected run output directory. Focused worker tests passed on Monday, July 27, 2026, and a three-camera dry-run validation on `1test.mp4`, `2test.mp4`, and `3test .mp4` completed as `RUN_20260727_131724` with 768 processed frames, 1176 detections, 24 completed tracks, and YOLO debug artifacts saved under `debug_runs/multicamera_vehicle_tracking_pipeline/user_three_videos_yolo_debug/`.
* Added track-level class stabilization for the multicamera vehicle-tracking pipeline. The lifecycle now keeps weighted per-class evidence, preserves raw frame labels for diagnostics, persists stabilized class diagnostics into track metadata, and exposes stable class/confidence fields through the read-only API and React Track Detail page instead of letting the latest YOLO label overwrite the saved track class.
* Added a dry-run-first `recalculate_track_classes.py` repair/audit CLI for existing runs. The script recalculates classes from persisted observation metadata when available, can optionally persist updated final classes back to `analytics.vehicle_track`, and explicitly reports insufficient history for older dry-run artifact reports such as `RUN_20260727_131724` instead of guessing.
* Revalidated the same three-camera worker pipeline after the stabilization change on Monday, July 27, 2026. The new dry-run worker validation completed as `RUN_20260727_135002` with the same 768 processed frames, 1176 detections, 1089 track observations, and 24 completed tracks, while `CAM_001 TRACK_2` now persists as stabilized class `3wheeler` in the run report under `debug_runs/multicamera_vehicle_tracking_pipeline/user_three_videos_yolo_debug_stabilized/`.
* Fixed grouped multicamera vehicle cards on Monday, July 27, 2026 so `PLATE_CROP`-style media can never be promoted to `primary_vehicle_media`. The read-only repository now selects vehicle and plate media from explicit role-specific type lists (`BEST_VEHICLE_CROP`, `BEST_OVERALL`, `VEHICLE_CROP`, `TRACK_CROP` vs `PLATE_CROP`, `NUMBER_PLATE_CROP`, `ANPR_CROP`, `OCR_CROP`) and keeps the frontend defensive with the same rule.
* Redesigned the shared `VehicleIdentityCard` and list containers on Monday, July 27, 2026 so search, run detail, track detail, global vehicle detail, and cross-camera match views render a wide grouped card with vehicle image, plate image, plate text/status, readable timestamps, and detail actions in one layout instead of a narrow card plus separate plate-evidence block.
* Added focused regression coverage on Monday, July 27, 2026 for media-role separation, grouped card rendering, semantic partial-plate formatting, and wide search results. Validation passed with `5` focused backend pytest tests green, `31` focused frontend Vitest tests green, and a successful frontend production build.
* Simplified the multicamera frontend vehicle cards again on Monday, July 27, 2026 to remove per-field metadata tiles and nested card chrome. The shared card now uses one outer card, horizontal grouped media, a lightweight definition-list metadata block, smaller secondary identifiers, and inline search relevance instead of extra metric boxes.
* Updated the run-detail, track-detail, global-vehicle-detail, search, and cross-camera match presentations on Monday, July 27, 2026 to keep one wide result per row by default, avoid overflowing long IDs in headings, and reduce repeated borders and badges while preserving grouped vehicle/plate evidence and detail navigation.
* Focused validation for the simplified card redesign passed on Monday, July 27, 2026 with `32` frontend Vitest tests green and a successful frontend production build.
* Fixed the persisted multicamera plate-read contract on Monday, July 27, 2026 so the read-only API now joins `plate_summary.selected_plate_reading_id` to the saved `plate_reading` and `plate_detection` rows, publishes a canonical `plate_result` object for tracks, member tracks, global vehicles, matches, and vehicle-search results, and keeps the legacy `canonical_plate` / `plate_status` fields backward compatible. This closed the live UI gap for `RUN_20260727_142313:CAM_002:TRACK_4`, where the database stored plate media plus selected OCR text `DL8CBF6268` but `plate_summary.canonical_plate` was null.
* Focused validation for the persisted plate-result fix passed on Monday, July 27, 2026 with `16` backend pytest tests green, `12` focused frontend Vitest tests green, and a successful frontend production build.
* Fixed two multicamera validation reliability issues on Monday, July 27, 2026. The read-only `verify_enrichment_run.py` audit no longer compares one run's plate detections against every `plate_reading` row in the whole database, so strict verification no longer reports false orphan plate readings for unrelated historical runs. The worker `PersistenceWorker` now applies backpressure when ANPR or vehicle-colour queues are temporarily full instead of failing the run with a `persistence_worker` pipeline error. Focused regression coverage passed with `48` pytest tests green across `test_verify_enrichment_run.py` and `test_persistence_worker.py`. A full three-camera dry-run rerun completed as `RUN_20260727_163614` with `errors: []` in the worker report and no persistence-worker failure.
* Fixed the multicamera frontend runtime-status failure path on Monday, July 27, 2026. The FastAPI analytics health check now degrades to a successful `status=degraded` response when Supabase is temporarily unreachable instead of hard-failing the whole API surface, and the React dashboard now continues rendering recent runs when the health probe is warning or unavailable. Focused validation passed with `5` pytest tests green for `test_api_health.py` and a successful production frontend build.
* Hardened multicamera single-camera identity continuity and evidence ranking on Monday, July 27, 2026. The lifecycle now splits a logical track when a reused ByteTrack ID shows a strong spatial/class/scale identity break instead of merging two physical vehicles into one saved track, while fragment-link recovery still preserves the original logical track for same-object reappearances under a new tracker ID. The evidence collector also normalizes sharpness before scoring and now weights visibility/centeredness so `best_overall` prefers a fuller in-frame vehicle crop instead of a clipped but numerically sharper crop. Focused validation passed with `15` pytest tests green across `test_track_lifecycle.py` and `test_track_evidence_collector.py`. A three-camera dry-run rerun completed as `RUN_20260727_170538` with `errors: []`; in that run `CAM_002:TRACK_8` now resolves to the blue truck plate `DL1LAA1556` instead of the earlier mixed truck/car artifact chain. A fully persisted rerun remains blocked in the current session by local socket/Supabase access failure `WinError 10013`.

## 2026-06-04

* Added Event Catalog architecture
* Fixed Summary API event loading
* Added schema alignment
* Added event catalog persistence
* Implemented dynamic event catalog fallback and auto-rebuilding from individual event files
* Implemented dynamic on-the-fly event generation fallback from raw frame metadata if events are missing
* Updated roadmap priorities

---

# Agent Reminder

Before making changes:

1. Read PROJECT_CONTEXT.md
2. Read ROADMAP.md
3. Read this file

Important Rules:

* Events are the source of truth.
* Do not build features directly on frame metadata.
* Preserve event-centric architecture.
* Search and investigation capabilities are the primary business goal.
* Validate each phase before moving to the next.
