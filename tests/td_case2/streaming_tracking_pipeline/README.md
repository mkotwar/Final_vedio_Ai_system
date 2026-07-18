# td_case2 Streaming Tracking Pipeline Foundation

This package is an isolated foundation for a future streaming, tracking-first `td_case2` experiment. It does not modify or replace the current production-oriented `tests/td_case2` pipeline.

## Target architecture

```text
Video / RTSP
  -> Continuous Frame Reader
  -> YOLO Detection on Sequential Frames
  -> ByteTrack
  -> Track Lifecycle Manager
  -> Per-Track Crop Collector
  -> Best-Crop Selector
  -> Plate YOLO
  -> Florence-2 OCR / Vehicle Colour
  -> Validation and Retry
  -> Structured Object Record
  -> Search Index
  -> Event / VLM Pipeline
```

ByteTrack must remain sequential per camera. Future multi-camera work should run one ordered tracker stream per camera and correlate completed object records later.

## What Step 1 implements

Step 1 creates the package structure, shared dataclass schemas, configuration objects, JSON-safe serialization helpers, reusable validators, and offline tests.

Schemas:

- `BoundingBox`: XYXY geometry, clipping, metrics, edge-touch checks.
- `FramePacket`: runtime frame container. `frame` is runtime-only.
- `DetectionRecord` and `DetectionPacket`: detector outputs.
- `TrackedObject` and `TrackedFramePacket`: tracker outputs.
- `TrackStatus` and `TrackCompletionReason`: constrained lifecycle enums.
- `CropQualityMetrics` and `CropCandidate`: crop references and bounded quality inputs.
- `TrackRecord`: aggregate track lifecycle state with deterministic dominant class.
- `PlateResult` and `ColourResult`: constrained enrichment result containers.
- `ObjectRecord`: future search-index-ready object record.

Configuration groups:

- `SourceConfig`
- `DetectionConfig`
- `TrackingConfig`
- `CropSelectionConfig`
- `CropCollectionConfig`
- `BestCropScoreConfig`
- `BestCropSelectionConfig`
- `PlateDetectionConfig`
- `FlorenceConfig`
- `RetryConfig`
- `QueueConfig`
- `OutputConfig`
- `PipelineConfig`

Configuration precedence:

```text
defaults
  -> JSON configuration file
  -> TD_CASE2_STREAM_* environment variables
  -> explicit runner overrides in future steps
```

Supported Step 1 environment variables include:

- `TD_CASE2_STREAM_SOURCE_PATH`
- `TD_CASE2_STREAM_SOURCE_ID`
- `TD_CASE2_STREAM_TARGET_PROCESSING_FPS`
- `TD_CASE2_STREAM_USE_SOURCE_FPS`
- `TD_CASE2_STREAM_RTSP_TRANSPORT`
- `TD_CASE2_STREAM_MAX_PROCESSED_FRAMES`
- `TD_CASE2_STREAM_DETECTOR_MODEL_PATH`
- `TD_CASE2_STREAM_DETECTION_CONFIDENCE`
- `TD_CASE2_STREAM_DETECTION_IOU`
- `TD_CASE2_STREAM_DETECTION_DEVICE`
- `TD_CASE2_STREAM_DETECTION_IMAGE_SIZE`
- `TD_CASE2_STREAM_TRACKING_BACKEND`
- `TD_CASE2_STREAM_TRACKING_BUFFER`
- `TD_CASE2_STREAM_TRACKING_ACTIVATION_THRESHOLD`
- `TD_CASE2_STREAM_TRACKING_MATCH_THRESHOLD`
- `TD_CASE2_STREAM_CROP_COLLECTION_ENABLED`
- `TD_CASE2_STREAM_CROP_SAVE_IMAGES`
- `TD_CASE2_STREAM_CROP_MAX_CANDIDATES`
- `TD_CASE2_STREAM_CROP_MAX_OBSERVATIONS`
- `TD_CASE2_STREAM_CROP_RETENTION_POLICY`
- `TD_CASE2_STREAM_CROP_PADDING_RATIO`
- `TD_CASE2_STREAM_CROP_MIN_WIDTH`
- `TD_CASE2_STREAM_CROP_MIN_HEIGHT`
- `TD_CASE2_STREAM_CROP_MIN_AREA_RATIO`
- `TD_CASE2_STREAM_CROP_MAX_AREA_RATIO`
- `TD_CASE2_STREAM_SELECTION_PRIMARY_COUNT`
- `TD_CASE2_STREAM_SELECTION_MIN_PRIMARY_SCORE`
- `TD_CASE2_STREAM_SELECTION_MIN_FALLBACK_SCORE`
- `TD_CASE2_STREAM_SELECTION_MIN_FRAME_SEPARATION`
- `TD_CASE2_STREAM_SELECTION_MIN_TIME_SEPARATION`
- `TD_CASE2_STREAM_SELECTION_KEEP_FALLBACK`
- `TD_CASE2_STREAM_SELECTION_PRIMARY_POLICY`
- `TD_CASE2_STREAM_SELECTION_FALLBACK_POLICY`
- `TD_CASE2_STREAM_PLATE_MODEL_PATH`
- `TD_CASE2_STREAM_PLATE_ENABLED`
- `TD_CASE2_STREAM_FLORENCE_MODEL_PATH`
- `TD_CASE2_STREAM_FLORENCE_ADAPTER_PATH`
- `TD_CASE2_STREAM_FLORENCE_ENABLED`
- `TD_CASE2_STREAM_RETRY_ENABLED`
- `TD_CASE2_STREAM_OUTPUT_ROOT`
- `TD_CASE2_STREAM_SAVE_ANNOTATED_VIDEO`
- `TD_CASE2_STREAM_ANNOTATED_VIDEO_FPS`

## What Step 1 does not implement

This package does not decode video, read RTSP streams, run YOLO, instantiate ByteTrack, load Florence, detect plates, score crops, create threads, create queues, detect events, build search indexes, or integrate with production pipeline stages.

Raw frames are runtime-only and are excluded from JSON by default. The serialization helpers intentionally reject unsupported runtime objects so arrays or model outputs are not written accidentally.

## What Step 2 implements

Step 2 adds the sequential source and stage-contract layer needed before real YOLO and ByteTrack are connected.

New modules:

- `contracts.py`: `FrameSource`, `DetectionStage`, `TrackingStage`, `PacketSink`, and metadata-preservation validators.
- `sources.py`: `SyntheticFrameSource` and `build_processing_frame_indices`.
- `mock_stages.py`: deterministic model-free detection and tracking stages.
- `adapters.py`: tracker-ID normalization, current `td_case2` Step 03/04B compatibility adapters, in-memory sink, and JSONL sink.
- `sequential_pipeline.py`: one-frame-at-a-time contract pipeline and JSON-safe report.
- `run_step2_contract_validation.py`: CPU-only validation runner that writes structured JSONL artifacts.

## Source lifecycle

`SyntheticFrameSource` follows:

```text
created -> opened -> reading -> end of stream -> closed
```

Reading before `open()` raises. End of stream returns `None` repeatedly. `close()` is idempotent. Reading after close raises. `reset()` clears the cursor and closed/opened state so deterministic sources restart from frame zero.

## Frame-selection policy

If `target_processing_fps` is `None` or `use_source_fps=True`, every source frame is emitted.

If target FPS is lower than source FPS, selected frame indices are chosen by exact timestamp scheduling using `fractions.Fraction`, not rounded integer intervals. The first frame is included for non-empty sources. The last source frame is not forced, because forcing it can break timing consistency. Target FPS above source FPS is rejected; Step 2 does not upsample synthetic sources.

Example:

```text
source_fps=30, target_processing_fps=7, total_frames=30
selected indices: 0, 5, 9, 13, 18, 22, 26
```

## Timestamp policy

Each `FramePacket.timestamp_sec` is calculated from the original source frame index:

```python
timestamp_sec = frame_index / source_fps
```

Timestamps never use processed-frame count.

## Tracker-ID normalization policy

`track_id` is the normalized integer identifier used internally by the streaming pipeline.

`source_track_id` preserves the original identifier from an existing tracker or `td_case2` artifact.

Non-negative integer source IDs keep their value. String IDs, including numeric strings such as `"1"`, are treated as external IDs and receive deterministic first-seen integer IDs. Allocation skips existing native integer IDs to prevent collisions. `reset()` clears all mappings. No hash-based IDs are used.

## Tracker configuration note

The config can represent both older Supervision ByteTrack-style names (`track_activation_threshold`, `lost_track_buffer`, `minimum_matching_threshold`, `minimum_consecutive_frames`) and current Ultralytics ByteTrack-style names (`track_high_threshold`, `track_low_threshold`, `new_track_threshold`, `match_threshold`, `fuse_score`). These similarly named thresholds may not map one-to-one between implementations.

## Compatibility notes

The current `td_case2` artifacts use string track IDs such as `vehicle_track_0001`; this foundation now preserves them in `source_track_id` while using normalized integer `track_id` internally.

Proposed mapping:

```text
new TrackedObject.track_id
  -> normalized integer from existing tracker_id / track_id

new TrackedObject.source_track_id
  -> existing Step 04B track_id, such as vehicle_track_0001

new CropCandidate.vehicle_crop_path
  -> Step 03 detection crop_path or Step 05 selected_crop_path

new CropCandidate.full_frame_path
  -> Step 03 image_path or Step 05 selected_full_frame_path

new CropQualityMetrics.bbox_area_ratio
  -> existing bbox_area_ratio

new ObjectRecord.object_class
  -> Step 04B dominant_class_name / Step 07B class_name

new ObjectRecord.plate.normalized_text
  -> Step 06 best_license_plate_text or verified_license_plate

new ObjectRecord.plate.verified
  -> Step 06 best_license_plate_valid / verified_license_plate_valid

new ObjectRecord.colour.normalized_colour
  -> Step 06 best_vehicle_color or Step 07B verified_vehicle_color
```

Actual Step 03 mapping:

```text
frame_group.frame_idx -> DetectionPacket.frame_index
frame_group.timestamp_seconds -> DetectionPacket.timestamp_sec
provided source_id -> DetectionPacket.source_id
provided frame_width/frame_height -> DetectionPacket dimensions
detection.bbox_xyxy -> DetectionRecord.bbox
detection.confidence -> DetectionRecord.confidence
detection.class_id -> DetectionRecord.class_id
detection.class_name -> DetectionRecord.class_name
```

Actual Step 04B mapping:

```text
track/detection track_id -> TrackedObject.source_track_id
TrackIdNormalizer.normalize(track_id) -> TrackedObject.track_id
detection.frame_idx -> TrackedObject.frame_index
detection.timestamp_seconds -> TrackedObject.timestamp_sec
detection.bbox_xyxy -> TrackedObject.bbox
detection.confidence -> TrackedObject.confidence
detection.class_id, when present -> TrackedObject.class_id
detection.class_name -> TrackedObject.class_name
```

The reverse helper emits a Step 05-shaped detection dictionary only. It does not call or modify Step 05.

## Sequential execution guarantee

`SequentialContractPipeline` uses this exact order:

```text
read one selected frame
  -> write optional frame artifact
  -> process detection
  -> write optional detection artifact
  -> process tracking
  -> write optional tracked-frame artifact
  -> read next selected frame
```

It does not use threads, multiprocessing, queues, or preloading all frames before processing.

## What Step 3 implements

Step 3 connects the isolated contracts to a real sequential reference path:

```text
Real video file
  -> OpenCvVideoSource
  -> Ultralytics YOLO detection stage
  -> ByteTrack stage
  -> TrackedFramePacket
  -> JSONL/report artifacts
```

New modules:

- `video_source.py`: `OpenCvVideoSource` with lazy OpenCV import, metadata validation, source-frame timestamps, target-FPS frame selection, and max processed-frame limits.
- `yolo_stage.py`: `UltralyticsYoloDetectionStage` with lazy YOLO model loading, class filtering, frame-bound box clipping, invalid-box rejection, and per-run detector metrics.
- `bytetrack_stage.py`: one ByteTrack instance per source stream, with both `ultralytics_bytetrack` and `supervision_bytetrack` adapters. The Ultralytics backend installs a local LAP shim so it does not attempt package installation at runtime.
- `tracking_metrics.py`: descriptive track-observation metrics, short-track counts, gap counts, and optional expected-object comparison.
- `real_pipeline.py`: strict one-frame-at-a-time real pipeline plus Step 3 artifact sink.
- `run_step3_real_tracking_validation.py`: real validation CLI with single-backend and backend-comparison modes.

The Step 3 runner remains deliberately sequential. It does not create threads, multiprocessing workers, queues, lifecycle managers, crop collectors, OCR calls, plate detectors, Florence calls, search indexes, events, or production pipeline hooks.

## Step 3 artifact layout

Each successful run writes:

```text
debug_runs/streaming_tracking_pipeline/step3_real_<video>_<backend>_<timestamp>/
|-- 01_source/
|   `-- source_metadata.json
|-- 02_detections/
|   `-- detection_packets.jsonl
|-- 03_tracks/
|   |-- tracked_frame_packets.jsonl
|   |-- track_observations.jsonl
|   `-- track_summary.json
|-- 04_visualization/
|   `-- tracked_video.mp4
`-- reports/
    `-- step3_real_tracking_report.json
```

`04_visualization/tracked_video.mp4` is written only when `--save-annotated-video` is passed.

`--tracking-backend both` writes sibling backend folders and a parent `reports/backend_comparison_report.json`. Backend failures are recorded explicitly rather than treated as silent skips.

## Step 3 validation notes

Validated commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\video_source.py tests\td_case2\streaming_tracking_pipeline\yolo_stage.py tests\td_case2\streaming_tracking_pipeline\bytetrack_stage.py tests\td_case2\streaming_tracking_pipeline\tracking_metrics.py tests\td_case2\streaming_tracking_pipeline\real_pipeline.py tests\td_case2\streaming_tracking_pipeline\run_step3_real_tracking_validation.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step3_real_tracking_validation --help
```

Real backend comparison on `data/videos/00000000-0000-0000-0000-000000000001.mp4`:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step3_real_tracking_validation --video data\videos\00000000-0000-0000-0000-000000000001.mp4 --detector-model yolo11n.pt --tracking-backend both --target-fps 5 --max-processed-frames 15 --device cpu --output-root debug_runs\streaming_tracking_pipeline
```

Result:

- `ultralytics_bytetrack` passed the real sequential path.
- `supervision_bytetrack` failed clearly because `supervision` is not installed in the current environment.
- The small `data/videos` sample produced zero detections, so it validates plumbing and artifact writing, not tracking quality.

Configured vehicle-model smoke validation:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step3_real_tracking_validation --video "debug_runs\test_anpr_day_10min_20260715_155615\10C_search_event_clips\obj_track_vehicle_track_0001_event_preview.mp4" --detector-model object\vehical_detection\best_old.pt --tracking-backend ultralytics_bytetrack --target-fps 2 --max-processed-frames 4 --device cpu --output-root debug_runs\streaming_tracking_pipeline --save-annotated-video
```

Result:

- The configured model loaded and the pipeline completed.
- The 2 FPS / 4-frame preview clip emitted zero detections.

Real tracked-object validation:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step3_real_tracking_validation --video "debug_runs\vidssave.com Woman crashes into lamppost, flips car during driving test 720P_20260716_112408\evidence_video.mp4" --detector-model yolo11n.pt --tracking-backend ultralytics_bytetrack --target-fps 4 --max-processed-frames 60 --confidence 0.05 --track-high-threshold 0.05 --track-low-threshold 0.01 --new-track-threshold 0.05 --device cpu --output-root debug_runs\streaming_tracking_pipeline --save-annotated-video
```

Result:

- Processed 60 selected frames from a 4 FPS real video.
- YOLO made 60 model calls and emitted 43 filtered detections across 23 frames.
- Ultralytics ByteTrack emitted 22 track observations across 2 unique track IDs.
- Tracking summary recorded one long track, one short track, 25 observation gaps for track 1, and `gap_note` stating that gaps are not automatic ReID success.
- Artifacts were written under `debug_runs/streaming_tracking_pipeline/step3_real_evidence_video_ultralytics_bytetrack_20260718_134920/`.

## Planned future stages

## What Step 4 Implements

Step 4 adds an application-level track lifecycle layer around sequential `TrackedFramePacket` objects:

```text
TrackedFramePacket
  -> TrackLifecycleManager
  -> lifecycle events
  -> active TrackRecord snapshots
  -> completed TrackRecord outputs
  -> lifecycle summary/report
```

This layer is not ByteTrack internals and does not alter tracker IDs. It decides how the application accepts, holds, expires, recovers, completes, and flushes tracker-generated IDs.

New modules:

- `lifecycle.py`: deterministic `TrackLifecycleManager`, private runtime state, `LifecycleUpdateResult`, transition events, same-ID recovery, and completed-ID generation handling.
- `lifecycle_metrics.py`: lifecycle metrics such as created/confirmed/completed counts, recovery attempts, lost-buffer expiry, short tracks, generation count, and completion reasons.
- `lifecycle_pipeline.py`: strict sequential source -> detection -> tracking -> lifecycle pipeline and Step 4 artifact sink.
- `run_step4_lifecycle_validation.py`: synthetic and real lifecycle validation runner.

State transitions:

```text
new ID -> tentative
tentative + enough observations -> confirmed
tentative + missed_processed_frames > maximum_tentative_missed_frames -> completed / invalid_track
confirmed + absent from a processed frame -> temporarily_lost
temporarily_lost + same track_id returns before expiry -> confirmed / recovered
temporarily_lost + missed_processed_frames > maximum_lost_processed_frames -> completed / lost_buffer_expired
temporarily_lost + elapsed seconds > maximum_lost_seconds -> completed / lost_buffer_expired
active track at end of stream -> completed / video_ended
completed track_id appears again -> new track_generation
```

The equality policy is conservative and consistent: expiry happens only when a configured threshold is exceeded (`>`), not when it is exactly equal.

Missing-count policy:

`missed_processed_frames` counts absent processed packets, not raw source-frame index gaps. If selected source frames are `0, 5, 9, 13` and a track is seen at `5` then absent at `9`, the missed count is `1`.

Recovery policy:

Recovery is only same-ID recovery. It is not ReID, appearance matching, fragment merging, or physical-object identity repair.

Generation policy:

`TrackRecord.track_generation` defaults to `0`. If a completed `track_id` appears again, the manager creates generation `1`, then `2`, and so on. Public records preserve both `track_id` and `source_track_id`.

Class-vote policy:

Class votes are counted per observation. Dominant class uses the existing deterministic `TrackRecord` policy: highest vote wins, ties sort by class name. A `class_updated` event is emitted only when the dominant class changes.

Flush behavior:

At end of video, `tracking_stage.flush()` runs before `lifecycle_manager.flush(reason=video_ended)`. Flush emits `completed` and `flushed` lifecycle events for remaining active tracks.

Step 4 artifact layout:

```text
debug_runs/streaming_tracking_pipeline/streaming_tracking_step4_<video>_<backend>_<timestamp>/
|-- 01_source/
|   `-- source_metadata.json
|-- 02_detections/
|   `-- detection_packets.jsonl
|-- 03_tracks/
|   `-- tracked_frame_packets.jsonl
|-- 04_lifecycle/
|   |-- lifecycle_events.jsonl
|   |-- active_track_snapshots.jsonl
|   |-- completed_tracks.jsonl
|   `-- lifecycle_summary.json
`-- reports/
    `-- step4_lifecycle_report.json
```

Validated Step 4 commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\lifecycle.py tests\td_case2\streaming_tracking_pipeline\lifecycle_metrics.py tests\td_case2\streaming_tracking_pipeline\lifecycle_pipeline.py tests\td_case2\streaming_tracking_pipeline\run_step4_lifecycle_validation.py tests\td_case2\streaming_tracking_pipeline\config.py tests\td_case2\streaming_tracking_pipeline\schemas.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step4_lifecycle_validation --mode synthetic_lifecycle
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step4_lifecycle_validation --mode real_tracking_lifecycle --video "debug_runs\vidssave.com Woman crashes into lamppost, flips car during driving test 720P_20260716_112408\evidence_video.mp4" --detector-model yolo11n.pt --tracking-backend ultralytics_bytetrack --target-fps 4 --max-processed-frames 60 --confidence 0.05 --track-high-threshold 0.05 --track-low-threshold 0.01 --new-track-threshold 0.05 --device cpu --output-root debug_runs\streaming_tracking_pipeline
```

Synthetic validation covered tentative confirmation, confirmed lost, same-ID recovery, lost expiry, tentative invalid completion, EOS flush, completed-ID generation increment, and class-vote changes.

Real validation on the Step 3 evidence video processed 60 frames, saw 2 raw tracker IDs, created 6 lifecycle generations, confirmed 2 tracks, emitted 2 temporarily-lost transitions, emitted 0 recoveries, and completed 6 tracks with reasons `invalid_track=3`, `lost_buffer_expired=2`, and `video_ended=1`. The real run did not naturally produce same-ID recovery before expiry.

## What Step 5 Implements

Step 5 adds deterministic per-track observation and crop-candidate collection on top of Step 4 lifecycle records:

```text
TrackedFramePacket
  -> TrackLifecycleManager
  -> TrackObservationCollector
  -> CropCandidateCollector
  -> completed crop bundles
  -> 05_crops artifacts
```

New modules:

- `observations.py`: JSON-safe `TrackObservation`, runtime-only `RuntimeTrackObservation`, generation-aware `TrackIdentity`, and bounded observation collection.
- `crop_quality.py`: padded crop extraction, crop completeness, edge-touch checks, brightness, contrast, Laplacian sharpness when OpenCV is available, and bounded preliminary scoring.
- `crop_collector.py`: bounded per-track crop candidate retention with `highest_preliminary_score`, `uniform_temporal`, and `hybrid_quality_temporal` policies.
- `crop_artifacts.py`: optional crop image writing and JSONL artifacts for observations, candidates, and completed bundles.
- `crop_pipeline.py`: strict sequential source -> detection -> tracking -> lifecycle -> crop collection pipeline.
- `run_step5_crop_collection_validation.py`: synthetic and real crop collection validation runner.

Identity policy:

```text
source_id + track_id + track_generation
```

Step 5 never treats `track_id` alone as a completed-track identity. It does not run plate YOLO, Florence, OCR, colour detection, search indexing, events, ReID, queues, threads, multiprocessing, or production pipeline integration.

Step 5 artifact layout:

```text
debug_runs/streaming_tracking_pipeline/streaming_tracking_step5_<video>_<backend>_<timestamp>/
|-- 01_source/
|   `-- source_metadata.json
|-- 02_detections/
|   `-- detection_packets.jsonl
|-- 03_tracks/
|   `-- tracked_frame_packets.jsonl
|-- 04_lifecycle/
|   |-- lifecycle_events.jsonl
|   |-- active_track_snapshots.jsonl
|   |-- completed_tracks.jsonl
|   `-- lifecycle_summary.json
|-- 05_crops/
|   |-- track_observations.jsonl
|   |-- crop_candidates.jsonl
|   |-- completed_track_crop_bundles.jsonl
|   |-- crop_collection_summary.json
|   `-- images/
`-- reports/
    |-- step5_crop_collection_report.json
    `-- step5_validation_result.json
```

Validated Step 5 commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover tests.td_case2.streaming_tracking_pipeline.tests
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step5_crop_collection_validation --mode synthetic_crop_collection --save-crop-images --max-candidates-per-track 3
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step5_crop_collection_validation --mode real_tracking_crop_collection --max-processed-frames 60 --target-fps 4 --confidence 0.05 --track-high-threshold 0.05 --track-low-threshold 0.01 --new-track-threshold 0.05 --max-candidates-per-track 4 --save-crop-images
```

Synthetic validation emitted 10 observations, 10 crop candidates, 1 completed crop bundle, and retained 3 candidates for the completed identity.

Real validation on the Step 3 evidence video processed 60 frames, emitted 22 observations, created 21 crop candidates, retained 13 bounded candidates, emitted 6 completed crop bundles, rejected 1 too-small box, and wrote crop images under `debug_runs/streaming_tracking_pipeline/streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138/`.

## What Step 6 Implements

Step 6 adds deterministic final best-crop selection from each `CompletedTrackCropBundle`:

```text
CompletedTrackCropBundle
  -> final crop-quality score
  -> primary eligibility checks
  -> temporal / visual diversity filtering
  -> primary crops and optional fallback crop
  -> SelectedTrackCropSet
```

New modules:

- `crop_selection.py`: `FinalBestCropSelector`, final score breakdowns, `SelectedCrop`, `SelectedTrackCropSet`, and `SelectedCropJob`.
- `crop_selection_metrics.py`: selection summaries by status, completion reason, dominant class, observation-count bucket, rejection reason, score component, and missing metrics.
- `crop_selection_artifacts.py`: JSONL writers, Step 5 bundle artifact loader, selected-crop job output, and optional preview sheets.
- `crop_selection_pipeline.py`: strict sequential crop selection on live Step 5 bundles or existing Step 5 artifacts.
- `run_step6_best_crop_validation.py`: synthetic, existing-artifact, and bounded real validation runner.

Final score:

```text
weighted normalized quality components
  - bounded penalties for edge touching, clipping/incomplete crop, and low resolution
```

The selector preserves both `preliminary_rank_score` from Step 5 and `final_score` from Step 6. Missing optional metrics are deterministic zero-valued components and are listed in `missing_metric_names`; they are not treated as perfect.

Default normalization:

- confidence and completeness are probabilities.
- bbox area ratio is divided by `bbox_area_normalization_cap`.
- sharpness is divided by `sharpness_normalization_cap`.
- contrast is divided by `contrast_normalization_cap`.
- brightness peaks at `target_brightness`.
- temporal position prefers the middle of retained candidates.
- plate visibility is a zero-weight component until Step 7 exists.

Primary crops require track-level gates (`minimum_track_observations_for_primary`, `minimum_candidates_for_primary`) and candidate-level gates such as final score threshold, crop path, dimensions, non-edge touching, completeness, brightness range, contrast threshold, and optional sharpness threshold. Every rejection is counted with an explicit reason.

Fallback is intentionally less strict. It can keep a valid crop from a short, weak, fragmented, edge-touching, or lower-score track when any valid candidate exists. By default, one distinct fallback is kept only when fewer than the desired number of primary crops were selected. If no candidate is valid, Step 6 still emits a `SelectedTrackCropSet` with `selection_status=no_valid_crop`.

Diversity policy:

- `quality_only` ignores diversity after eligibility.
- `quality_with_temporal_diversity` enforces frame/time separation.
- `quality_with_visual_diversity` rejects near-duplicate crop boxes by IoU.
- `hybrid` enforces both temporal and bbox diversity.

Step 6 artifact layout:

```text
debug_runs/streaming_tracking_pipeline/streaming_tracking_step6_<video>_<backend>_<timestamp>/
|-- 01_source/
|-- 02_detections/
|-- 03_tracks/
|-- 04_lifecycle/
|-- 05_crops/
|-- 06_selected_crops/
|   |-- selected_track_crop_sets.jsonl
|   |-- selected_primary_crops.jsonl
|   |-- selected_fallback_crops.jsonl
|   |-- selected_crop_jobs.jsonl
|   |-- crop_selection_rejections.jsonl
|   `-- previews/
`-- reports/
    |-- crop_selection_summary.json
    |-- step6_best_crop_report.json
    |-- step6_best_crop_pipeline_report.json
    `-- step6_validation_result.json
```

Validated Step 6 commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\config.py tests\td_case2\streaming_tracking_pipeline\crop_selection.py tests\td_case2\streaming_tracking_pipeline\crop_selection_metrics.py tests\td_case2\streaming_tracking_pipeline\crop_selection_artifacts.py tests\td_case2\streaming_tracking_pipeline\crop_selection_pipeline.py tests\td_case2\streaming_tracking_pipeline\run_step6_best_crop_validation.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step6_best_crop_validation --mode synthetic_best_crop_selection --no-create-previews
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step6_best_crop_validation --mode existing_step5_artifacts --step5-run-dir debug_runs\streaming_tracking_pipeline\streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138 --create-previews
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step6_best_crop_validation --mode real_best_crop_selection --max-processed-frames 60 --target-fps 4 --confidence 0.05 --track-high-threshold 0.05 --track-low-threshold 0.01 --new-track-threshold 0.05 --max-candidates-per-track 4 --create-previews
```

Synthetic validation processed 16 completed bundles, selected 12 primary crops and 9 fallback crops, produced 5 primary-selected tracks, 9 fallback-only tracks, 2 no-valid tracks, and verified disabled-selector behavior.

Existing Step 5 artifact validation processed 6 completed bundles from `streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138`, selected 3 primary crops and 6 fallback crops, produced 2 tracks with primary crops, 4 fallback-only tracks, 0 no-valid tracks, and generated 6 preview images.

Bounded real Step 1-6 validation produced the same Step 6 selection counts after rerunning the evidence video through YOLO, ByteTrack, lifecycle, and crop collection.

Comparison with current `td_case2` Step 05:

- Existing Step 05 selects from historical Step 04B track dictionaries; streaming Step 6 selects from lifecycle-completed `CompletedTrackCropBundle` records.
- Existing Step 05 uses track IDs as provided by that pipeline; streaming Step 6 uses `source_id + track_id + track_generation`.
- Existing Step 05 scores confidence, area, border touch, class consistency, temporal middle, and crop existence; streaming Step 6 scores confidence, area, sharpness, brightness, contrast, completeness, temporal position, and explicit penalties.
- Existing Step 05 has simple near-duplicate avoidance by time gap; streaming Step 6 supports temporal and bbox visual diversity policies.
- Existing Step 05 writes `05_best_track_frames.json`; streaming Step 6 writes normalized JSONL selected crop sets, primary/fallback streams, rejection logs, OCR-ready job descriptions, and previews.
- Streaming Step 6 can later act as an alternate backend, but no backend switch is integrated yet.

## What Step 7 Implements

Step 7 consumes Step 6 `SelectedCropJob` records and runs bounded, sequential ANPR and vehicle-colour enrichment:

```text
Selected vehicle crop
  -> Plate YOLO on the crop only
  -> bounded plate crop candidates
  -> Florence-2 OCR on plate crops
  -> Florence-2 vehicle colour on the selected vehicle crop
  -> raw structured track-level ANPR/colour result
```

New modules:

- `anpr_schemas.py`: raw plate candidate, Florence OCR, Florence colour, and track result dataclasses plus safe raw text/colour normalization.
- `plate_detection.py`: lazy local Ultralytics plate detector with crop-local bbox clipping, min-size filtering, deterministic sorting, and deterministic plate crop output paths.
- `florence_inference.py`: shared local-only Florence-2 loader with optional PEFT adapter, fake-model injection for tests, OCR generation, and colour VQA generation.
- `anpr_pipeline.py`: strict sequential Step 7 orchestration over selected crop jobs, primary/fallback ordering, bounded crop/candidate caps, stop-after-first-raw-text behavior, and one colour call per track by default.
- `anpr_artifacts.py`: JSONL artifact writer and Step 6 selected-crop-job loader.
- `anpr_metrics.py`: summary metrics by status, crop role/rank, object class, generation, confidence bucket, colour, and failure reason.
- `run_step7_anpr_colour_validation.py`: synthetic fake-model and existing Step 6 artifact replay runner.

Step 7 intentionally does not validate final plate formats, retry OCR, merge fragments, alter tracking, create object records, index search, emit events, or add queues/threads/multiprocessing. Those remain later-stage boundaries.

Step 7 artifact layout:

```text
debug_runs/streaming_tracking_pipeline/streaming_tracking_step7_<mode>_<timestamp>/
|-- 07_anpr/
|   |-- plate_detection_candidates.jsonl
|   |-- florence_ocr_results.jsonl
|   |-- florence_colour_results.jsonl
|   |-- track_anpr_colour_results.jsonl
|   |-- step7_selected_crop_jobs.jsonl
|   `-- plate_crops/
`-- reports/
    |-- anpr_colour_summary.json
    |-- step7_anpr_colour_report.json
    `-- step7_validation_result.json
```

Validated Step 7 commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\anpr_schemas.py tests\td_case2\streaming_tracking_pipeline\plate_detection.py tests\td_case2\streaming_tracking_pipeline\florence_inference.py tests\td_case2\streaming_tracking_pipeline\anpr_pipeline.py tests\td_case2\streaming_tracking_pipeline\anpr_artifacts.py tests\td_case2\streaming_tracking_pipeline\anpr_metrics.py tests\td_case2\streaming_tracking_pipeline\run_step7_anpr_colour_validation.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step7_anpr_colour_validation --mode synthetic_step7
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step7_anpr_colour_validation --mode existing_step6_artifacts --max-tracks 6 --florence-model-path "C:\Mukul K\mk\models\Florence-2-base-ft" --florence-adapter-path OCR_MUKUL\adaptor_florance_baseFT --plate-detector-model-path OCR_MUKUL\license_plate_weights.pt --plate-device cpu --florence-device cpu --florence-dtype float32 --max-new-tokens 64
```

Synthetic validation passed with 1 processed track, 1 plate candidate, 1 raw OCR result, and 1 normalized vehicle colour.

Existing Step 6 artifact replay on `streaming_tracking_step6_existing_step5_20260718_144055` processed 6 bounded identities and 9 selected crop jobs. Florence loaded locally and colour inference ran without errors; plate YOLO ran on all selected vehicle crops but found 0 plate candidates in this bounded replay, so OCR attempts correctly remained 0.

## What Step 7.5 Implements

Step 7.5 is a focused diagnostic and bounded retry layer for plate detection. It replays only Step 6 selected crops and explains whether Step 7 produced no plates because the detector returned no boxes or because boxes were filtered by confidence, class, geometry, clipping, size, or crop-writing rules.

Retry order:

```text
primary_rank_1 -> primary_rank_2 -> primary_rank_3 -> fallback
```

The controller skips unavailable ranks, skips duplicate crop paths, and respects `maximum_vehicle_crop_attempts_per_track`. It does not return to rejected Step 6 candidates.

New modules:

- `plate_diagnostics.py`: diagnostic config consumer, constrained attempt/box statuses, raw detector box preservation, box classification, vehicle-crop metadata, annotation writing, and valid/rejected crop saving.
- `plate_retry.py`: bounded selected-crop retry controller.
- `plate_diagnostic_artifacts.py`: JSONL/report writer for attempts, raw boxes, and track results.
- `plate_diagnostic_metrics.py`: summary metrics by status, role/rank, diagnostic threshold, generation, Step 6 score bucket, and vehicle crop size bucket.
- `run_step75_plate_diagnostics.py`: synthetic, existing Step 6 artifact, and existing Step 7 artifact diagnostic runner.

Diagnostic statuses distinguish:

```text
no_raw_detector_boxes
all_boxes_below_threshold
all_boxes_wrong_class
all_boxes_invalid_geometry
all_boxes_empty_after_clipping
all_boxes_too_small
plate_crop_write_failed
vehicle_crop_missing
vehicle_crop_unreadable
detector_disabled
detector_load_error
detector_inference_error
ocr_not_requested
ocr_not_attempted_no_plate
ocr_success_non_empty
ocr_success_empty
ocr_inference_error
```

Threshold probing:

- The normal Step 7 detection threshold remains explicit.
- Step 7.5 uses the configured diagnostic thresholds only in diagnostic mode.
- The detector is called at the minimum diagnostic threshold where practical.
- Every raw box records the diagnostic acceptance threshold, whether it passed the normal threshold, and which thresholds it passed.
- Low-threshold accepted candidates are labeled as accepted by diagnostic threshold, not normal Step 7 inference.

Vehicle crop metadata captured per attempt:

```text
crop width, crop height, area, brightness, sharpness, edge touching, crop completeness,
Step 6 role/rank, Step 6 selection score, Step 6 warnings, source bbox metadata
```

Annotation and crop outputs:

```text
07_5_plate_diagnostics/
|-- annotated_vehicle_crops/
|-- accepted_plate_crops/
|-- rejected_plate_crops/
|-- plate_diagnostic_attempts.jsonl
|-- raw_plate_box_diagnostics.jsonl
`-- track_plate_diagnostic_results.jsonl
```

OCR policy:

- OCR runs only when `--run-ocr` is passed and only on accepted plate candidates.
- OCR never runs on vehicle crops without plate candidates, below-threshold boxes, wrong-class boxes, invalid boxes, empty clipped boxes, or too-small boxes.
- `--stop-after-first-plate` stops after the accepted candidate attempt, after optional OCR for that candidate.
- `--stop-after-first-non-empty-ocr` continues past empty OCR until a non-empty raw OCR result or selected crops are exhausted.
- No final license-plate validation, correction, cross-crop voting, search indexing, event creation, ReID, queues, threads, multiprocessing, tracking changes, or production integration is included.

Validated Step 7.5 commands:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\plate_diagnostics.py tests\td_case2\streaming_tracking_pipeline\plate_retry.py tests\td_case2\streaming_tracking_pipeline\plate_diagnostic_artifacts.py tests\td_case2\streaming_tracking_pipeline\plate_diagnostic_metrics.py tests\td_case2\streaming_tracking_pipeline\run_step75_plate_diagnostics.py tests\td_case2\streaming_tracking_pipeline\plate_detection.py tests\td_case2\streaming_tracking_pipeline\config.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step75_plate_diagnostics --mode synthetic_plate_diagnostics --run-ocr --stop-after-first-non-empty-ocr --save-annotations --save-rejected-plate-crops
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step75_plate_diagnostics --mode existing_step6_artifacts --step6-run-dir debug_runs\streaming_tracking_pipeline\streaming_tracking_step6_existing_step5_20260718_144055 --plate-detector-model OCR_MUKUL\license_plate_weights.pt --device cpu --normal-plate-confidence 0.20 --diagnostic-thresholds 0.25,0.15,0.10,0.05 --minimum-plate-width 6 --minimum-plate-height 4 --max-attempts-per-track 4 --save-annotations --save-rejected-plate-crops
```

Real Step 6 artifact diagnostic result:

- Selected crop jobs inspected: 9.
- Vehicle crop attempts: 9.
- Plate model: `OCR_MUKUL/license_plate_weights.pt`.
- Model classes: `{0: "License_Plate"}`.
- Normal threshold: `0.20`.
- Diagnostic thresholds: `0.25, 0.15, 0.10, 0.05`.
- Raw detector boxes: 0.
- Boxes below normal threshold: 0.
- Boxes rejected by class/geometry/size: 0.
- Accepted plate candidates: 0.
- OCR calls: 0.
- Tracks exhausting selected crops: 6.
- Annotated vehicle crops written: 9.
- Known-good check found an existing plate crop at `debug_runs/ANPR1-D_20260715_175538/06_plate_crops/vehicle_track_0001_frame_000030_combined_001_plate.jpg`; the same detector produced 1 raw/accepted box at confidence `0.776766`.

Direct conclusion: on the Step 6 selected crop replay, the detector produced truly zero raw boxes even at the lowest diagnostic threshold (`0.05`). The likely cause is crop suitability for ANPR: 8 of 9 selected crops are tiny, most are person crops, and the only car crop is `110x33`, too small/low-context for the plate detector to localize a plate. Step 8 remains responsible for validation and final object-record policy; it has not been started.

## Image ANPR Validation

The image ANPR validation runner tests the local plate YOLO and Florence OCR on a folder of standalone images without running video decoding, tracking, lifecycle, crop collection, crop selection, search, events, ReID, queues, threads, or production code.

It is useful for separating model behavior from the streaming crop-selection pipeline:

```text
image folder
  -> deterministic image discovery
  -> raw plate YOLO diagnostics on each full image
  -> accepted/rejected plate crop saving
  -> optional Florence OCR on accepted detector crops
  -> optional Florence OCR directly on the original input image
  -> per-image and overall reports
```

Detector-crop OCR and direct-input OCR are recorded separately. Direct-input OCR is intended for tightly cropped plate images or quick Florence checks; it is not merged with detector-crop OCR and it is not treated as verified plate text.

New modules:

- `image_anpr_validation.py`: image discovery, image-level config and schemas, detector diagnostic reuse, crop copying, OCR execution, summary generation, and JSONL/report writing.
- `run_image_anpr_validation.py`: reusable CLI runner.

Output layout:

```text
debug_runs/streaming_tracking_pipeline/image_anpr_validation_<timestamp>/
|-- input_manifest.json
|-- annotated_images/
|-- accepted_plate_crops/
|-- rejected_plate_crops/
|-- image_results.jsonl
|-- raw_plate_box_diagnostics.jsonl
|-- florence_ocr_results.jsonl
`-- reports/
    |-- image_anpr_summary.json
    `-- image_anpr_report.json
```

Validated command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_image_anpr_validation --input-dir "debug_runs\test pictures" --plate-detector-model OCR_MUKUL\license_plate_weights.pt --florence-model "C:\Mukul K\mk\models\Florence-2-base-ft" --florence-adapter OCR_MUKUL\adaptor_florance_baseFT --device cpu --dtype float32 --normal-plate-confidence 0.25 --diagnostic-thresholds 0.25,0.15,0.10,0.05 --minimum-plate-width 6 --minimum-plate-height 4 --max-plate-candidates 3 --run-ocr --direct-ocr-on-input --stop-after-first-non-empty-ocr --save-annotations --save-rejected-plate-crops
```

Real run result on `debug_runs/test pictures`:

- Images discovered/read: 7 / 7.
- Plate model calls: 7.
- Raw detector boxes: 13.
- Images with raw boxes: 7.
- Images with accepted plates: 7.
- Boxes below normal threshold: 1.
- Rejections by class/geometry/clipping/size: 0.
- Accepted plate crops saved: 13.
- Detector-crop OCR calls: 7.
- Detector-crop non-empty OCR outputs: 7.
- Direct-input OCR calls: 7.
- Direct-input non-empty OCR outputs: 7.
- Final status: all 7 images `plate_found_ocr_non_empty`.
- Artifacts: `debug_runs/streaming_tracking_pipeline/image_anpr_validation_20260718_153928/`.

All seven images are suitable image-level ANPR inputs for model validation because the detector produced accepted plate boxes and Florence returned non-empty raw OCR on accepted detector crops. Raw OCR remains unverified; Step 8 still owns plate normalization, validation, agreement, and final object-record policy.

## 10 FPS Video ANPR Validation

`run_anpr_video_10fps_validation.py` orchestrates the existing isolated modules for a bounded sequential video pass:

```text
OpenCV 10 FPS source
  -> vehicle/person YOLO
  -> Ultralytics ByteTrack
  -> lifecycle manager
  -> crop collection
  -> Step 6 selected crops
  -> vehicle-only ANPR eligibility gate
  -> Step 7.5 plate diagnostics and bounded retry
  -> Florence OCR on accepted plate crops
  -> Florence colour on eligible vehicle crops
```

The runner does not introduce queues, threads, multiprocessing, ReID, fragment merging, search indexing, event creation, VLM review, production integration, or final plate verification. Step 8 now performs final plate validation as a separate artifact-only pass.

New modules:

- `anpr_job_eligibility.py`: explicit vehicle-class, readable-crop, and minimum-size ANPR job gate with JSON-safe exclusion records.
- `run_anpr_video_10fps_validation.py`: end-to-end orchestration runner that reuses the existing Step 5, Step 6, Step 7.5, plate YOLO, and Florence wrappers.

Validated bounded command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_anpr_video_10fps_validation --video "C:\Mukul K\mk\test_video\anpr_test_5min.mp4" --target-fps 10 --max-processed-frames 600 --tracking-backend ultralytics_bytetrack --vehicle-detector-model object\vehical_detection\best_old.pt --plate-detector-model OCR_MUKUL\license_plate_weights.pt --florence-model "C:\Mukul K\mk\models\Florence-2-base-ft" --florence-adapter OCR_MUKUL\adaptor_florance_baseFT --normal-plate-confidence 0.25 --diagnostic-thresholds 0.25,0.15,0.10,0.05 --minimum-plate-width 6 --minimum-plate-height 4 --max-attempts-per-track 4 --save-annotations --save-rejected-plate-crops
```

Bounded real result on `anpr_test_5min.mp4`:

- Source FPS: 30.0.
- Target FPS: 10.0.
- Processed frames: 600, covering frame 0 through 1797 (`59.9` seconds).
- Vehicle model: `object/vehical_detection/best_old.pt` on CUDA, no fallback.
- Device policy: vehicle YOLO CUDA, plate YOLO CUDA, Florence CUDA `float16`; ByteTrack and OpenCV remain CPU.
- Detections: 106 filtered detections (`bus=24`, `car=39`, `motorcycle=19`, `truck=24`) over 600 frames.
- Tracking: 61 emitted track observations and 20 normalized tracker IDs.
- Lifecycle/crops: 23 completed vehicle track generations, 61 crop candidates, 56 retained candidates.
- Step 6 selected crops: 18 fallback vehicle crops, 0 primary vehicle crops, 5 no-valid-crop vehicle tracks.
- ANPR gate: 17 eligible vehicle crop jobs, 1 crop excluded as `vehicle_crop_too_small`, 0 person crops sent to ANPR.
- Plate diagnostics: 17 plate detector calls, 12 raw detector boxes, 12 accepted plate candidates, 5 no-raw-box attempts.
- OCR: 12 accepted plate-crop OCR calls, all 12 returned non-empty raw OCR text.
- Colour: 17 Florence colour calls.
- Runtime: `19.936328` seconds total and `4.731093` seconds Florence runtime with auto CUDA, versus the previous CPU-Florence bounded run at `45.403463` seconds total and `32.13337` seconds Florence runtime.
- CUDA: available on `NVIDIA GeForce RTX 5070 Ti`; peak allocated VRAM reported by PyTorch was `718.092 MB`.
- Artifacts: `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/`.

Conclusion: the 10 FPS bounded video pipeline can produce vehicle crops large enough for plate localization and OCR on this ANPR-oriented video. The useful ANPR evidence came from fallback crops, not primary crops, because Step 6 selected no primary vehicle crops under the current primary gates. Step 8 validates these saved raw OCR artifacts without rerunning inference.

## Step 8 Plate Validation

`run_step8_plate_validation.py` consumes saved Step 7 and Step 7.5 artifacts only:

```text
Raw OCR candidates
  -> text cleaning
  -> bounded OCR character alternatives
  -> Indian plate-format validation
  -> per-track-generation agreement
  -> candidate scoring
  -> final verified / weak / invalid / no-plate result
```

Validated command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step8_plate_validation --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910
```

Step 8 real result on the bounded `160910` run:

- Track generations processed: 23.
- Tracks with plate detection: 12.
- Tracks without plate detection: 11.
- Raw OCR records read: 12.
- Normalized candidates: 12.
- Corrected candidates: 2 (`I->1`, `1->I`).
- Strict format matches: 3.
- Relaxed format matches: 0.
- Partial candidates: 6.
- Not-plate-like candidates: 3.
- Final statuses: 3 verified, 6 weak, 3 invalid, 11 no-plate-detected.
- Exact agreement groups: 0.
- Similarity-based agreements: 0.
- Verified texts: `21g0=UP81CH4158`, `22g0=UP81CW4150`, `30g0=UP14CW4087`.
- Artifacts: `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910/08_plate_validation/`.

Step 8 does not rerun vehicle YOLO, plate YOLO, Florence, ByteTrack, tracking, crop collection, or video processing.

Exact full-video command, not executed during the bounded validation:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_anpr_video_10fps_validation --video "C:\Mukul K\mk\test_video\anpr_test_5min.mp4" --target-fps 10 --full-video --tracking-backend ultralytics_bytetrack --vehicle-detector-model object\vehical_detection\best_old.pt --plate-detector-model OCR_MUKUL\license_plate_weights.pt --florence-model "C:\Mukul K\mk\models\Florence-2-base-ft" --florence-adapter OCR_MUKUL\adaptor_florance_baseFT --normal-plate-confidence 0.25 --diagnostic-thresholds 0.25,0.15,0.10,0.05 --minimum-plate-width 6 --minimum-plate-height 4 --max-attempts-per-track 4 --save-annotations --save-rejected-plate-crops
```

## Step 9 Searchable Object Records

`run_step9_searchable_object_records.py` is an artifact-only export stage. It consumes the completed lifecycle tracks, selected crop sets, raw ANPR/colour records, and Step 8 final plate-validation records from a run directory:

```text
completed track generations
  -> selected primary/fallback crop evidence
  -> Step 8 verified / weak / invalid / no-plate status
  -> one generation-aware searchable vehicle record
```

Step 9 does not rerun vehicle YOLO, ByteTrack, crop collection, crop selection, plate YOLO, Florence, OpenCV video processing, Step 8 validation, ReID, event creation, vector indexing, queues, threads, multiprocessing, or production integration.

Validated command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step9_searchable_object_records --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012
```

Output layout:

```text
09_searchable_objects/
|-- searchable_vehicle_records.jsonl
|-- searchable_vehicle_records_flat.json
|-- verified_plate_vehicle_records.jsonl
|-- weak_plate_vehicle_records.jsonl
|-- no_plate_vehicle_records.jsonl
`-- reports/
    |-- step9_searchable_objects_summary.json
    `-- step9_searchable_objects_report.json
```

Full-video Step 9 result on `streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012`:

- Input artifact counts: 167 completed tracks, 167 selected crop sets, 145 track ANPR/colour records, 167 Step 8 final results.
- Vehicle records created: 167.
- Records by class: `car=82`, `bus=32`, `truck=28`, `motorcycle=25`.
- Records by plate status: `verified=58`, `weak=28`, `invalid=14`, `no_plate_detected=67`.
- Records with vehicle crop evidence: 149.
- Records with plate crop evidence: 100.
- Records with colour: 145.
- No missing input artifacts, no join failures, no duplicate record IDs, and no records missing track times.
- Query smoke checks: `white car=27`, `verified plates=58`, `UP81CH4158=1`, `UP81=2`, `red vehicle=17`, `vehicles between 60 and 120 seconds=16`, `weak OCR=28`, `no plate=67`.

Weak plate text is included only as weak-search evidence. Invalid raw OCR is preserved in metadata where available, but it is not exposed as verified/searchable plate text.

## Step 10 Structured Search

`run_step10_search_validation.py` searches Step 9 vehicle records with deterministic parsing, structured filters, token matching, and score-component ranking:

```text
user query
  -> deterministic parser
  -> structured filters
  -> searchable token/text matching
  -> deterministic ranking
  -> result package
```

Step 10 consumes only:

```text
09_searchable_objects/searchable_vehicle_records.jsonl
```

It does not rerun video processing, YOLO, ByteTrack, Florence, OCR, colour detection, Step 8, Step 9, embeddings, FAISS, a vector database, an LLM/VLM, ReID, UI, or production APIs. Step 10 does not mutate source records and does not treat invalid OCR as a valid plate.

Supported structured filters:

- object class
- colour
- exact plate text
- plate prefix
- plate status
- first/last seen time overlap
- track ID
- track generation

Validated command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step10_search_validation --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012
```

Single-query mode:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step10_search_validation --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012 --query "UP81CH4158" --top-k 10
```

Output layout:

```text
10_structured_search/
|-- search_index_summary.json
|-- validation_queries.json
|-- validation_search_results.jsonl
|-- validation_search_results_flat.json
`-- reports/
    |-- step10_search_summary.json
    |-- step10_search_paths.json
    `-- step10_search_report.json
```

Full-video Step 10 result:

- Records indexed: 167.
- Validation queries executed: 11.
- Expected-count mismatches: 0.
- Parser warnings: 0.
- Query counts: `white car=27`, `verified plates=58`, `UP81CH4158=1`, `UP81=2`, `red vehicle=17`, `vehicles between 60 and 120 seconds=16`, `weak OCR=28`, `no plate=67`, `motorcycle without plate=11`, `white car between 2 and 3 minutes=8`, `truck with verified plate=3`.
- Exact plate search: `UP81CH4158` returned `anpr_test_5min:track_000021:gen_000`.
- Prefix search: `UP81` returned `anpr_test_5min:track_000021:gen_000` and `anpr_test_5min:track_000022:gen_000`.
- Time search uses visible-range overlap, not only first-seen containment.

Ranking priority favors exact verified plate matches, exact weak plate matches, plate-prefix matches, complete structured filter matches, free-text token matches, and evidence completeness. Weak plate text can participate when `--include-weak-plates` is enabled. Invalid OCR is not plate-searchable.

## Step 11 Result Cards

`run_step11_result_card_packaging.py` converts saved Step 10 search responses into UI-ready result-card JSON packages:

```text
Step 10 search response
  -> join Step 9 record by record_id
  -> preserve rank and score
  -> add labels, badges, image paths, warnings
  -> JSON packages and optional static HTML preview
```

Step 11 consumes only:

```text
09_searchable_objects/searchable_vehicle_records.jsonl
10_structured_search/validation_search_results.jsonl
```

It does not rerun video processing, tracking, YOLO, Florence, OCR, colour detection, Step 8, Step 9, Step 10 indexing, embeddings, vector search, production UI/API code, queues, worker threads, Step 07b, events, or VLM integration.

Validated full demo command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step11_result_card_packaging --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012 --write-html-preview
```

Validated direct-query command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step11_result_card_packaging --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012 --query "white car between 2 and 3 minutes" --top-k 20 --write-html-preview
```

Output layout:

```text
11_result_cards/
|-- result_card_packages.jsonl
|-- result_cards_flat.json
|-- demo_query_cards.json
|-- result_card_schema.json
|-- demo_result_cards.html
`-- reports/
    |-- step11_result_card_paths.json
    |-- step11_result_cards_summary.json
    `-- step11_result_cards_report.json
```

Full-video Step 11 result:

- Queries packaged: 11.
- Cards created: 138 with top 20 results per query.
- Cards with vehicle images: 133.
- Cards with plate images: 98.
- Missing vehicle-image cards: 5.
- Missing plate-image cards: 40.
- Cards by status: `verified=61`, `weak=30`, `no_plate_detected=40`, `invalid=7`.
- Cards by class: `car=78`, `motorcycle=29`, `truck=17`, `bus=14`.
- Cards by colour: `white=52`, `red=25`, `black=14`, `unknown=13`, `green=10`, `yellow=9`, `silver=6`, `blue=5`, `gray=2`, `pink=2`.
- Static HTML preview: `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/11_result_cards/demo_result_cards.html`.
- HTML preview reference check: 231 image references, 0 missing referenced files.

Plate display policy:

- Verified plates are shown normally.
- Weak plates show a `Weak OCR` badge.
- Invalid plate candidates are labeled invalid and are not displayed as valid number plates.
- No-plate cards show `No plate detected`.

## Object Class and Dominant Colour Corrections

The isolated pipeline now has a central class-normalization policy:

```text
person
car
motorcycle
bicycle
bus
truck
other_vehicle
other_object
```

Two-wheeler synonyms such as `motorcycle`, `motorbike`, `scooter`, and `two_wheeler` normalize to `motorcycle`, never to `car`. Unknown vehicle-like classes such as `3Wheeler` normalize to `other_vehicle`.

Object-folder inspection on this workspace found:

- `object/vehical_detection/best_old.pt`: loads successfully, task `detect`, raw classes `{0: 3Wheeler, 1: bus, 2: car, 3: motorcycle, 4: truck}`.
- `object/Person_detection.pt`: present but unloadable in the current environment because its zip archive contains a single root entry named `Person_detection.pt`, which PyTorch rejects with `file in archive is not in a subdirectory`.
- `object/Person_detection.py`: present and zip-like with the same size as the `.pt` file; not a usable Python source module.

The multi-model detection helper can combine vehicle and person/object detection packets and applies class-aware duplicate suppression. Real person/object detection is blocked until the local person checkpoint is repaired or re-exported.

Dominant colour analysis uses object crop regions instead of the full frame and writes debug artifacts:

```text
12_object_class_colour_validation/
|-- object_model_report.json
|-- dominant_colour_results.jsonl
|-- search_probe_results.json
|-- object_class_colour_validation_summary.json
`-- colour_debug/
    `-- <record_id>/
        |-- original_crop.jpg
        |-- analysed_region.jpg
        |-- excluded_region_mask.jpg
        `-- dominant_colour_report.json
```

Validated command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_object_class_colour_validation --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012 --max-colour-records 40
```

Validation result on the full-video Step 9 artifacts:

- Records read: 167.
- Normalized class counts: `car=82`, `bus=32`, `truck=28`, `motorcycle=25`.
- Person records/crops: 0 because the discovered person model is unloadable.
- Dominant colour samples analysed: 40.
- Dominant colour sample counts: `gray=25`, `black=7`, `maroon=4`, `silver=4`.
- Example tail-light/small-region rejection: `anpr_test_5min:track_000019:gen_000` raw Florence colour was `red`, while central-body dominant colour was `gray` with coverage `0.648272`.
- Search probes: `motorcycle=25`, `black motorcycle=4`, `white car=27`, `red truck=0`, `person=0`, `person in white=0`, `person between 60 and 120 seconds=0`, `vehicles without plates=67`.

Search uses `normalized_class_name` and validated `dominant_colour` when those fields exist, falling back to older Step 9 fields for backward-compatible artifacts.

## Interactive artifact UI

The local inspection UI is implemented with Streamlit and reads existing run artifacts only. Opening the UI does not rerun YOLO, ByteTrack, Florence, OCR, colour analysis, Step 8, or video processing.

Launch command:

```powershell
streamlit run tests\td_case2\streaming_tracking_pipeline\interactive_results_ui.py -- --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012
```

The UI tabs cover dashboard metrics, structured search, all objects, per-vehicle details, plate review, tracking review, colour review, pipeline metrics, run artifacts, and a disabled-by-default new-video runner. Manual plate review decisions are written separately under:

```text
<run-dir>/ui_state/
|-- manual_plate_reviews.jsonl
|-- saved_searches.json
`-- ui_preferences.json
```

The UI resolves artifact paths against the run directory and repository root, lazily renders images, paginates result cards, and shows missing-image or missing-artifact warnings instead of failing the page. Search widgets reuse the deterministic Step 10 parsing, filtering, and ranking layer.

Object evidence display now uses a normalized artifact-only evidence record for each visible object:

```python
{
    "track_id": ...,
    "object_class": ...,
    "timestamp_sec": ...,
    "frame_index": ...,
    "full_frame_path": ...,
    "object_crop_path": ...,
    "plate_crop_path": ...,
}
```

Full-frame path resolution uses this deterministic priority:

```text
1. Selected-crop evidence full-frame path matching the object crop or frame index
2. Search-record full-frame/source-frame/evidence-frame fields
3. Crop-bundle full-frame path matching the object crop or frame index
4. Lifecycle crop-candidate full-frame path matching the object crop or frame index
5. Track-observation full-frame path matching the object frame
6. No full frame available
```

The UI never constructs guessed frame paths from crop filenames. If no saved full-frame artifact exists, it keeps showing the crop and renders `Full frame unavailable`. For the completed run `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012`, the current saved artifacts expose `0` full-frame paths for `167` visible records, so all 167 records currently use the missing-full-frame fallback.

Evidence images are rendered through centralized helpers in `ui_media.py` using fixed, letterboxed containers:

```text
Full frame:  640 x 360, 16:9, object-fit: contain
Object crop: 320 x 240, 4:3,  object-fit: contain
Plate crop:  300 x 100, 3:1,  object-fit: contain
Thumbnail:   240 x 135, 16:9, object-fit: contain
```

The same renderer is used by Search results, All Objects, Vehicle Details, Plate Review, Tracking Review, Colour Review, and future person cards.

## Person tracking support

The isolated pipeline now carries an explicit object group on detection and tracking records:

```text
person  -> object_group=person
car/bus/truck/motorcycle/bicycle/other_vehicle -> object_group=vehicle
unknown classes -> object_group=object
```

Class normalization resolves the group before ByteTrack, and ByteTrack preserves it on `TrackedObject`. Lifecycle records preserve the group for completed tracks. Step 9 can now emit mixed searchable records: vehicle records keep the existing ANPR fields and person records use `object_type=person`, `object_group=person`, null plate/colour fields, `plate_status=not_applicable`, and searchable tokens such as `person`, `people`, `pedestrian`, `track_<id>`, and time buckets.

Branch policy:

```text
object_group=vehicle -> existing best crop -> plate YOLO -> Florence OCR/colour -> Step 8 -> vehicle record
object_group=person  -> best crop/fallback crop -> person object record -> search/UI/event candidate input
```

Person records deliberately bypass plate YOLO, plate OCR, and vehicle-colour OCR. The implementation does not invent clothing colour, identity, age, gender, or activity attributes.

Detection configuration now supports combined and dual-detector modes through `ObjectTrackingConfig`:

```python
ObjectTrackingConfig(
    enable_vehicle_tracking=True,
    enable_person_tracking=False,
    detection_mode="combined",  # combined | dual
    vehicle_model_path=None,
    person_model_path="object/Person_detection.pt",
    track_object_groups=("vehicle",),
)
```

`validate_object_tracking_config()` fails clearly when person tracking is enabled but no configured detector advertises a `person` class. It resolves class names from model mappings; it does not hardcode COCO class ID `0`.

Current audit for `debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012`:

```text
vehicle model: object/vehical_detection/best_old.pt
vehicle model classes: 3Wheeler, bus, car, motorcycle, truck
vehicle_detector_supports_person: false
person model: object/Person_detection.pt
person_detector_load_status: loaded in corrected-model validation runs
person_detections_before_filter: 0
person_detections_after_filter: 0
person_tracks_created: 0
person_tracks_confirmed: 0
person_records_written: 0
vehicle_records_written: 167
```

Audit artifact:

```text
debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012/person_tracking_audit.json
```

Root cause of zero person records in the completed `163012` run: that run was vehicle-only and the configured vehicle detector does not include a `person` class. The corrected dedicated person detector now loads and is used by the combined runner below.

## Combined Vehicle + Person Pipeline

`run_combined_vehicle_person_pipeline.py` processes vehicles and persons in one video pass, one run directory, one lifecycle/crop stream, one searchable object artifact, and one Streamlit UI. Detector scheduling is sequential per frame:

```text
FramePacket
  -> vehicle YOLO
  -> person YOLO
  -> normalized class-aware merge
  -> one ordered ByteTrack stage
  -> lifecycle
  -> crop collection with object crop + matching full frame
  -> vehicle ANPR branch or person clothing-colour branch
  -> Step 8 / Step 9 / Step 10 combined artifacts
```

The vehicle detector is `object/vehical_detection/best_old.pt` with classes `3Wheeler`, `bus`, `car`, `motorcycle`, and `truck`. The person detector is `object/Person_detection.pt`, resolved from its model mapping with `person` at class ID `0`. `3Wheeler` is normalized internally as `3wheeler` while preserving the raw user-facing class label in model mappings and records.

Vehicles keep the existing plate YOLO -> Florence OCR -> Step 8 plate validation -> vehicle-colour path. Persons bypass plate YOLO, plate OCR, plate validation semantics, and vehicle-colour prompts; person records use `plate_status=not_applicable`, `anpr_bypassed=true`, and `vehicle_colour_status=not_applicable`.

Person clothing colour is deterministic visible-clothing analysis over selected person crops. It stores `upper_clothing_color`, `lower_clothing_color`, `dominant_clothing_color`, `clothing_color_confidence`, and `clothing_color_status`. The method does not infer identity, gender, age, race, ethnicity, or skin colour; unclear crops are marked `uncertain` or `unknown`.

Combined full-video command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_combined_vehicle_person_pipeline --video "C:\Mukul K\mk\test_video\anpr_test_5min.mp4" --target-fps 10 --full-video --tracking-backend ultralytics_bytetrack --vehicle-detector-model object\vehical_detection\best_old.pt --person-detector-model object\Person_detection.pt --plate-detector-model OCR_MUKUL\license_plate_weights.pt --florence-model "C:\Mukul K\mk\models\Florence-2-base-ft" --florence-adapter OCR_MUKUL\adaptor_florance_baseFT --device auto --normal-plate-confidence 0.25 --diagnostic-thresholds 0.25,0.15,0.10,0.05 --minimum-plate-width 6 --minimum-plate-height 4 --max-attempts-per-track 4 --save-annotations --save-rejected-plate-crops
```

Combined Streamlit UI command:

```powershell
tests\td_case2\.venv\Scripts\python.exe -m streamlit run tests\td_case2\streaming_tracking_pipeline\interactive_results_ui.py --server.headless true --server.port 8515 -- --run-dir debug_runs\streaming_tracking_combined_anpr_test_5min_20260718_191939
```

Full-video validation on `C:\Mukul K\mk\test_video\anpr_test_5min.mp4` wrote `debug_runs/streaming_tracking_combined_anpr_test_5min_20260718_191939/`. It processed 3,779 frames at 10 FPS over 377.8 seconds. Vehicle YOLO, person YOLO, plate YOLO, and Florence ran on CUDA with Florence float16; ByteTrack and OpenCV remained CPU. Peak allocated VRAM was 753.201 MB.

Validation metrics:

```text
detections: 3wheeler=4077, bus=87, car=469, motorcycle=46, person=578, truck=117
completed generations: 3wheeler=36, bus=16, car=58, motorcycle=12, person=70, truck=13
records indexed: 205
vehicle records: 145
person records: 60
UI class filter: 3wheeler, bus, car, motorcycle, person, truck
person records with object crops: 60/60
person records with matching full frames: 60/60
all records with matching full frames: 205/205
person clothing colour: detected=54 records, uncertain=4 records
vehicle plate statuses: verified=48, weak=22, invalid=18, no_plate_detected=57
person ANPR bypass count: 60 records
combined search examples: person=60, motorcycle=13, red car=7
runtime: 143.495465 seconds
real-time factor: 0.299411
```

Remaining limitations: the combined validation uses one ByteTrack stage over merged detections and does not add ReID/fragment merging; track fragmentation remains visible in short completed generations. Person clothing-colour analysis is deterministic crop-region colour estimation, not a semantic clothing parser.

## Run tests

```powershell
tests\td_case2\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\streaming_tracking_pipeline\tests -p "test_*.py"
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\anpr_job_eligibility.py tests\td_case2\streaming_tracking_pipeline\run_anpr_video_10fps_validation.py tests\td_case2\streaming_tracking_pipeline\video_source.py tests\td_case2\streaming_tracking_pipeline\yolo_stage.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step2_contract_validation
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step3_real_tracking_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step4_lifecycle_validation --mode synthetic_lifecycle
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step5_crop_collection_validation --mode synthetic_crop_collection
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step6_best_crop_validation --mode synthetic_best_crop_selection
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step7_anpr_colour_validation --mode synthetic_step7
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step75_plate_diagnostics --mode synthetic_plate_diagnostics
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_image_anpr_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_anpr_video_10fps_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step8_plate_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step9_searchable_object_records --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step10_search_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_step11_result_card_packaging --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_object_class_colour_validation --help
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_combined_vehicle_person_pipeline --help
tests\td_case2\.venv\Scripts\python.exe -m py_compile tests\td_case2\streaming_tracking_pipeline\interactive_results_ui.py tests\td_case2\streaming_tracking_pipeline\ui_data_loader.py tests\td_case2\streaming_tracking_pipeline\ui_filters.py tests\td_case2\streaming_tracking_pipeline\ui_media.py
tests\td_case2\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.run_person_tracking_audit --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012 --vehicle-model-path object\vehical_detection\best_old.pt --person-model-path object\Person_detection.pt
streamlit run tests\td_case2\streaming_tracking_pipeline\interactive_results_ui.py -- --run-dir debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012
```

The validation runner writes:

```text
debug_runs/streaming_tracking_pipeline/step2_contract_validation/
├── frame_packets.jsonl
├── detection_packets.jsonl
├── tracked_frame_packets.jsonl
└── step2_contract_validation_report.json
```

Still deliberately unimplemented in this isolated validation lane: RTSP production integration, production object-record emission, queues, workers, search indexing, event detection, physical-ID merging/ReID, speed/line analytics, and production integration.
