# Phase 1 Repository Implementation Notes

## Scope

Phase 1 repository implementation covers only:

1. `analytics.camera`
2. `analytics.video_source`
3. `analytics.processing_run`
4. `analytics.camera_run`
5. `analytics.processing_job`
6. `analytics.vehicle_track`
7. `analytics.track_observation`
8. `analytics.ai_model`
9. `analytics.run_model`
10. `analytics.processing_error`

## Actual Table Columns

### `analytics.camera`

- `id`
- `camera_code`
- `external_camera_id`
- `camera_name`
- `site_code`
- `location_name`
- `timezone`
- `enabled`
- `metadata`
- `created_at`
- `updated_at`

### `analytics.video_source`

- `id`
- `camera_id`
- `source_type`
- `external_recording_id`
- `source_reference`
- `source_start_at`
- `source_end_at`
- `source_fps`
- `frame_width`
- `frame_height`
- `metadata`
- `created_at`
- `updated_at`

Allowed `source_type` values:

- `LOCAL_FILE`
- `RTSP`
- `LIVE_STREAM`
- `VMS_RECORDING`
- `PLAYBACK_API`

### `analytics.processing_run`

- `id`
- `run_code`
- `pipeline_name`
- `pipeline_version`
- `execution_mode`
- `status`
- `configured_camera_count`
- `active_camera_count`
- `started_at`
- `completed_at`
- `total_frames_processed`
- `total_detections`
- `total_track_observations`
- `total_tracks`
- `host_name`
- `runtime_device`
- `configuration`
- `metrics`
- `created_at`
- `updated_at`

Allowed `execution_mode` values:

- `SEQUENTIAL`
- `THREADED`
- `BATCHED`
- `LIVE`

Allowed `status` values:

- `QUEUED`
- `RUNNING`
- `COMPLETED`
- `PARTIAL`
- `FAILED`
- `CANCELLED`

### `analytics.camera_run`

- `id`
- `processing_run_id`
- `camera_id`
- `video_source_id`
- `status`
- `reader_worker_name`
- `resolved_source_fps`
- `effective_processing_fps`
- `first_frame_number`
- `last_frame_number`
- `frames_read`
- `frames_processed`
- `detections_count`
- `track_observations_count`
- `completed_tracks_count`
- `discarded_tracks_count`
- `started_at`
- `completed_at`
- `metrics`
- `created_at`
- `updated_at`

Allowed `status` values:

- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

Idempotency key:

- unique `(processing_run_id, camera_id)`

### `analytics.processing_job`

- `id`
- `processing_run_id`
- `camera_run_id`
- `job_type`
- `status`
- `priority`
- `attempt_number`
- `worker_name`
- `started_at`
- `completed_at`
- `processing_time_ms`
- `input_summary`
- `output_summary`
- `error_code`
- `error_message`
- `created_at`
- `updated_at`

Allowed `job_type` values:

- `READ`
- `DETECT`
- `TRACK`
- `PERSIST`
- `BEST_FRAME_SELECTION`
- `PLATE_DETECTION`
- `OCR`
- `COLOR`
- `CROSS_CAMERA_MATCH`
- `EVENT_ANALYSIS`

Allowed `status` values:

- `QUEUED`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `RETRYING`
- `CANCELLED`

### `analytics.vehicle_track`

- `id`
- `processing_run_id`
- `camera_run_id`
- `camera_id`
- `track_uuid`
- `local_track_id`
- `vehicle_class`
- `first_seen_at`
- `last_seen_at`
- `first_frame_number`
- `last_frame_number`
- `first_video_time_seconds`
- `last_video_time_seconds`
- `observation_count`
- `best_detection_confidence`
- `average_detection_confidence`
- `lifecycle_state`
- `completion_reason`
- `tracker_backend`
- `tracker_configuration`
- `searchable`
- `metadata`
- `created_at`
- `updated_at`

Allowed `vehicle_class` values:

- `3WHEELER`
- `BUS`
- `CAR`
- `MOTORCYCLE`
- `TRUCK`
- `UNKNOWN`

Allowed `lifecycle_state` values:

- `TENTATIVE`
- `ACTIVE`
- `TEMPORARILY_LOST`
- `COMPLETED`
- `DISCARDED`

Idempotency keys:

- unique `track_uuid`
- unique `(processing_run_id, camera_id, local_track_id)`

### `analytics.track_observation`

- `id`
- `vehicle_track_id`
- `camera_id`
- `frame_number`
- `observed_at`
- `video_time_seconds`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `center_x`
- `center_y`
- `detection_confidence`
- `tracker_confidence`
- `is_key_observation`
- `metadata`
- `created_at`

Idempotency key:

- unique `(vehicle_track_id, frame_number)`

### `analytics.ai_model`

- `id`
- `model_code`
- `model_name`
- `model_type`
- `provider`
- `model_reference`
- `model_version`
- `checksum`
- `configuration`
- `active`
- `created_at`
- `updated_at`

Idempotency key:

- unique `model_code`

### `analytics.run_model`

- `id`
- `processing_run_id`
- `ai_model_id`
- `stage_name`
- `device`
- `resolved_configuration`
- `created_at`

Idempotency key:

- unique `(processing_run_id, ai_model_id, stage_name)`

### `analytics.processing_error`

- `id`
- `processing_run_id`
- `camera_run_id`
- `vehicle_track_id`
- `processing_job_id`
- `stage_name`
- `worker_name`
- `severity`
- `exception_type`
- `error_code`
- `message`
- `traceback`
- `frame_number`
- `structured_context`
- `resolution_state`
- `created_at`
- `updated_at`

Allowed `severity` values:

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

Allowed `resolution_state` values:

- `OPEN`
- `ACKNOWLEDGED`
- `RESOLVED`
- `IGNORED`

## Repository Method To Table Mapping

1. `camera_repository.upsert_camera(...) -> analytics.camera`
2. `video_source_repository.create_video_source(...) -> analytics.video_source`
3. `processing_run_repository.create_run(...) -> analytics.processing_run`
4. `camera_run_repository.create_camera_run(...) -> analytics.camera_run`
5. `camera_run_repository.update_camera_run(...) -> analytics.camera_run`
6. `processing_job_repository.create_job(...) -> analytics.processing_job`
7. `processing_job_repository.update_job(...) -> analytics.processing_job`
8. `vehicle_track_repository.upsert_vehicle_track(...) -> analytics.vehicle_track`
9. `track_observation_repository.insert_observations_batch(...) -> analytics.track_observation`
10. `model_audit_repository.upsert_ai_model(...) -> analytics.ai_model`
11. `model_audit_repository.create_run_model(...) -> analytics.run_model`
12. `processing_error_repository.create_error(...) -> analytics.processing_error`

## Batch-Write Strategy

1. camera, video_source, processing_run, camera_run, ai_model, run_model, and processing_job are low-volume control rows and may be written one record at a time.
2. `track_observation` must be inserted in batches.
3. current runtime batch size default is `100`; repositories should accept a list payload to avoid one network call per observation.
4. `vehicle_track` writes should be idempotent by `track_uuid`.

## Current Runtime Model Mismatches

1. Current runtime `LocalVehicleTrack.class_name` uses detector/runtime values like `3Wheeler`, `car`, `truck`; SQL requires canonical uppercase values.
2. Current runtime `LocalVehicleTrack.state` uses lowercase lifecycle names; SQL requires uppercase lifecycle values.
3. Current runtime simple `database.models.VehicleTrackRecord` is missing:
   - `processing_run_id`
   - `camera_run_id`
   - `first_video_time_seconds`
   - `last_video_time_seconds`
   - `average_detection_confidence`
   - `lifecycle_state`
   - `completion_reason`
   - `tracker_backend`
   - `tracker_configuration`
   - `searchable`
   - `metadata`
4. Current runtime observation DTO is missing:
   - `camera_id`
   - `video_time_seconds`
   - `center_x`
   - `center_y`
   - `tracker_confidence`
   - `is_key_observation`
   - `metadata`
5. Current runtime has no typed DTOs yet for:
   - processing runs
   - camera runs
   - jobs
   - ai models
   - run models
   - processing errors

## Schema Ambiguities Requiring Decisions

1. `video_source.source_reference` should use relative/project-style paths for local files when possible rather than raw absolute filesystem paths.
2. `processing_job` granularity needs a runtime decision:
   - minimal Phase 1 can record run-level and camera-level lifecycle jobs
   - deeper per-stage timing can be added later without schema changes
3. `vehicle_track.average_detection_confidence` is not directly tracked today and will need either:
   - runtime accumulation, or
   - temporary derivation from observation confidences at persistence time
4. `track_observation.center_x` and `center_y` are not stored today but can be deterministically derived from bbox values.
5. `processing_error` allows nullable foreign keys, so Phase 1 can log run-level errors even when camera-run or track ids are unavailable.

## Current Decision For Phase 1

1. use canonical database vehicle classes from the shared mapping helper
2. uppercase lifecycle values before persistence
3. derive observation centers from bbox coordinates
4. write batched observation rows through a single repository batch call
5. keep old simplified repository untouched as fallback while Phase 1 analytics repositories are validated
