# Analytics Persistence Integration Plan

## Scope

This document maps the current runtime persistence path and identifies the exact code locations that must change to move Phase 1 persistence from the old simplified `public` tables to the new `analytics` schema.

Phase 1 target tables:

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

Out of scope for this phase:

1. `analytics.track_media`
2. `analytics.vehicle_attribute`
3. `analytics.plate_detection`
4. `analytics.plate_reading`
5. `analytics.plate_summary`
6. `analytics.cross_camera_match`
7. `analytics.global_vehicle`
8. `analytics.global_vehicle_track`
9. `analytics.event_candidate`
10. `analytics.analytics_event`

## Current Persistence Entry Points

### Worker pipeline entry point

[`orchestration/worker_multicamera_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/worker_multicamera_tracking_orchestrator.py)

Current behavior:

1. loads `PersistenceConfig`
2. when persistence is enabled, builds a repository in `_build_repository()`
3. creates `TrackingPersistenceService(repository, persistence_config)`
4. calls `persistence_service.sync_cameras(camera_configs)` before worker startup
5. passes `persistence_service` into `WorkerSupervisor`

### Non-worker orchestration entry point

[`orchestration/multicamera_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py)

Current behavior:

1. builds `TrackingPersistenceService` when persistence is enabled
2. calls `sync_cameras(camera_configs)` before frame processing
3. calls `save_completed_track(completed_track)` directly during main loop and flush

### Persistence worker entry point

[`workers/persistence_worker.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/persistence_worker.py)

Current behavior:

1. reads `CompletedTrackMessage` from `completed_track_queue`
2. calls `persistence_service.save_completed_track(item.track)`
3. stores `results_by_track_uuid`
4. updates worker metrics based on returned status

## Old Table References

### Runtime Supabase repository still targets simplified `public` tables

[`database/repository.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/repository.py)

Current remote table usage:

1. `.table("cameras")`
2. `.table("vehicle_tracks")`
3. `.table("vehicle_observations")`
4. `.table("vehicle_attributes")`
5. `.table("vehicle_matches")`
6. `.table("searchable_vehicles")`

### Current health check still probes old simplified schema

[`database/client.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/client.py)

Current health-check behavior:

1. queries `.table("cameras")`
2. is not schema-scoped to `analytics`

### Docs and tests still coupled to old table names

Main files:

1. [`docs/tracking_to_supabase_mapping.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/tracking_to_supabase_mapping.md)
2. [`README.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/README.md)
3. [`tests/test_repository.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_repository.py)
4. [`tests/test_schema.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_schema.py)
5. [`tests/test_tracking_persistence_service.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_tracking_persistence_service.py)
6. [`tests/test_persistence_worker.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_persistence_worker.py)

## Current Track And Observation Models

### Runtime tracking models

[`tracking/tracking_models.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/tracking_models.py)

`TrackObservation` currently carries:

1. `camera_code`
2. `local_track_id`
3. `frame_number`
4. `video_time_seconds`
5. `camera_timestamp`
6. `class_name`
7. `confidence`
8. `bbox_xyxy`
9. `track_uuid`
10. `state`

`LocalVehicleTrack` currently carries:

1. `track_uuid`
2. `camera_code`
3. `local_track_id`
4. `class_name`
5. `first_frame_number`
6. `last_frame_number`
7. `first_seen_at`
8. `last_seen_at`
9. `first_video_time_seconds`
10. `last_video_time_seconds`
11. `observation_count`
12. `best_confidence`
13. `state`
14. `observations`
15. `camera_name`
16. `source_path`
17. `lost_frame_count`

### Current simplified persistence DTOs

[`database/models.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/models.py)

Current DTO coverage is still shaped for the old schema:

1. `CameraRecord`
2. `VehicleTrackRecord`
3. `VehicleObservationRecord`
4. `VehicleAttributeRecord`
5. `VehicleMatchRecord`

Gap for Phase 1:

1. no DTO yet for `processing_run`
2. no DTO yet for `camera_run`
3. no DTO yet for `video_source`
4. no DTO yet for `processing_job`
5. no DTO yet for `ai_model`
6. no DTO yet for `run_model`
7. no DTO yet for `processing_error`

## Where Completed Tracks Are Emitted

### Worker path

[`workers/tracking_worker.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/tracking_worker.py)

Completed track emission happens in:

1. `_flush_camera()`
2. `_flush_all()`
3. `_emit_tracks()`

Important behavior:

1. `CompletedTrackMessage` is placed on `completed_track_queue`
2. duplicate `track_uuid` emission is prevented by `_emitted_track_uuids`
3. `completed` and `discarded` states are counted separately

### Non-worker path

[`orchestration/multicamera_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py)

Completed tracks are accumulated from:

1. `tracking_result.completed_tracks` during frame loop
2. `flush_result.completed_tracks` after `router.flush_all()`

## Whether Full Observation History Is Currently Retained

Yes, full observation history is available at runtime on each `LocalVehicleTrack` through `track.observations`.

Current persistence behavior:

1. `TrackingPersistenceService._select_observations()` can persist:
   - all observations
   - sampled observations
   - no observations
2. default config is `observation_mode: all`
3. default batch size is `100`

Current limitation:

1. persistence happens only after the track completes
2. there is no run-level persistence of packets or live per-frame jobs yet

## Where Run Totals Are Finalized

### Worker pipeline totals

[`orchestration/worker_multicamera_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/worker_multicamera_tracking_orchestrator.py)

Run totals are finalized after `supervisor.run()` returns:

1. `total_frames_read`
2. `total_frames_processed`
3. `total_detections`
4. `total_track_observations`
5. `total_completed_tracks`
6. `total_discarded_tracks`
7. per-camera counters
8. worker metrics
9. persistence summary

These are the natural source fields for:

1. `analytics.processing_run`
2. `analytics.camera_run`
3. `analytics.processing_job`

### Non-worker pipeline totals

[`orchestration/multicamera_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py)

Run totals are finalized after:

1. main reader loop
2. `router.flush_all()`
3. report assembly

## Exact Files That Need Modification

### New files to add

1. `persistence/analytics_database_client.py`
2. `persistence/analytics_persistence_models.py`
3. `persistence/analytics_repositories.py`
4. `tests/test_analytics_database_client.py`

### Existing files that will need Phase 1 updates

1. `persistence/tracking_persistence_service.py`
2. `persistence/persistence_models.py`
3. `persistence/persistence_config.py`
4. `config/persistence.yaml`
5. `orchestration/worker_multicamera_tracking_orchestrator.py`
6. `orchestration/multicamera_tracking_orchestrator.py`
7. `workers/persistence_worker.py`

### Existing files that should remain as fallback during migration

1. `database/repository.py`
2. `database/models.py`
3. `database/client.py`

Reason:

1. old simplified persistence must remain available until analytics persistence is validated
2. existing tests rely on `SimpleVehicleRepository`
3. persistence-disabled behavior must remain unchanged

## Phase 1 Integration Strategy

### New persistence flow

1. initialize schema-scoped `AnalyticsDatabaseClient`
2. create one `processing_run` at orchestrator start
3. synchronize selected cameras into `analytics.camera`
4. create one `video_source` per selected camera
5. create one `camera_run` per selected camera
6. register active detector and tracker in `analytics.ai_model`
7. create `run_model` rows for detector and tracker stages
8. on completed track:
   - normalize canonical vehicle class
   - upsert `analytics.vehicle_track` by `track_uuid`
   - bulk insert `analytics.track_observation`
9. on errors:
   - write `analytics.processing_error`
10. on orchestrator finish:
   - finalize `analytics.camera_run`
   - finalize `analytics.processing_run`

### Required new state not currently tracked inside the persistence service

1. `processing_run_id`
2. `camera_run_id` by `camera_code`
3. `video_source_id` by `camera_code`
4. registered `ai_model` ids
5. `run_model` ids or dedupe keys
6. per-camera final counters for database finalization

## Immediate Implementation Priority

1. add schema-scoped analytics client
2. keep secrets environment-only
3. keep persistence disabled by default
4. avoid any writes to `public` from the new code path
5. preserve batched observation insertion
6. leave the old simplified repository path available until tests pass
