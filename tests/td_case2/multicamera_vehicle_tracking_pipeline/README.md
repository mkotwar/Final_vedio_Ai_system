# Multi-Camera Vehicle Tracking Pipeline

This folder is a deliberately small proof-of-concept pipeline for testing multi-camera input, shared vehicle detection, independent local ByteTrack vehicle tracks, and opt-in persistence into the simplified Supabase schema. It does not modify `tests/td_case2/streaming_tracking_pipeline`.

## Current pipeline purpose

The current implementation supports this staged flow:

```text
Multiple Camera Videos
        ->
Shared Vehicle Detection
        ->
Independent ByteTrack per Camera
        ->
Completed Local Vehicle Tracks
        ->
TrackingPersistenceService
        ->
Supabase vehicle_tracks
        ->
Supabase vehicle_observations
```

Persistence is optional and disabled by default.

## Current scope

- Multiple local camera videos
- Shared YOLO vehicle detection across cameras
- One isolated ByteTrack instance per camera
- Per-camera local track observations
- Lifecycle-managed completed and discarded local vehicle tracks
- Opt-in persistence of finalized tracks into `vehicle_tracks`
- Batched persistence of selected observations into `vehicle_observations`

## Supported input type

- Local video files only

## Detection configuration

Detection config lives in [config/detection.yaml](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/config/detection.yaml).

## Tracking configuration

Tracking config lives in [config/tracking.yaml](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/config/tracking.yaml).

## Persistence configuration

Persistence config lives in [config/persistence.yaml](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/config/persistence.yaml).

Default persistence config:

```yaml
persistence:
  enabled: false
  sync_cameras: true
  write_completed_tracks_only: true
  include_discarded_tracks: false
  observation_mode: all
  observation_batch_size: 100
  observation_sample_every_n: 5
  dry_run: false
  fail_on_database_error: true
```

Observation modes:

- `all`: store every accepted observation
- `sampled`: store first, last, and every Nth observation
- `none`: store only the `vehicle_tracks` row

## Persistence rules

- Persistence is opt-in.
- `--persist-to-supabase` is required for real writes.
- `discarded` tracks are skipped by default.
- Only `completed` tracks are written by default.
- Active, tentative, and temporarily lost tracks are not written.
- Observations are written in batches.
- Duplicate runs reuse `track_uuid` as the deduplication key.
- No `vehicle_attributes` rows are inserted in this stage.

## Environment variables

Required for trusted backend writes:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

This pipeline also accepts the exact local key name:

```env
supabase_database_url=
```

If `supabase_database_url` is provided as a Supabase Postgres URL, the client derives the matching Supabase project HTTP URL automatically.

## Documentation

- [simple_database_schema.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/simple_database_schema.md)
- [test_data_flow.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/test_data_flow.md)
- [supabase_setup.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/supabase_setup.md)
- [existing_vehicle_detector_mapping.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/existing_vehicle_detector_mapping.md)
- [existing_bytetrack_mapping.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/existing_bytetrack_mapping.md)
- [tracking_to_supabase_mapping.md](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/tracking_to_supabase_mapping.md)

## Validation and test commands

Unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests\td_case2\multicamera_vehicle_tracking_pipeline\tests -p "test_*.py"
```

Tracking validation without persistence:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_multicamera_tracking --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml --mode round_robin --max-frames-per-camera 100
```

Dry-run persistence:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_multicamera_tracking --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml --persistence-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\persistence.yaml --mode round_robin --max-frames-per-camera 100 --dry-run-persistence
```

Real Supabase writes:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_multicamera_tracking --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml --persistence-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\persistence.yaml --mode round_robin --max-frames-per-camera 100 --persist-to-supabase
```

Verification read-back:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_persisted_tracks --camera CAM_001 --limit 20
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_persisted_tracks --camera CAM_002 --limit 20
```

Direct track lookup:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_persisted_tracks --track-uuid RUN_20260722_175158:CAM_001:TRACK_1
```

## Validation report location

Tracking validation output:

```text
debug_runs/multicamera_vehicle_tracking_pipeline/tracking_validation_<timestamp>/report.json
```

## Known limitations

- local video only
- persistence depends on the target Supabase project having the simplified schema available via PostgREST
- no `vehicle_attributes` writes in this stage
- no best-frame selection
- no best-crop selection
- no plate OCR
- no colour enrichment
- no object search
- no cross-camera matching
- no RTSP
- no VMS integration
- no parallelism
- no GPU batching

## Next planned stage

Add per-track evidence collection and best-frame/best-crop selection before plate OCR and vehicle-colour enrichment.
