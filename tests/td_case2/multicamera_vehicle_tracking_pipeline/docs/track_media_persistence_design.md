## Track Media Persistence Design

### Backend Routing Audit

Current routing before this dry-run fix was:

- `disabled`
  no persistence service
- `dry_run`
  `TrackingPersistenceService(SimpleVehicleRepository())`
- `old_public`
  `TrackingPersistenceService(SupabaseVehicleRepository(...))`
- `analytics_supabase`
  `AnalyticsPersistenceService(AnalyticsDatabaseClient(...))`

That meant `dry_run` never entered the analytics path at all, so:

- `build_track_media_records(...)` was never called
- `TrackMediaRecord` validation never ran
- `storage_uri` normalization was never checked
- `media_records_attempted` stayed `0`
- `media_persistence` stayed `null`

Required routing after the fix is:

- `disabled`
  no persistence service
- `dry_run`
  `AnalyticsPersistenceService(..., enable_database_writes=False)`
- `old_public`
  existing legacy `TrackingPersistenceService(...)`
- `analytics_supabase`
  `AnalyticsPersistenceService(..., enable_database_writes=True)`

In the fixed `dry_run` mode:

- analytics payload models are built and validated
- deterministic dry-run placeholder references are used
- evidence files remain local under `artifacts/...`
- only portable relative `storage_uri` values are validated
- no Supabase client is initialized
- no repositories are called
- no HTTP requests are made

### Source of Truth

The actual schema comes from:

- [010_track_media.sql](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/010_track_media.sql)
- [analytics_full_schema.sql](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/supabase/analytics_full_schema.sql)

### Exact Table Definition

`analytics.track_media` columns:

- `id uuid primary key default gen_random_uuid()`
- `vehicle_track_id uuid not null`
- `media_type varchar(40) not null`
- `storage_provider varchar(30) not null default 'LOCAL'`
- `storage_uri text not null`
- `thumbnail_uri text null`
- `mime_type varchar(255) null`
- `file_size_bytes bigint null`
- `checksum_sha256 varchar(64) null`
- `frame_number bigint null`
- `captured_at timestamptz null`
- `video_time_seconds numeric null`
- `bbox jsonb null`
- `width integer null`
- `height integer null`
- `quality_score numeric null`
- `sharpness_score numeric null`
- `visibility_score numeric null`
- `occlusion_score numeric null`
- `selection_rank integer null`
- `is_primary boolean not null default false`
- `metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

### Required Columns

- `vehicle_track_id`
- `media_type`
- `storage_provider`
- `storage_uri`
- `is_primary`
- `metadata`

The database supplies defaults for:

- `id`
- `storage_provider`
- `is_primary`
- `metadata`
- `created_at`
- `updated_at`

### Nullable Columns

- `thumbnail_uri`
- `mime_type`
- `file_size_bytes`
- `checksum_sha256`
- `frame_number`
- `captured_at`
- `video_time_seconds`
- `bbox`
- `width`
- `height`
- `quality_score`
- `sharpness_score`
- `visibility_score`
- `occlusion_score`
- `selection_rank`

### Foreign Keys

- `fk_track_media_vehicle_track`
  `vehicle_track_id -> analytics.vehicle_track(id)`

Indirect linkage to processing run, camera, and camera run exists through `analytics.vehicle_track`, which already stores:

- `processing_run_id`
- `camera_run_id`
- `camera_id`

There are no direct `processing_run_id`, `camera_id`, or `camera_run_id` columns on `analytics.track_media`.

### Check Constraints

`media_type` allowed values:

- `FULL_FRAME`
- `VEHICLE_CROP`
- `BEST_VEHICLE_CROP`
- `PLATE_CROP`
- `THUMBNAIL`

`storage_provider` allowed values:

- `LOCAL`
- `NAS`
- `S3`
- `SUPABASE_STORAGE`

Other checks:

- `file_size_bytes >= 0` when present
- `frame_number >= 0` when present
- `width > 0` when present
- `height > 0` when present
- `quality_score` in `[0, 1]` when present
- `sharpness_score` in `[0, 1]` when present
- `visibility_score` in `[0, 1]` when present
- `occlusion_score` in `[0, 1]` when present

### Unique Constraints

There is no unique constraint declared on `analytics.track_media`.

This is important:

- database-level upsert on a true unique key is not available from the schema as written
- idempotency must be handled at the application layer unless the schema is later extended

### Indexes

From [023_indexes.sql](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/023_indexes.sql):

- `idx_track_media_vehicle_track_type`
  on `(vehicle_track_id, media_type, captured_at desc)`

- `idx_track_media_primary`
  on `(vehicle_track_id, selection_rank)`
  where `is_primary = true`

These are performance indexes only, not uniqueness constraints.

### Trigger

From [024_triggers.sql](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/024_triggers.sql):

- `trg_track_media_set_updated_at`
  before update
  executes `analytics.set_updated_at()`

### Path-Related Columns

The path-like columns in the real schema are:

- `storage_uri`
- `thumbnail_uri`

There is no `relative_path` column.

For this task, the portable relative file path must therefore be stored in `storage_uri`.

Absolute machine-local paths such as `F:\...` must not be persisted.

### Metadata Field

`metadata jsonb not null default '{}'::jsonb`

This is the correct place to preserve evidence-role details that do not have first-class columns, such as:

- evidence candidate role
- original candidate type name
- track UUID
- confidence
- bbox
- crop area

### Actual SQL Mismatch With Current Evidence Package

Current evidence package roles:

- `best_overall`
- `first`
- `middle`
- `last`
- `highest_confidence`
- `largest`
- `sharpest`

Actual SQL `media_type` values are more generic:

- `FULL_FRAME`
- `VEHICLE_CROP`
- `BEST_VEHICLE_CROP`
- `PLATE_CROP`
- `THUMBNAIL`

So the evidence role cannot be stored directly in `media_type` for most roles.

Practical mapping implication:

- `best_overall` should map to `BEST_VEHICLE_CROP`
- other evidence roles should map to `VEHICLE_CROP`
- the specific evidence role should be preserved in `metadata`

### BEST_OVERALL Contract

`BEST_OVERALL` is the first required persisted role from the runtime perspective.

The actual SQL-compatible media type for that role is:

- `BEST_VEHICLE_CROP`

This matches the existing view usage in [025_views.sql](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/025_views.sql), where `searchable_vehicle` joins:

- `tm.media_type = 'BEST_VEHICLE_CROP'`
- `tm.is_primary = true`

### Current Evidence Output Contract

Current local evidence writing happens in:

- [track_evidence_collector.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/evidence/track_evidence_collector.py)

Files are written under:

- `Path(self.config.output_root) / run_id / camera_code / track_<local_track_id> / track_uuid_with_colons_replaced`

Example structure:

- `artifacts/RUN_.../CAM_001/track_000004/RUN_..._CAM_001_TRACK_4/best_overall.jpg`

Current `EvidenceCandidate.file_path` is set to:

- `str(target_path)`

Current `TrackEvidencePackage.output_directory` is set to:

- `str(base_dir)`

So both are currently machine-local paths as strings, not portable relative paths.

### Whether `best_overall.jpg` Exists When Metadata Says It Exists

Today:

- when `save_final_selected_crops` is enabled
- `_persist_candidates()` writes each candidate file first
- then updates `candidate.file_path`
- then returns `output_directory`

So if a candidate has `file_path`, the file should exist at the time the evidence package is created.

This still needs verification in the media mapper before persistence.

### How Evidence Is Attached To A Completed Track

Evidence is attached here:

- worker path:
  [tracking_worker.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/tracking_worker.py)
  assigns `track.evidence_package = self.evidence_collector.finalize_track(track)`

- sequential path:
  [multicamera_tracking_orchestrator.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py)
  assigns `completed_track.evidence_package = evidence_collector.finalize_track(completed_track)`

The evidence package is therefore available on `LocalVehicleTrack` before persistence runs.

### Whether Persistence Receives The Evidence Package

Yes.

`PersistenceWorker` forwards the completed `LocalVehicleTrack` object to:

- `persistence_service.save_completed_track(item.track)`

So the analytics persistence service has access to:

- `track.evidence_package`

### Whether Vehicle-Track Database ID Is Available Before Evidence Persistence

Yes, in the analytics path.

Current order inside [analytics_persistence_service.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/persistence/analytics_persistence_service.py):

1. insert/upsert `analytics.vehicle_track`
2. get returned `vehicle_track_id`
3. persist `track_observation`

That is the correct place to insert media persistence immediately afterward.

### Best Invocation Point

Preferred runtime sequence for analytics persistence should be:

1. persist `vehicle_track`
2. obtain `vehicle_track_id`
3. persist `track_observation`
4. inspect `track.evidence_package`
5. build `TrackMediaRecord` rows
6. write `analytics.track_media`

This matches the requested order and avoids persisting media before its parent `vehicle_track` exists.

### Idempotency Reality

The schema does not provide a unique key for `track_media`.

Therefore:

- true SQL `upsert(... on_conflict=...)` cannot rely on a declared unique constraint
- retry-safe behavior must be implemented by checking for an existing row before insert, using an application business key

Given the current schema and required roles, the safest application-level idempotency strategy is:

- for `BEST_VEHICLE_CROP`, match by:
  `vehicle_track_id + media_type + is_primary + storage_uri`

- for non-primary `VEHICLE_CROP` role rows, match by:
  `vehicle_track_id + media_type + storage_uri`

This is not a database constraint. It is an application safeguard necessitated by the current schema.
