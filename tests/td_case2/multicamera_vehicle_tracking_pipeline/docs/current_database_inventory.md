# Current Database Inventory

## Scope

This inventory documents the current database-related implementation inside:

`tests/td_case2/multicamera_vehicle_tracking_pipeline`

Date of audit: `2026-07-23`

Current state summary:

- The active migration is still the simplified proof-of-concept schema in [`database/migrations/simplified_schema.sql`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/simplified_schema.sql).
- The simplified schema uses the `public` schema, not the required independent `analytics` schema.
- Runtime persistence code still writes to simplified tables through [`database/repository.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/repository.py).
- The current runtime class constraint is still the old lowercase set `car`, `bus`, `truck`, `motorcycle`, `unknown`.
- The current persistence layer does not yet support canonical `3WHEELER`.

## Current Schema Objects

### Current tables

Defined in the current migration:

1. `public.cameras`
2. `public.vehicle_tracks`
3. `public.vehicle_attributes`
4. `public.vehicle_observations`
5. `public.vehicle_matches`

Legacy objects also dropped by the reset-style migration, but not recreated:

1. `public.pipeline_errors`
2. `public.processing_jobs`
3. `public.association_decisions`
4. `public.association_candidates`
5. `public.global_vehicle_tracks`
6. `public.global_vehicles`
7. `public.track_evidence`
8. `public.plate_readings`
9. `public.vehicle_track_observations`
10. `public.processing_windows`
11. `public.vms_recordings`
12. `public.stream_sessions`
13. `public.camera_connections`

### Current views

Created:

1. `public.searchable_vehicles`

Dropped if present before reset:

1. `public.searchable_vehicle_tracks`
2. `public.global_vehicle_timeline`
3. `public.pending_association_review`

### Current trigger functions

1. `public.set_updated_at()`
2. `public.validate_vehicle_match_cameras()`

### Current triggers

1. `trg_vehicle_attributes_updated_at` on `public.vehicle_attributes`
2. `trg_vehicle_matches_validate_cameras` on `public.vehicle_matches`

### Current extensions

1. `pgcrypto`
2. `pg_trgm`

## Current Column Inventory

### `public.cameras`

- `id uuid primary key default gen_random_uuid()`
- `camera_code text not null unique`
- `camera_name text`
- `source_path text`
- `enabled boolean not null default true`
- `created_at timestamptz not null default now()`

### `public.vehicle_tracks`

- `id uuid primary key default gen_random_uuid()`
- `track_uuid text not null unique`
- `camera_id uuid not null references public.cameras(id) on delete cascade`
- `local_track_id integer not null`
- `vehicle_class text not null`
- `first_seen_at timestamptz not null`
- `last_seen_at timestamptz not null`
- `first_frame_number integer`
- `last_frame_number integer`
- `observation_count integer not null default 0`
- `best_confidence real`
- `best_frame_path text`
- `best_crop_path text`
- `created_at timestamptz not null default now()`

Current check constraint behavior:

- `vehicle_class in ('car', 'bus', 'truck', 'motorcycle', 'unknown')`
- `last_seen_at >= first_seen_at`
- `first_frame_number is null or first_frame_number >= 0`
- `last_frame_number is null or last_frame_number >= 0`
- `last_frame_number >= first_frame_number` when both exist

### `public.vehicle_attributes`

- `id uuid primary key default gen_random_uuid()`
- `vehicle_track_id uuid not null unique references public.vehicle_tracks(id) on delete cascade`
- `vehicle_colour text`
- `colour_confidence real`
- `plate_text text`
- `plate_pattern text`
- `plate_confidence real`
- `plate_verified boolean not null default false`
- `plate_readings jsonb not null default '[]'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

### `public.vehicle_observations`

- `id bigint generated always as identity primary key`
- `vehicle_track_id uuid not null references public.vehicle_tracks(id) on delete cascade`
- `frame_number integer not null`
- `observed_at timestamptz not null`
- `bbox_x1 real not null`
- `bbox_y1 real not null`
- `bbox_x2 real not null`
- `bbox_y2 real not null`
- `confidence real`
- `created_at timestamptz not null default now()`

Current check constraint behavior:

- `frame_number >= 0`
- `bbox_x2 > bbox_x1`
- `bbox_y2 > bbox_y1`

### `public.vehicle_matches`

- `id uuid primary key default gen_random_uuid()`
- `source_track_id uuid not null references public.vehicle_tracks(id) on delete cascade`
- `candidate_track_id uuid not null references public.vehicle_tracks(id) on delete cascade`
- `plate_similarity real`
- `colour_match boolean not null default false`
- `class_match boolean not null default false`
- `time_gap_seconds real`
- `match_score real`
- `match_status text not null`
- `created_at timestamptz not null default now()`

Current check constraint behavior:

- `match_status in ('confirmed', 'probable', 'ambiguous', 'rejected')`
- `source_track_id <> candidate_track_id`
- trigger-enforced cross-camera validation

## Primary Keys, Foreign Keys, Unique Constraints

### Primary keys

1. `public.cameras(id)`
2. `public.vehicle_tracks(id)`
3. `public.vehicle_attributes(id)`
4. `public.vehicle_observations(id)`
5. `public.vehicle_matches(id)`

### Foreign keys

1. `public.vehicle_tracks.camera_id -> public.cameras.id`
2. `public.vehicle_attributes.vehicle_track_id -> public.vehicle_tracks.id`
3. `public.vehicle_observations.vehicle_track_id -> public.vehicle_tracks.id`
4. `public.vehicle_matches.source_track_id -> public.vehicle_tracks.id`
5. `public.vehicle_matches.candidate_track_id -> public.vehicle_tracks.id`

### Unique constraints

1. `public.cameras.camera_code`
2. `public.vehicle_tracks.track_uuid`
3. `public.vehicle_tracks(camera_id, local_track_id, first_seen_at)`
4. `public.vehicle_attributes.vehicle_track_id`
5. `public.vehicle_matches(source_track_id, candidate_track_id)`

## Current Index Inventory

Defined in the migration:

1. `idx_vehicle_tracks_camera_first_seen` on `public.vehicle_tracks(camera_id, first_seen_at)`
2. `idx_vehicle_tracks_vehicle_class` on `public.vehicle_tracks(vehicle_class)`
3. `idx_vehicle_attributes_plate_text` on `public.vehicle_attributes(plate_text)`
4. `idx_vehicle_attributes_vehicle_colour` on `public.vehicle_attributes(vehicle_colour)`
5. `idx_vehicle_matches_source_track_id` on `public.vehicle_matches(source_track_id)`
6. `idx_vehicle_matches_candidate_track_id` on `public.vehicle_matches(candidate_track_id)`
7. `idx_vehicle_matches_match_score_desc` on `public.vehicle_matches(match_score desc)`
8. `idx_vehicle_observations_track_time` on `public.vehicle_observations(vehicle_track_id, observed_at)`
9. `idx_vehicle_attributes_plate_text_trgm` on `public.vehicle_attributes using gin (plate_text gin_trgm_ops)`

## Current Views

### `public.searchable_vehicles`

Current join shape:

- `public.vehicle_tracks vt`
- `join public.cameras c on c.id = vt.camera_id`
- `left join public.vehicle_attributes va on va.vehicle_track_id = vt.id`

Current output columns:

- `track_id`
- `track_uuid`
- `camera_code`
- `camera_name`
- `vehicle_class`
- `first_seen_at`
- `last_seen_at`
- `vehicle_colour`
- `plate_text`
- `plate_pattern`
- `plate_confidence`
- `best_frame_path`
- `best_crop_path`

## Current Trigger Inventory

### `public.set_updated_at()`

Purpose:

- Maintains `updated_at` on mutable rows.

Used by:

- `trg_vehicle_attributes_updated_at`

### `public.validate_vehicle_match_cameras()`

Purpose:

- Prevents inserting a `vehicle_match` between two tracks from the same camera.

Used by:

- `trg_vehicle_matches_validate_cameras`

## Current RLS Policies

RLS is enabled on:

1. `public.cameras`
2. `public.vehicle_tracks`
3. `public.vehicle_attributes`
4. `public.vehicle_observations`
5. `public.vehicle_matches`

Authenticated read policies:

1. `authenticated_read_cameras`
2. `authenticated_read_vehicle_tracks`
3. `authenticated_read_vehicle_attributes`
4. `authenticated_read_vehicle_observations`
5. `authenticated_read_vehicle_matches`

Service-role full-access policies:

1. `service_role_all_cameras`
2. `service_role_all_vehicle_tracks`
3. `service_role_all_vehicle_attributes`
4. `service_role_all_vehicle_observations`
5. `service_role_all_vehicle_matches`

## Current Row Counts

Live row counts are not yet confirmed in this audit because a verified database connection was not available during document creation.

Current status:

- `.env.example` may contain credentials, but runtime code does not auto-load it.
- The last observed connection check failed before connecting because required environment variables were not loaded into the shell session.
- No database query was executed during this inventory run, so no honest row counts can be claimed yet.

When credentials are loaded, use these SQL queries to capture real counts before any destructive action:

```sql
select 'cameras' as table_name, count(*) as row_count from public.cameras
union all
select 'vehicle_tracks', count(*) from public.vehicle_tracks
union all
select 'vehicle_attributes', count(*) from public.vehicle_attributes
union all
select 'vehicle_observations', count(*) from public.vehicle_observations
union all
select 'vehicle_matches', count(*) from public.vehicle_matches;
```

Also capture object inventory from the live database:

```sql
select schemaname, tablename
from pg_catalog.pg_tables
where schemaname in ('public', 'analytics')
order by schemaname, tablename;

select schemaname, viewname
from pg_catalog.pg_views
where schemaname in ('public', 'analytics')
order by schemaname, viewname;
```

## Current Persistence Code References

### Runtime code still coupled to the simplified schema

[`database/repository.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/repository.py):

- `client.table("cameras")`
- `client.table("vehicle_tracks")`
- `client.table("vehicle_observations")`
- `client.table("vehicle_attributes")`
- `client.table("vehicle_matches")`
- `client.table("searchable_vehicles")`

[`database/client.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/client.py):

- health check still queries `client.table("cameras")`

[`database/config.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/database/config.py):

- default `schema_version` is still `"simplified_schema"`

[`persistence/tracking_persistence_service.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/persistence/tracking_persistence_service.py):

- current supported class set is `{"car", "bus", "truck", "motorcycle", "unknown"}`
- class normalization is not centralized
- `3Wheeler` is not accepted

[`config/persistence.yaml`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/config/persistence.yaml):

- still describes the old simple persistence behavior
- no `database_schema: analytics`
- observation batch size is currently `100`

### Documentation coupled to the old schema

1. [`docs/tracking_to_supabase_mapping.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/tracking_to_supabase_mapping.md)
2. [`docs/simple_database_schema.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/simple_database_schema.md)
3. [`docs/test_data_flow.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/test_data_flow.md)
4. [`README.md`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/README.md)

### Tests coupled to the old schema

1. [`tests/test_schema.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_schema.py)
2. [`scripts/validate_database_schema.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/scripts/validate_database_schema.py)
3. [`tests/test_repository.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_repository.py)
4. [`tests/test_tracking_persistence_service.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_tracking_persistence_service.py)
5. [`tests/test_tracking_orchestrator.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_tracking_orchestrator.py)
6. [`tests/test_persistence_worker.py`](/F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tests/test_persistence_worker.py)

### Demo and helper scripts coupled to the old repository model

1. `scripts/demo_insert_tracks.py`
2. `scripts/seed_test_data.py`
3. `scripts/demo_search.py`
4. `scripts/check_supabase_connection.py`

## Current Vehicle-Class Gap

Current database and code assumption:

- `car`
- `bus`
- `truck`
- `motorcycle`
- `unknown`

Required future canonical values:

- `3WHEELER`
- `BUS`
- `CAR`
- `MOTORCYCLE`
- `TRUCK`
- `UNKNOWN`

Main gaps:

1. `3Wheeler` from the YOLO model is currently rejected by the persistence service.
2. The current SQL constraint omits `3Wheeler` and uses lowercase non-canonical values.
3. No single shared normalization helper exists yet.

## Data That Can Likely Be Migrated

Potentially reusable data if present:

1. `public.cameras`
2. `public.vehicle_tracks`
3. `public.vehicle_attributes`
4. `public.vehicle_observations`
5. `public.vehicle_matches`

Important caveat:

- Old rows cannot be assumed lossless against the new target architecture because the new schema introduces `processing_run`, `camera_run`, `video_source`, `processing_job`, ANPR normalization tables, model-audit tables, and global-identity tables that do not exist in the simplified model.

Likely direct mapping candidates:

1. `public.cameras -> analytics.camera`
2. `public.vehicle_observations -> analytics.track_observation`
3. `public.vehicle_attributes -> analytics.vehicle_attribute` with ANPR details split further later
4. `public.vehicle_matches -> analytics.cross_camera_match`

Rows that require synthetic or inferred migration fields:

1. `public.vehicle_tracks -> analytics.vehicle_track`
2. missing `processing_run_id`
3. missing `camera_run_id`
4. missing `tracker_backend`
5. missing `lifecycle_state`
6. missing canonical vehicle class

## Disposable Test Data Assessment

Based on the current repository structure, the simplified schema appears to be a proof-of-concept and local-development data model.

Evidence:

1. Migration file is labeled destructive and development-oriented.
2. Tests and demos rely on `SimpleVehicleRepository`, an in-memory fake.
3. Documentation describes the current persistence model as a proof-of-concept stage.
4. The prompt itself refers to replacing a simplified analytics schema.

Operational assumption:

- Existing simplified rows should be treated as disposable test or prototype data unless the user explicitly confirms a live dataset must be preserved.

## Destructive Migration Risks

1. The current migration file is itself destructive and drops many objects beyond the five recreated tables.
2. The current reset pattern targets `public`, which is dangerous if run against a non-disposable Supabase project.
3. Runtime code still points at the old tables, so dropping them before repository rewrites will break persistence immediately.
4. Tests currently assert old table names and old SQL markers, so migration without test rewrites will create broad failures.
5. The current class vocabulary is incompatible with the required five-class canonical model.
6. The current connection helper and health check are schema-agnostic but table-name-specific and will still probe `cameras`.

## Old-To-New Mapping

### Table mapping

1. `public.cameras -> analytics.camera`
2. `public.vehicle_tracks -> analytics.vehicle_track`
3. `public.vehicle_observations -> analytics.track_observation`
4. `public.vehicle_attributes -> analytics.vehicle_attribute`
5. `public.vehicle_matches -> analytics.cross_camera_match`
6. `public.searchable_vehicles -> analytics.searchable_vehicle`

### Structural expansion required

The old schema has no equivalents for:

1. `analytics.video_source`
2. `analytics.camera_relation`
3. `analytics.processing_run`
4. `analytics.camera_run`
5. `analytics.processing_job`
6. `analytics.track_media`
7. `analytics.plate_detection`
8. `analytics.plate_reading`
9. `analytics.plate_summary`
10. `analytics.global_vehicle`
11. `analytics.global_vehicle_track`
12. `analytics.ai_model`
13. `analytics.run_model`
14. `analytics.processing_error`
15. `analytics.event_candidate`
16. `analytics.analytics_event`

## Backup Instructions Before Any Reset

Do this before any destructive development reset:

1. Export current simplified tables.
2. Export schema-only definitions.
3. Save row-count evidence.
4. Save policy and trigger definitions.

Example PostgreSQL commands:

```powershell
pg_dump --schema=public --schema-only --file public_schema_before_analytics_migration.sql "<POSTGRES_CONNECTION_STRING>"
pg_dump --schema=public --data-only --table=public.cameras --table=public.vehicle_tracks --table=public.vehicle_attributes --table=public.vehicle_observations --table=public.vehicle_matches --file public_analytics_data_before_migration.sql "<POSTGRES_CONNECTION_STRING>"
```

Example SQL backups:

```sql
create table backup_vehicle_tracks as
select * from public.vehicle_tracks;

create table backup_vehicle_observations as
select * from public.vehicle_observations;
```

If using Supabase-managed SQL tooling instead of `pg_dump`, capture:

1. table DDL
2. view DDL
3. trigger function DDL
4. policies
5. row counts

## Dry-Run Migration Plan

1. Load environment variables into the shell session.
2. Verify database connectivity without changing schema.
3. Query and record real row counts and live object inventory.
4. Back up the simplified `public` tables and definitions.
5. Add the new `analytics` schema migrations without dropping old tables.
6. Add centralized vehicle-class normalization with all five YOLO classes.
7. Rewrite repositories and persistence services to target `analytics`.
8. Update schema validation tests to check the new migration set.
9. Validate pipeline behavior with persistence disabled.
10. Validate dry-run persistence payloads against the new repositories.
11. Validate one-camera, three-camera, and five-camera persistence on a disposable database.
12. Only after successful validation, prepare an explicit manual development reset for the old simplified objects.

## Rollback Plan

If the new analytics migration path fails:

1. Stop the application from using the new analytics repositories.
2. Restore the code path that writes only to the current simplified repository.
3. Drop only newly created `analytics` objects on the disposable environment.
4. Re-import backed-up `public` schema and data if any simplified objects were changed.
5. Re-run the old validation script against `simplified_schema.sql`.

Rollback safety rule:

- No old simplified table should be dropped until the new `analytics` migrations, repositories, tests, and persistence validations all pass on a disposable database.

## Immediate Next Implementation Targets

Safe next changes after this inventory:

1. Add centralized vehicle-class normalization for `3Wheeler`, `bus`, `car`, `motorcycle`, `truck`, and unknown input.
2. Introduce non-destructive `analytics` schema migrations in dependency order.
3. Rewrite persistence to use `analytics` repositories and batch observation writes.
4. Replace all old simplified-schema validation tests and docs.
