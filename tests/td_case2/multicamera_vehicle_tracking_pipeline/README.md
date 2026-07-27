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

The read-only FastAPI backend under `api/` now auto-loads these values from `tests\td_case2\multicamera_vehicle_tracking_pipeline\.env.example` for local runs.

If you also want these values loaded into the current PowerShell session, run:

```powershell
. .\load-env.ps1
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

Enhanced ANPR artifact validation:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_anpr_on_existing_run `
  --run-code RUN_20260727_112517 `
  --track-uuid RUN_20260727_112517:CAM_003:TRACK_2 `
  --artifact-root artifacts `
  --anpr-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\anpr.yaml `
  --florence-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\florence.yaml `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\anpr_track2_validation.json
```

Enhanced ANPR behavior:

- ANPR now examines multiple saved evidence roles per completed track instead of only one crop.
- Plate YOLO runs on the original saved vehicle crop and a padded vehicle-crop variant, and the collector can fall back to class-aware heuristic plate regions when detector boxes are absent.
- OCR now evaluates bounded preprocessing variants per candidate and aggregates repeated text across saved evidence roles before selecting the final result.
- Runtime track-level statuses can now surface `VERIFIED`, `PARTIAL`, `UNREADABLE`, `NO_PLATE_DETECTED`, and `CONFLICTING_CANDIDATES`.
- Persistence remains backward compatible by mapping richer runtime states into the existing analytics plate-reading schema while keeping full diagnostics in metadata during dry-run validation.

Enrichment-run audit:

```powershell
$env:SUPABASE_URL = "https://<project-ref>.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"

.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402
```

Enrichment-run audit with JSON output:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_audit.json"
```

Strict enrichment-run audit:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --strict
```

Camera-level enrichment audit:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --camera-code CAM_001
```

Track-level enrichment audit:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --track-uuid "RUN_20260724_151402:CAM_001:TRACK_4"
```

Track-level enrichment audit with JSON output:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --track-uuid "RUN_20260724_151402:CAM_001:TRACK_4" `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\TRACK_4_audit.json"
```

Notes:

- `verify_enrichment_run.py` remains read-only and only reports candidate hints.
- `build_global_vehicle_objects.py` is the stage that persists auditable cross-camera matches and global vehicle objects.
- The verifier stays read-only and does not fetch image bytes.

Global-object build dry run:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.build_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --global-match-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\global_matching.yaml `
  --dry-run `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_match_dry_run.json"
```

Global-object persistence:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.build_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --global-match-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\global_matching.yaml `
  --persist `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_match_persisted.json"
```

Global-object verification:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --strict `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_object_audit.json"
```

Further documentation:

- [docs/GLOBAL_VEHICLE_MATCHING.md](/C:/Mukul%20K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/GLOBAL_VEHICLE_MATCHING.md)
- [docs/API.md](/C:/Mukul%20K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/docs/API.md)

## Read-only backend API

The pipeline now includes a read-only FastAPI backend under `api/` that reuses the existing `AnalyticsDatabaseClient` and a schema-scoped read repository.

Current API scope:

- run health and status
- processing-run list and detail
- camera list and detail inside a run
- local track list, detail, observations, and media references
- cross-camera match list and detail
- global vehicle list, detail, and memberships
- safe media-reference lookup

Run the backend locally with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  tests.td_case2.multicamera_vehicle_tracking_pipeline.api.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

Docs:

- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

Notes:

- The API is read-only in this stage.
- The API uses the `analytics` schema only.
- Sensitive metadata keys such as keys, tokens, and local model paths are stripped from responses.
- Local evidence media can now be streamed safely through FastAPI when the stored `track_media.storage_uri` resolves inside an approved root such as `artifacts/`.
- Track detail now exposes structured persisted global membership so the frontend can link directly to a confirmed global vehicle instead of inferring membership client-side.
- The dedicated media endpoints are:
  - `GET /api/v1/media/{media_id}`
  - `GET /api/v1/media/{media_id}/content`
  - `GET /api/v1/media/{media_id}/url`
- Unsafe paths, traversal attempts, unsupported extensions, and unsupported providers are blocked without exposing absolute local filesystem paths in responses.
- Supabase Storage signed URLs remain optional and are only returned when a bucket is configured server-side.

## React frontend

The pipeline now includes a React + TypeScript + Vite frontend under `frontend/`.

Current frontend scope:

- dashboard health and recent runs
- processing-run list and detail views
- local-track list and detail views
- global-vehicle list and detail views
- cross-camera match list view
- structured vehicle search route at `/search`
- natural-language vehicle search layered onto the same `/search` route and backend search service
- loading, empty, retry, and typed API error states
- URL-driven filters and pagination for list screens
- reference-only evidence cards without direct Supabase access

Run the frontend locally from `tests\td_case2\multicamera_vehicle_tracking_pipeline\frontend`:

```powershell
npm install
npm run dev
```

Environment:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

Notes:

- The frontend calls FastAPI only.
- Supabase service-role credentials must remain server-side.
- The frontend now renders live local evidence images for available `LOCAL_FILE` media and shows placeholders for `REFERENCE_ONLY`, `MISSING`, `UNSAFE_REFERENCE`, and `UNSUPPORTED_PROVIDER`.
- Track detail now shows the linked `global_vehicle_code` and `Open Global Vehicle` action whenever the API returns a persisted membership; only unlinked tracks render `Not linked`.
- Evidence cards use a fixed `240px` preview viewport with `object-fit: contain`, keeping both vehicle and plate crops fully visible while preserving aspect ratio.
- The React UI now groups vehicle and plate evidence per track through a shared `VehicleIdentityCard`, so the plate crop stays visually attached to its parent vehicle in Track Detail, Global Vehicle Detail, Cross-Camera Matches, Vehicle Search, and list summaries.
- List-oriented API responses now expose lightweight `primary_vehicle_media` and `primary_plate_media` fields so tracks, global vehicles, run detail tabs, and search results can render representative thumbnails without loading full evidence galleries.
- The frontend includes focused Vitest coverage for API parsing, list/detail rendering, status handling, evidence rendering, placeholder behavior, and source-code checks that block direct Supabase usage.
- The new structured vehicle search page reuses the same backend-only contract and does not call Supabase directly from the browser.
- The same `/search` page now also supports natural-language vehicle queries that post to `POST /api/v1/search/natural-language`.
- The backend validates and normalizes interpreted filters, then reuses the existing structured `VehicleSearchService` instead of generating SQL or allowing direct LLM/database access.
- `POST /api/v1/search/natural-language/parse` is available for safe parser-only validation and does not hit the repository search methods.
- Supported natural-language phrases include exact/full plates, plate starts/ends/contains forms, class words, colour words, `both cameras`, `verified plates`, and common time phrases such as `around 2 PM` or `between 2 PM and 3 PM`.
- Explicit run and scope selections remain authoritative. The parser never invents another run, never searches all history when a selected run already exists, and validates parsed camera codes against the selected run before search execution.
- The interpreted-filters panel shows the backend-validated translation plus fallback status and lets the operator apply those values back into the structured search form.

## Structured vehicle search

Read-only backend route:

- `GET /api/v1/search/vehicles`

Frontend route:

- `http://127.0.0.1:5173/search`

Supported structured filters:

- `run_code`
- `result_scope`
- `vehicle_class`
- `colour`
- `plate`
- `plate_match_type`
- `camera_codes`
- `date`
- `start_time`
- `end_time`
- `minimum_confidence`
- `multi_camera_only`
- `verified_plate_only`
- `limit`
- `offset`
- `sort_by`
- `sort_order`

Natural-language layer:

- `POST /api/v1/search/natural-language`
- `POST /api/v1/search/natural-language/parse`
- returns interpreted filters, parser metadata, clarification state, pagination, and the same shared result-card data contract
- reuses existing local-track and global-vehicle detail drill-down routes

Response characteristics:

- Results can mix local tracks and global vehicles in one ranked response.
- Pagination uses `limit`, `offset`, `returned`, `total`, and `has_more`.
- Relevance is deterministic and based on structured match reasons only.
- Existing track and global-vehicle detail routes remain the drill-down targets for search cards.

Live evidence validation on July 27, 2026:

- `RUN_20260725_131944:CAM_002:TRACK_4` returned both `BEST_VEHICLE_CROP` and `PLATE_CROP` through FastAPI with `LOCAL_FILE` availability.
- The same track rendered `GVO:RUN_20260725_131944:FA3FCF9E3ABC` and `Open Global Vehicle` in the live Track Detail page.
- The confirmed global vehicle `GVO:RUN_20260725_131944:FA3FCF9E3ABC` rendered evidence from both `CAM_001 TRACK_4` and `CAM_002 TRACK_4`.
- The grouped vehicle card kept plate `DL8CBF6268` beside its owning vehicle crop instead of rendering plate evidence as a disconnected card.
- Browser evidence requests were served from FastAPI content routes under `http://127.0.0.1:8000/api/v1/media/.../content` during validation.

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
- no RTSP
- no VMS integration
- no parallelism
- no GPU batching

## Next planned stage

Add per-track evidence collection and best-frame/best-crop selection before plate OCR and vehicle-colour enrichment.
