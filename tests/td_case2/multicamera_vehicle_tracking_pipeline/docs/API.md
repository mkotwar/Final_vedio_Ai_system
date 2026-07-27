# Read-Only FastAPI API

This API is the first production-safe backend layer between the multicamera frontend and the `analytics` Supabase schema.

## Architecture

```text
Frontend
    ->
FastAPI routers
    ->
Read-only services
    ->
AnalyticsReadRepository
    ->
AnalyticsDatabaseClient
    ->
Supabase analytics schema
```

The API is read-only in this stage. It does not create, update, or delete analytics rows.

## Environment variables

Required:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

For local development, the API settings now auto-load these values from `tests\td_case2\multicamera_vehicle_tracking_pipeline\.env.example`.

Optional:

```env
API_HOST=127.0.0.1
API_PORT=8000
API_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
API_PAGE_SIZE_DEFAULT=25
API_PAGE_SIZE_MAX=100
API_LOG_LEVEL=INFO
API_MEDIA_MODE=auto
API_MEDIA_ALLOWED_ROOTS=artifacts,debug_runs/multicamera_vehicle_tracking_pipeline,debug_runs
API_MEDIA_URL_TTL_SECONDS=300
SUPABASE_MEDIA_BUCKET=
```

Startup logging reports only whether credentials are `SET` or `MISSING`.
Media startup logging reports the configured mode, root count, existing-root count, TTL, and whether a media bucket is configured. It does not print absolute evidence paths or secret values.

## Run command

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  tests.td_case2.multicamera_vehicle_tracking_pipeline.api.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

Swagger:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Endpoints

- `GET /api/v1/health`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_code}`
- `GET /api/v1/runs/{run_code}/cameras`
- `GET /api/v1/runs/{run_code}/cameras/{camera_code}`
- `GET /api/v1/runs/{run_code}/tracks`
- `GET /api/v1/tracks/{track_uuid}`
- `GET /api/v1/tracks/{track_uuid}/observations`
- `GET /api/v1/tracks/{track_uuid}/media`
- `GET /api/v1/global-vehicles`
- `GET /api/v1/global-vehicles/{global_vehicle_code}`
- `GET /api/v1/global-vehicles/{global_vehicle_code}/tracks`
- `GET /api/v1/cross-camera-matches`
- `GET /api/v1/cross-camera-matches/{match_id}`
- `GET /api/v1/media/{media_id}`
- `GET /api/v1/media/{media_id}/content`
- `GET /api/v1/media/{media_id}/url`
- `GET /api/v1/search/vehicles`
- `POST /api/v1/search/natural-language`
- `POST /api/v1/search/natural-language/parse`

Track detail notes:

- `GET /api/v1/tracks/{track_uuid}` now returns a structured `global_membership` object.
- `linked` is based on a persisted `global_vehicle_track` membership, not inferred from plate or colour similarity.
- The response includes the linked `global_vehicle_code` so the frontend can navigate directly to the existing global-vehicle detail route.

ANPR validation and evidence notes:

- The worker-side ANPR enrichment now examines multiple saved evidence roles, including `best_overall`, `highest_confidence`, `largest`, `sharpest`, `first`, `middle`, and `last`.
- Plate detection runs on the saved vehicle crop plus a padded vehicle-crop variant, and may emit heuristic plate-region candidates for configured classes such as `3WHEELER`.
- OCR evaluates bounded preprocessing variants per candidate and aggregates repeated normalized text before final classification.
- Dry-run validation reports may contain runtime statuses `VERIFIED`, `PARTIAL`, `UNREADABLE`, `NO_PLATE_DETECTED`, or `CONFLICTING_CANDIDATES` even though persisted analytics rows remain mapped into the existing `VERIFIED` / `PROBABLE` / `PARTIAL` / `UNKNOWN` schema.
- Decorative vehicle text such as `STOP`, `CNG`, and `Keep Distance` is intentionally left unverified.

## Filters and sorting

- Runs: `status`, `run_code`, `sort_by`, `sort_order`
- Cameras: `status`, `camera_code`, `sort_by`, `sort_order`
- Tracks: `camera_code`, `vehicle_class`, `colour`, `plate`, `plate_status`, `lifecycle_state`, `minimum_confidence`, `has_media`, `sort_by`, `sort_order`
- Global vehicles: `run_code`, `status`, `vehicle_class`, `colour`, `plate`, `minimum_confidence`, `minimum_camera_count`, `sort_by`, `sort_order`
- Cross-camera matches: `run_code`, `decision`, `rule_version`, `minimum_score`, `camera_code`, `sort_by`, `sort_order`
- Structured vehicle search: `run_code`, `result_scope`, `vehicle_class`, `colour`, `plate`, `plate_match_type`, `camera_codes`, `date`, `start_time`, `end_time`, `minimum_confidence`, `multi_camera_only`, `verified_plate_only`, `limit`, `offset`, `sort_by`, `sort_order`
- Natural-language vehicle search body: `query`, `run_code`, `result_scope`, `default_time_tolerance_minutes`, `limit`, `offset`

Pagination uses:

```json
{
  "items": [],
  "page": 1,
  "page_size": 25,
  "total": 0,
  "has_next": false
}
```

Structured vehicle search uses:

```json
{
  "filters": {
    "run_code": "RUN_20260725_131944",
    "result_scope": "GLOBAL_VEHICLES",
    "vehicle_class": "CAR",
    "colour": "GREY",
    "plate": "6268",
    "plate_match_type": "ENDS_WITH",
    "camera_codes": ["CAM_001", "CAM_002"],
    "multi_camera_only": true
  },
  "pagination": {
    "limit": 25,
    "offset": 0,
    "returned": 1,
    "total": 1,
    "has_more": false
  },
  "results": []
}
```

Search notes:

- The endpoint is read-only and validates all query parameters through Pydantic.
- Empty strings are normalized away before filtering.
- Plate text is normalized with the existing India ANPR cleanup logic before matching.
- `camera_codes` is a safe comma-separated list; arbitrary SQL fragments and sort fields are rejected.
- `LOCAL_TRACKS` returns local track results, `GLOBAL_VEHICLES` returns global objects, and `ALL` merges both result types into one ranked response.
- Relevance ranking is deterministic and based on structured signals only: plate match strength, class, colour, requested cameras, time overlap, confidence, and evidence availability.

## Natural-language vehicle search

Natural-language search is read-only and reuses the existing structured `VehicleSearchQuery` and `VehicleSearchService`.

Routes:

- `POST /api/v1/search/natural-language`
- `POST /api/v1/search/natural-language/parse`

Behavior:

- The parser endpoint never queries the database.
- The search endpoint converts validated natural-language intent into the existing structured search contract, then calls the existing structured search service.
- Explicit `run_code` and `result_scope` override parsed values.
- The parser never invents a different run code.
- Time-only phrases without a date resolve against the selected run's own run date, not the current day.
- If a query needs time interpretation across runs and no run is selected, the API returns clarification instead of searching all history.
- Parsed `camera_codes` are validated against cameras available for the selected run.
- Unknown camera codes return a safe validation or clarification outcome and do not reach the repository search methods.
- Invalid provider output is rejected through Pydantic validation before search execution.
- Deterministic fallback handles common plate, class, colour, camera, multi-camera, verified-plate, and time phrases when the configured provider is unavailable or invalid.

Supported fallback phrases include:

- full plate-like text
- `plate ending in 6268`
- `plate starts with DL8`
- `plate contains CBF`
- class words such as `car`, `bus`, `truck`, `motorcycle`
- supported colour words such as `grey`
- `verified plates`
- `both cameras`, `multiple cameras`
- `around 2 PM`
- `between 2 PM and 3 PM`
- `after 1:30 PM`
- `before 3 PM`
- `same vehicle across CAM_001 and CAM_002`

Natural-language response shape:

```json
{
  "original_query": "Find the grey car with plate ending in 6268.",
  "parser": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "fallback_used": false
  },
  "clarification_required": false,
  "clarification_message": null,
  "interpreted_filters": {
    "run_code": "RUN_20260725_131944",
    "result_scope": "GLOBAL_VEHICLES",
    "vehicle_class": "CAR",
    "colour": "GREY",
    "plate": "6268",
    "plate_match_type": "ENDS_WITH",
    "camera_codes": ["CAM_001", "CAM_002"],
    "multi_camera_only": true
  },
  "pagination": {
    "limit": 25,
    "offset": 0,
    "returned": 1,
    "total": 1,
    "has_more": false
  },
  "results": []
}
```

Provider configuration:

```env
NATURAL_LANGUAGE_SEARCH_ENABLED=true
NATURAL_LANGUAGE_SEARCH_PROVIDER=gemini
NATURAL_LANGUAGE_SEARCH_MODEL=gemini-2.5-flash
NATURAL_LANGUAGE_SEARCH_TIMEOUT_SECONDS=20
NATURAL_LANGUAGE_SEARCH_MAX_RETRIES=2
GEMINI_API_KEY=
```

Security boundaries:

- The frontend never calls Gemini directly.
- Provider keys remain backend-only.
- The API never exposes raw provider prompts, raw provider JSON, SQL, credentials, or filesystem paths.
- Clarification-required responses prevent database execution.

## Errors

Errors use:

```json
{
  "error": {
    "code": "TRACK_NOT_FOUND",
    "message": "Track was not found.",
    "details": null
  }
}
```

Database exceptions are masked. The API does not expose stack traces or raw Supabase errors.

## Security rules

- Service-role credentials remain server-side only.
- Responses sanitize nested metadata recursively.
- Known sensitive keys are removed:
  - `service_role_key`
  - `api_key`
  - `token`
  - `authorization`
  - `model_path`
  - `processor_path`
  - `adapter_path`
  - `local_absolute_path`
- CORS allows configured frontend origins only.
- No image bytes or base64 payloads are returned by list endpoints.
- Wildcard CORS origins are rejected.
- Local media is served only when the resolved file remains inside an approved root after normalization and symlink resolution.
- Path traversal, unsafe absolute paths, directories, unsupported extensions, and unsupported storage providers are blocked without exposing local filesystem paths in JSON responses.

## Media behavior

Track and global-vehicle responses now include safe media metadata only. Image bytes are delivered only through the dedicated content route.

List and detail media fields:

- `primary_vehicle_media` exposes the representative `BEST_VEHICLE_CROP` or `VEHICLE_CROP` for a track or global vehicle when available.
- `primary_plate_media` exposes the representative `PLATE_CROP` for the same owning track when available.
- Detail responses continue to expose the complete `media` / `evidence` arrays, while list-style responses stay lightweight and avoid returning every image record.
- Media references never include absolute local filesystem paths or raw `storage_uri` values.

Delivery modes:

- `LOCAL_FILE`: FastAPI resolves a safe project-local file and exposes `/api/v1/media/{media_id}/content`
- `SIGNED_URL`: reserved for Supabase Storage objects when a server-side bucket mapping is configured
- `REFERENCE_ONLY`: safe metadata only, no directly deliverable content
- `MISSING`: stored reference is safe but the file does not exist
- `UNSAFE_REFERENCE`: stored reference failed safety checks
- `UNSUPPORTED_PROVIDER`: storage provider is not currently supported for browser delivery

`GET /api/v1/media/{media_id}` returns metadata like:

```json
{
  "media_id": "bd043d85-8be5-464e-96e9-108f19499f87",
  "availability": "LOCAL_FILE",
  "media_type": "BEST_VEHICLE_CROP",
  "content_url": "/api/v1/media/bd043d85-8be5-464e-96e9-108f19499f87/content",
  "thumbnail_url": null,
  "frame_number": 73,
  "width": 764,
  "height": 573,
  "quality_score": 0.8794354796409607,
  "selection_rank": 1,
  "is_primary": true,
  "error_detail": null
}
```

`GET /api/v1/media/{media_id}/content` streams the file with `FileResponse`, conservative cache headers, and `X-Content-Type-Options: nosniff`.

`GET /api/v1/media/{media_id}/url` returns a short-lived signed URL only when `SUPABASE_MEDIA_BUCKET` is configured and the stored provider is `SUPABASE_STORAGE`. Otherwise it degrades safely to `REFERENCE_ONLY`.

Current local evidence root:

- `artifacts/`

Verified live media on July 27, 2026:

- `bd043d85-8be5-464e-96e9-108f19499f87` `BEST_VEHICLE_CROP`
- `8971f0a1-28e4-47aa-9a2f-487b92a52753` `PLATE_CROP`

Both records from `RUN_20260725_131944:CAM_002:TRACK_4` returned `LOCAL_FILE` metadata and `200 image/jpeg` content responses with `Access-Control-Allow-Origin: http://127.0.0.1:5173`.

## Frontend contract

The React frontend under `tests/td_case2/multicamera_vehicle_tracking_pipeline/frontend` uses this API instead of direct Supabase service-role access.

Current frontend coverage:

- dashboard health and recent runs
- run list and run detail
- local track list and track detail
- global vehicle list and detail
- cross-camera match list
- retry, loading, empty, timeout, and API error handling
- evidence cards with live local vehicle and plate crop rendering, plus safe placeholders for missing or reference-only media

Frontend rules:

- Browser requests must go to `http://127.0.0.1:8000/api/v1` or the configured `VITE_API_BASE_URL`
- The frontend must not import or initialize a Supabase client
- Service-role credentials must not appear in frontend source or environment examples
- Browser-deliverable media must be fetched from FastAPI media routes or short-lived signed URLs only
- Track detail treats `global_membership.linked=false` as `Not linked`, but a populated linked membership renders the global vehicle code and `Open Global Vehicle` action instead of collapsing into an unlinked state
- Evidence cards use a fixed `240px` preview viewport with `object-fit: contain` and a larger modal preview so vehicle and plate crops remain fully visible without cropping
- The frontend groups vehicle and plate evidence per owning track through a shared `VehicleIdentityCard`, so list and detail pages do not present plate crops as unrelated standalone cards
- Cross-camera match comparisons now render source and candidate vehicles side by side using the same grouped card contract
