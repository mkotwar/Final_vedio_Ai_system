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

Optional:

```env
API_HOST=127.0.0.1
API_PORT=8000
API_CORS_ORIGINS=http://localhost:5173
API_PAGE_SIZE_DEFAULT=25
API_PAGE_SIZE_MAX=100
API_LOG_LEVEL=INFO
```

Startup logging reports only whether credentials are `SET` or `MISSING`.

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

## Filters and sorting

- Runs: `status`, `run_code`, `sort_by`, `sort_order`
- Cameras: `status`, `camera_code`, `sort_by`, `sort_order`
- Tracks: `camera_code`, `vehicle_class`, `colour`, `plate`, `plate_status`, `lifecycle_state`, `minimum_confidence`, `has_media`, `sort_by`, `sort_order`
- Global vehicles: `run_code`, `status`, `vehicle_class`, `colour`, `plate`, `minimum_confidence`, `minimum_camera_count`, `sort_by`, `sort_order`
- Cross-camera matches: `run_code`, `decision`, `rule_version`, `minimum_score`, `camera_code`, `sort_by`, `sort_order`

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

## Media behavior

Track media endpoints return safe metadata only.

`GET /api/v1/media/{media_id}` currently returns `REFERENCE_ONLY` metadata when the stored reference is safe. Unsafe local references such as path traversal or absolute local paths are rejected.

## Frontend contract

The frontend should use this API instead of direct Supabase service-role access. Current next step for the React frontend is wiring run, track, match, and global-vehicle list/detail screens to these read-only endpoints.
