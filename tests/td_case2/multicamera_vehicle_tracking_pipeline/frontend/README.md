# Multicamera Vehicle Tracking Frontend

This React frontend is the read-only operator UI for `tests/td_case2/multicamera_vehicle_tracking_pipeline`.

## Architecture

```text
Supabase analytics schema
        ^
        |
     FastAPI
        ^
        |
React + TypeScript + Vite
```

The browser talks to FastAPI only. It does not connect to Supabase directly and it does not contain service-role credentials.

## Stack

- React 19
- TypeScript
- Vite
- React Router
- TanStack Query
- Typed `fetch` client
- Vitest + React Testing Library

## Environment

Create a local `.env` file if needed:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

If `VITE_API_BASE_URL` is omitted, the frontend falls back to `http://127.0.0.1:8000/api/v1`.

## Install

```powershell
npm install
```

## Run

Start the backend first:

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  tests.td_case2.multicamera_vehicle_tracking_pipeline.api.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

Then start the frontend from this folder:

```powershell
npm run dev
```

Default frontend URL:

- `http://localhost:5173`

## Pages

- Dashboard
- Runs
- Run Detail
- Tracks
- Track Detail
- Global Vehicles
- Global Vehicle Detail
- Cross-Camera Matches
- Vehicle Search

## Supported filters

- Runs: status, run code, sorting, pagination
- Tracks: run, camera, class, colour, plate, plate status, lifecycle, confidence, media flag, pagination
- Global vehicles: run, status, class, colour, plate, confidence, camera count, pagination
- Matches: run, decision, score, rule version, camera, pagination
- Vehicle search: run, result scope, class, colour, plate, plate-match mode, multi-camera camera selection, date, time window, confidence, verified-plate-only flag, multi-camera-only flag, sorting, offset, limit
- Natural-language vehicle search: operator query text plus selected run, scope, pagination, and interpreted-filter handoff back into the structured form

## Evidence and media behavior

- The UI reads media metadata from FastAPI only
- `LOCAL_FILE` media renders real vehicle and plate crops through the FastAPI `/media/{media_id}/content` route
- `SIGNED_URL` media is fetched lazily through the FastAPI `/media/{media_id}/url` route
- `REFERENCE_ONLY`, `MISSING`, `UNSAFE_REFERENCE`, and `UNSUPPORTED_PROVIDER` render a safe placeholder instead of a broken image
- Absolute local filesystem paths are never rendered in the browser
- The evidence cards show media type, primary badge, frame number, quality score, selection rank, and dimensions
- Evidence cards use one shared `240px` preview viewport with `object-fit: contain`, so both vehicle crops and narrow plate crops stay fully visible without stretching or accidental cropping
- Clicking an available crop opens a modal preview that stays inside the browser viewport, preserves aspect ratio, and can be closed with the button or `Escape`
- `VehicleIdentityCard` groups the representative vehicle crop, plate crop, plate text, verification badge, class, colour, timing, and membership into one reusable operator-facing card
- Track Detail now leads with a `Vehicle Identity` section and moves non-primary images into `Additional Evidence`
- Global Vehicle Detail renders one representative summary card plus one grouped member-track card per vehicle track instead of a flat mixed evidence gallery
- Tracks, Global Vehicles, Vehicle Search, Cross-Camera Matches, and Run Detail all rely on lightweight `primary_vehicle_media` and `primary_plate_media` fields so list pages can show thumbnails without loading every evidence item
- Track Detail now renders linked global membership as the actual `global_vehicle_code` plus an `Open Global Vehicle` link; only truly unlinked tracks show `Not linked`

Live validation on July 27, 2026 against `RUN_20260725_131944` confirmed:

- `RUN_20260725_131944:CAM_002:TRACK_4` renders both `BEST_VEHICLE_CROP` and `PLATE_CROP`
- the same track now renders `GVO:RUN_20260725_131944:FA3FCF9E3ABC` with `Open Global Vehicle` instead of `Not linked`
- confirmed global vehicle `GVO:RUN_20260725_131944:FA3FCF9E3ABC` renders evidence from both `CAM_001 TRACK_4` and `CAM_002 TRACK_4`
- the shared grouped card keeps plate `DL8CBF6268` visually attached to its vehicle crop across track, global-vehicle, search, and match-comparison views
- evidence images are fetched from FastAPI content URLs, not from Supabase directly
- the `/search` page can return the known grey verified car `DL8CBF6268` as both a local track result and a linked global vehicle result
- the `/search` page also validates natural-language queries for the same known grey verified car and shows the interpreted filter panel before rendering the shared result cards

## Structured vehicle search

- Route: `http://127.0.0.1:5173/search`
- The page uses query-string state so searches can be refreshed and bookmarked.
- Searches submit on explicit button click only; the UI does not query on every keystroke.
- Result cards reuse the shared evidence renderer and open the existing local-track or global-vehicle detail routes.
- Natural-language search posts to `POST /api/v1/search/natural-language` and keeps the selected run and result scope in context.
- The interpreted-filters panel shows the validated translation, parser provider, and fallback-used status without exposing prompts or raw provider JSON.
- `Apply to filters` copies interpreted values into the existing structured form so operators can refine and rerun the same search contract manually.
- Clarification-required responses render safely and do not trigger direct browser access to Supabase or any LLM provider.
- Supported examples include `Find vehicle DL8CBF6268`, `Find the grey car with plate ending in 6268`, `Show cars seen in both cameras`, and `Find cars around 2 PM`.

## Commands

Run tests:

```powershell
npm run test
```

Create a production build:

```powershell
npm run build
```

## Backend dependency

- FastAPI must be reachable at `VITE_API_BASE_URL`
- The backend must have valid `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in its own server environment

## Known limitations

- No write actions are exposed in the UI
- Dashboard cards show only aggregates returned by the backend
- Placeholder behavior for missing or reference-only media is primarily covered by the frontend Vitest suite; the July 27, 2026 live validation used tracks whose evidence files were present locally
- Live frontend validation depends on the backend being started with valid analytics credentials
- The run dropdown defaults from the latest completed run currently returned by the backend when no explicit run query parameter is present
- Natural-language understanding is intentionally bounded to validated backend parsing plus deterministic fallback patterns; it is not a free-form investigative assistant
