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
- Processing Runs
- Run Detail
- Local Tracks
- Track Detail
- Global Vehicles
- Global Vehicle Detail
- Cross-Camera Matches

## Supported filters

- Runs: status, run code, sorting, pagination
- Tracks: run, camera, class, colour, plate, plate status, lifecycle, confidence, media flag, pagination
- Global vehicles: run, status, class, colour, plate, confidence, camera count, pagination
- Matches: run, decision, score, rule version, camera, pagination

## Evidence and media behavior

- Media is currently `REFERENCE_ONLY`
- The UI shows safe metadata and references only
- The UI does not assume image bytes are available
- No broken image tags are rendered for reference-only media

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
- Media remains reference-only in this stage
- Live frontend validation depends on the backend being started with valid analytics credentials
