# Global Vehicle Matching

This document describes the first production-safe cross-camera vehicle matching stage for `tests/td_case2/multicamera_vehicle_tracking_pipeline`.

## Concepts

- `vehicle_track`: one completed vehicle track within one camera stream. This remains the camera-level source of truth.
- `cross_camera_match`: one deterministic comparison between two camera-level tracks, with auditable scores, reasons, and a persisted decision.
- `global_vehicle`: one logical vehicle identity that can span one or more cameras.
- `global_vehicle_track`: one membership row linking a camera-level track into a global vehicle object.

No camera-level `vehicle_track` rows are merged, deleted, or rewritten.

## Hard rules

- The matcher never compares a track with itself.
- Same-camera pairs are rejected by default.
- Two different `VERIFIED` plates are always rejected.
- Two identical `VERIFIED` plates on different cameras may be `CONFIRMED` if route and time rules do not make the transition impossible.
- One verified plate plus one missing plate can become `POSSIBLE`, not `CONFIRMED`.
- Unverified OCR cannot independently create a confirmed identity.
- Colour and class are supporting evidence only.

## Scoring

Default scoring lives in `config/global_matching.yaml`.

```yaml
matching:
  rule_version: global_match_v1
  weights:
    verified_plate: 0.70
    time: 0.10
    camera_route: 0.10
    vehicle_class: 0.05
    vehicle_colour: 0.05
    visual_similarity: 0.00
  thresholds:
    confirmed: 0.85
    possible: 0.55
```

Hard verified-plate rules take precedence over the weighted score.

## Time handling

`time_matching.mode` supports:

- `recording_timestamp`
- `relative_video_time`
- `disabled`

The checked-in default is `disabled` because the current local validation data should not assume synchronized wall-clock timestamps across independent files.

## Camera routes

Routes can be configured in `config/global_matching.yaml`:

```yaml
camera_routes:
  CAM_001:
    CAM_002:
      allowed: true
      minimum_travel_seconds: 0
      maximum_travel_seconds: 300
```

If no route is configured, the matcher treats route evidence as neutral instead of inventing travel assumptions.

## Candidate generation

The standalone stage generates candidates in this order:

1. Same verified normalized plate across different cameras.
2. One verified plate plus a compatible no-plate track.
3. Same class or same colour with compatible camera scope.

The initial version does not add a new deep visual ReID dependency.

## Single-track objects

`create_single_track_global_objects: true` is the default.

That means unmatched camera-level tracks still receive a stable `global_vehicle` identity through a one-member object. This makes later audit and expansion simpler while keeping the source track unchanged.

## Idempotency

- Match pairs are canonicalized by track ID order before persistence.
- Global object codes are stable hashes of sorted member track UUIDs.
- Existing `global_vehicle` rows are reused by `(processing_run_id, global_vehicle_code)`.
- Existing `global_vehicle_track` membership rows are updated instead of duplicated.
- The partial unique index on active `global_vehicle_track.vehicle_track_id` still prevents one track from belonging to multiple active objects.

## Commands

Dry run:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.build_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --global-match-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\global_matching.yaml `
  --dry-run `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_match_dry_run.json"
```

Persist:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.build_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --global-match-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\global_matching.yaml `
  --persist `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_match_persisted.json"
```

Verify:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_global_vehicle_objects `
  --run-code RUN_20260724_151402 `
  --strict `
  --json-output "debug_runs\multicamera_vehicle_tracking_pipeline\RUN_20260724_151402_global_object_audit.json"
```

## Known limitations

- The current verifier is read-only and does not fetch image bytes.
- The first version prefers conservative review behavior over automatic multi-object merges.
- Route logic is config-driven and does not yet include learned travel-time priors.
- Visual similarity remains disabled in the checked-in config.

## API access

The new read-only FastAPI backend exposes persisted global-vehicle data without giving the frontend direct service-role access.

Relevant endpoints:

- `GET /api/v1/global-vehicles`
- `GET /api/v1/global-vehicles/{global_vehicle_code}`
- `GET /api/v1/global-vehicles/{global_vehicle_code}/tracks`
- `GET /api/v1/cross-camera-matches`
- `GET /api/v1/cross-camera-matches/{match_id}`

For the validated July 24, 2026 run `RUN_20260724_151402`, the confirmed object `GVO:RUN_20260724_151402:943BD1FE7C62` should expose both `TRACK_4` members from `CAM_001` and `CAM_002`, plus canonical plate `DL8CBF6268`, class `CAR`, and colour `GREY`.
