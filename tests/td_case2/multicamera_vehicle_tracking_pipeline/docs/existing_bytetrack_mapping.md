# Existing ByteTrack Mapping

This stage reuses only the stable ByteTrack patterns from `tests/td_case2/streaming_tracking_pipeline` and keeps all new code inside `tests/td_case2/multicamera_vehicle_tracking_pipeline`.

## Reused patterns

- The Ultralytics `BYTETracker` construction pattern is reused from `bytetrack_stage.py`.
- The local LAP shim is reused so Ultralytics ByteTrack does not attempt runtime package installation.
- Detections are converted into the same results-like structure expected by `BYTETracker.update(...)`.
- Tracker output rows are interpreted with the same field assumptions:
  - `row[:4]` is `x1, y1, x2, y2`
  - `row[4]` is the native/local track ID
  - `row[5]` is the score when present
  - `row[7]` is the source detection index when present
- Class labels are preserved by caching the class name seen for each track ID, matching the old adapter behavior.
- Lost-track handling follows the same two-step idea as the old lifecycle layer:
  - tracker stops emitting an ID
  - application lifecycle marks the track temporarily lost
  - if the gap exceeds the configured timeout, the track is completed

## Wrapped components

- `tracking/tracker_factory.py` wraps native tracker construction and guarantees one ByteTrack instance per camera.
- `tracking/camera_tracker.py` wraps one native tracker plus detection-to-tracker conversion and tracker-output parsing.
- `tracking/camera_detection_router.py` isolates routing so `CAM_001` packets can never update the `CAM_002` tracker.
- `tracking/track_lifecycle.py` wraps the tracker outputs into local test-stage lifecycle states and completion records.

## Excluded parts

The following parts of the old pipeline are intentionally not reused in this stage:

- Supervision ByteTrack backend support
- crop collection
- OCR and plate handling
- colour enrichment
- best-crop selection
- cross-camera matching
- person tracking
- global identity logic
- Supabase writes
- downstream UI or media packaging

## Output mapping into the new models

Native tracker output is mapped into the new local models as follows:

- Ultralytics native `track_id` becomes `TrackObservation.local_track_id`
- `camera_code + local_track_id` becomes deterministic `track_uuid`
- per-frame tracker rows become `TrackObservation`
- grouped observations across time become `LocalVehicleTrack`
- lifecycle states map to:
  - newly seen track: `tentative`
  - enough observations: `active`
  - missing but still inside timeout: `temporarily_lost`
  - timed out or flushed confirmed track: `completed`
  - short unconfirmed track flushed or expired: `discarded`

## Reset behavior

- The old stable code allowed full tracker reset between runs.
- The new factory exposes `reset()` for tests and explicit fresh runs only.
- Within one validation run, tracker state is preserved per camera and is never shared across cameras.
