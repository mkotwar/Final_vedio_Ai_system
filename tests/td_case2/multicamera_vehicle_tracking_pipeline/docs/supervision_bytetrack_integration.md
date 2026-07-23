# Supervision ByteTrack Integration

Installed version inspected in this environment: `supervision 0.29.1`.

## Constructor

The installed `supervision.tracker.byte_tracker.core.ByteTrack` class defines:

```python
ByteTrack(
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    frame_rate: float = 30,
    minimum_consecutive_frames: int = 1,
)
```

Important behavior from the installed implementation:

- `frame_rate` is used to compute `max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)`.
- new tracks are only initialized when `score >= track_activation_threshold + 0.1`.
- `minimum_consecutive_frames` controls when a track receives an external tracker ID.
- with the current sample detections (`~0.29` to `~0.31` confidence), `track_activation_threshold=0.25` produced no tracks, while `0.15` produced stable local tracker IDs.

## update_with_detections

The installed method signature is:

```python
update_with_detections(detections: Detections) -> Detections
```

Expected detection fields:

- `xyxy`: `np.ndarray` with shape `(N, 4)`
- `confidence`: `np.ndarray` with shape `(N,)`
- `class_id`: optional for tracking math, but preserved in the returned `Detections`

The method requires `confidence`; missing confidence raises `ValueError`.

## Output fields

Returned value is a `supervision.Detections` object filtered down to detections that matched active tracks.

- tracker IDs are returned in `detections.tracker_id`
- unmatched detections are assigned `-1` internally and filtered out before return
- `xyxy`, `confidence`, and `class_id` remain aligned with the returned rows

## Empty detections

Empty input should be passed as:

```python
sv.Detections(
    xyxy=np.empty((0, 4), dtype=np.float32),
    confidence=np.empty((0,), dtype=np.float32),
    class_id=np.empty((0,), dtype=np.int32),
)
```

The installed tracker also returns an empty `Detections` object with an empty `tracker_id` array when no tracks are active.

## Frame rate guidance

Use the effective frame rate being sent into ByteTrack:

- full-frame processing: source FPS
- sampled processing: processed FPS after sampling

For the current local videos, source FPS is approximately `19.951`, so using a hardcoded `30` changes the lost-track window calculation.

## Config mapping used in this pipeline

Supervision-native fields:

- `track_activation_threshold`
- `lost_track_buffer`
- `minimum_matching_threshold`
- `frame_rate`
- `minimum_consecutive_frames`

Compatibility aliases retained for clarity:

- `track_high_thresh` -> `track_activation_threshold`
- `track_buffer` -> `lost_track_buffer`
- `match_thresh` -> `minimum_matching_threshold`

Pipeline lifecycle confirmation remains separate:

- `minimum_consecutive_frames` controls when Supervision emits a tracker ID
- `min_confirmed_observations` controls when this pipeline promotes a local track from tentative to active
