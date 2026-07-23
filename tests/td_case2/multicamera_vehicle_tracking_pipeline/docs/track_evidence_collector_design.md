## Track Evidence Collector Design

### Current Frame-to-Track Flow

The current worker runtime flow is:

1. `CameraSource.read_next()` in [camera_source.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/ingestion/camera_source.py) reads an OpenCV frame and emits a `FramePacket`.
2. `CameraReaderWorker` in [camera_reader_worker.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/camera_reader_worker.py) pushes that `FramePacket` into the shared `frame_queue`.
3. `DetectionWorker` in [detection_worker.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/detection_worker.py) receives the same `FramePacket`, runs YOLO via `SharedVehicleDetector.detect()`, and emits a `DetectionPacket`.
4. `DetectionPacket` currently still carries the original `frame` reference. This is important: the tracking worker receives both detections and the frame pixels.
5. `TrackingWorker` in [tracking_worker.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/tracking_worker.py) receives `DetectionPacket`, routes it through `CameraDetectionRouter.route()`, and gets:
   - `observations`
   - `completed_tracks`
   - `active_tracks`
6. `CameraDetectionRouter` in [camera_detection_router.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/camera_detection_router.py) uses one `CameraTracker` per camera and a shared `LocalTrackLifecycle`.
7. `CameraTracker.update()` in [camera_tracker.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/camera_tracker.py) converts tracker output to `TrackObservation` rows with:
   - `camera_code`
   - `local_track_id`
   - `frame_number`
   - `video_time_seconds`
   - `camera_timestamp`
   - `class_name`
   - `confidence`
   - `bbox_xyxy`
8. `LocalTrackLifecycle.update()` in [track_lifecycle.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/track_lifecycle.py) owns active per-camera track state and emits `LocalVehicleTrack` only when tracks complete or are discarded.
9. `TrackingWorker._emit_tracks()` sends completed tracks to the downstream queue as `CompletedTrackMessage`.

The sequential path in [multicamera_tracking_orchestrator.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py) uses the same `DetectionPacket` -> `CameraDetectionRouter.route()` -> `completed_tracks` flow directly in-process.

### Runtime Models and Where Data Exists

#### Frame packet

Defined in [frame_packet.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/ingestion/frame_packet.py):

- includes raw `frame`
- includes `camera_code`, `camera_name`, `source_path`
- includes `frame_number`, `source_fps`, `video_time_seconds`, `camera_timestamp`

#### Detection packet

Defined in [detection_models.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/detection/detection_models.py):

- includes all per-frame metadata
- includes `detections`
- includes `frame_width`, `frame_height`
- includes original `frame`

This means the tracking worker currently receives the original frame, not detections only.

#### Track observation model

Defined in [tracking_models.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/tracking_models.py):

- `local_track_id`
- `bbox_xyxy`
- `confidence`
- `class_name`
- `track_uuid`
- frame/time metadata

#### Active-track state

Defined internally as `_TrackState` in [track_lifecycle.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/track_lifecycle.py).

Current `_TrackState` retains:

- camera metadata
- first/last frame information
- best confidence
- all `TrackObservation` rows for the track
- current lifecycle state

It does not retain image evidence today.

### Where Bounding Boxes Are Available

Bounding boxes are available at all of these points:

1. `VehicleDetection.bbox_xyxy` in `DetectionPacket.detections`
2. `TrackObservation.bbox_xyxy` in tracker output
3. `LocalVehicleTrack.observations[*].bbox_xyxy` after lifecycle completion

The safest evidence source is `TrackObservation.bbox_xyxy`, because it matches the final tracker-assigned local track ID and final per-frame track history.

### Where Original Frames Are Still Available

Original frames are available:

1. inside `FramePacket.frame`
2. inside `DetectionPacket.frame`
3. temporarily inside the tracking worker while processing a packet

After the tracking worker moves on to the next packet, the only durable state retained today is metadata and observations, not image pixels.

### Where Completed Tracks Are Emitted

Completed tracks are emitted in:

1. `TrackingWorker._emit_tracks()` after `router.route()`
2. `TrackingWorker._flush_camera()` on `EndOfCameraMessage`
3. `TrackingWorker._flush_all()` on `EndOfInputMessage`
4. sequential path after `tracking_result.completed_tracks`
5. sequential path after `router.flush_all()`

### Where Tracks Are Flushed

Per-camera flush occurs in:

- `TrackingWorker._flush_camera()`
- `CameraDetectionRouter.flush_camera()`
- `LocalTrackLifecycle.flush_camera()`

End-of-input flush occurs in:

- `TrackingWorker._flush_all()`
- `CameraDetectionRouter.flush_all()`
- `LocalTrackLifecycle.flush_all()`

### Best Integration Point

The best integration point is immediately after `router.route(item)` in `TrackingWorker.run()`, while we still have:

- `DetectionPacket.frame`
- `DetectionPacket.frame_width`
- `DetectionPacket.frame_height`
- `TrackObservation` rows already aligned to tracker-assigned `local_track_id`
- camera isolation already enforced

This gives the collector access to the actual frame pixels plus the tracker-stable bounding box for that frame.

For the sequential path, the matching integration point is immediately after:

- `tracking_result = self.router.route(detection_packet)`

in [multicamera_tracking_orchestrator.py](F:/vinfo/Final_vedio_Ai_system/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py).

### Why Not Integrate Earlier

Do not integrate in the detector:

- detector boxes are not yet associated with stable track IDs
- track evidence must be grouped by final track identity

Do not integrate only at completion time:

- the original frame is no longer retained
- recomputing evidence from metadata only is impossible

Do not retain whole-frame history:

- this would grow memory unbounded with track length and camera count

### Frame Ownership and Lifetime

Current lifetime:

1. OpenCV frame is created in `CameraSource.read_next()`
2. passed through `FramePacket`
3. carried into `DetectionPacket`
4. consumed by the tracking worker

The evidence collector should treat the frame as ephemeral and must copy only the cropped pixels that survive candidate filtering.

That means:

- never retain the original full frame beyond the current packet
- crop only for observations that pass initial validation
- keep only bounded candidate crops per active track

### Queue Safety

`DetectionPacket` already contains the original frame. Passing a frame reference further inside the tracking worker is safe because:

- the queue already transports the frame today
- no additional queue payload type is required
- no change to frame ordering is required

What should not be done:

- adding a second parallel frame queue
- retaining arbitrary packet references in long-lived track state
- storing every frame or every crop

### Candidate Selection Strategy

Per active track, keep a bounded `TrackEvidenceState` with a small set of named candidates:

- `first_valid`
- `latest_valid`
- `highest_confidence`
- `largest_visible`
- `sharpest`
- `best_overall`
- optional `middle_candidate`

Each candidate should store:

- crop image bytes or an encoded in-memory image
- crop dimensions
- source frame number
- source timestamp / video time
- bbox metadata
- confidence
- derived quality metrics
- selection reason

Suggested rules:

1. Reject invalid or too-small crops first.
2. Clamp bbox to frame when configured.
3. Track the first accepted crop once.
4. Replace latest on every accepted observation.
5. Replace highest-confidence only when confidence improves.
6. Replace largest when crop area improves.
7. Replace sharpest when sharpness score improves.
8. Maintain one best-overall score from a weighted combination of:
   - detection confidence
   - sharpness
   - visible area
   - edge penalty
9. Keep a simple middle candidate by replacing it near the temporal midpoint as track length grows.

This yields a bounded candidate count regardless of track duration.

### Memory Risks

Main risks:

1. unbounded active tracks across many cameras
2. retaining full frames per observation
3. retaining every crop
4. storing numpy arrays for all observations

Mitigations:

1. evidence state keyed by `camera_code` + `local_track_id` or `track_uuid`
2. store only selected candidate crops
3. immediately JPEG-encode retained crops if configured, instead of holding many raw arrays
4. drop evidence state as soon as the completed track evidence package is finalized
5. keep `max_candidates_per_track` hard bounded

### Proposed Evidence Package

On completion, emit a final in-memory evidence package attached to the completed track or carried alongside it. The package should include:

- `run_id`
- `camera_code`
- `local_track_id`
- `track_uuid`
- `class_name`
- `candidate_count`
- selected candidate metadata
- optional saved file paths if saving is enabled

It should not write database rows in this task.

### Exact Files To Create

1. `config/evidence.yaml`
2. `evidence/evidence_config.py`
3. `evidence/evidence_models.py`
4. `evidence/track_evidence_collector.py`
5. tests for config and collector behavior

### Exact Files To Modify

1. `tracking/tracking_models.py`
   Add optional evidence package field(s) on completed track objects, or add a parallel evidence attachment model.
2. `workers/tracking_worker.py`
   Update collector on each processed packet and finalize evidence on completed tracks and flush.
3. `orchestration/multicamera_tracking_orchestrator.py`
   Optional matching integration for sequential path so both runtime modes behave consistently.
4. `orchestration/worker_multicamera_tracking_orchestrator.py`
   Load evidence config and pass collector into worker runtime.
5. `workers/worker_supervisor.py`
   Thread the collector dependency into `TrackingWorker`.

### Missing Runtime Fields

The current runtime already has enough data to start bounded crop collection:

- frame pixels
- tracker-aligned bbox
- frame number
- video time
- timestamps
- class and confidence
- run ID is already available at router/orchestrator level

Potentially useful but not strictly required new fields:

- explicit evidence package field on `LocalVehicleTrack`
- per-candidate metadata model
- per-track quality summary

### Recommendation

Implement the collector as a separate bounded component owned by the tracking worker and updated per `DetectionPacket` after `router.route()`.

This keeps:

- YOLO behavior unchanged
- ByteTrack behavior unchanged
- queue structure unchanged
- frame ordering unchanged
- memory bounded

It also ensures evidence is isolated naturally by:

- run ID
- camera code
- local track ID
- final track UUID
