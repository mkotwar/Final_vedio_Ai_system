# Worker Pipeline Architecture

## Thread ownership

- one `CameraReaderWorker` thread per enabled camera
- one shared `DetectionWorker` thread owning the single `SharedVehicleDetector`
- one `TrackingWorker` thread owning the `CameraDetectionRouter`
- one optional `PersistenceWorker` thread only when worker persistence is enabled and pipeline persistence is enabled
- otherwise the supervisor drains completed tracks itself

## Queue ownership

- `frame_queue`: camera readers -> detection worker
- `detection_queue`: detection worker -> tracking worker
- `completed_track_queue`: tracking worker -> persistence worker or supervisor collector
- `error_queue`: all workers -> supervisor

All queues are bounded and use blocking puts with timeouts. No frame dropping is used for the local-video worker path.

## Packet flow

1. each camera reader opens its own `CameraSource`
2. each reader emits ordered `FramePacket` objects for a single camera
3. the shared detector consumes `FramePacket`, runs YOLO once per frame, and emits `DetectionPacket`
4. the tracking worker routes `DetectionPacket` by `camera_code` into the correct per-camera tracker
5. completed local tracks are emitted as `CompletedTrackMessage`
6. the persistence worker or supervisor collector drains completed tracks without blocking tracking progress permanently

## Camera ordering

- frame order is preserved inside each camera reader thread
- the tracking worker maintains `last_frame_number_by_camera`
- out-of-order packets are rejected per camera
- one camera ending emits `EndOfCameraMessage` and does not stop other cameras

## Tracker isolation

- the tracking worker uses one independent `supervision.ByteTrack` instance per camera
- tracker frame rate is resolved per camera from the effective source FPS currently being processed
- empty detection packets still pass through the tracker and lifecycle path
- tracker IDs may repeat numerically across cameras, but they remain isolated by `camera_code`

## Shutdown sequence

1. persistence worker or supervisor collector is ready first
2. tracking worker starts
3. detection worker starts
4. camera reader threads start
5. readers finish independently and emit `EndOfCameraMessage`
6. detection worker forwards per-camera end events and emits one `EndOfInputMessage` after all cameras finish
7. tracking worker flushes individual cameras on `EndOfCameraMessage`
8. tracking worker flushes remaining cameras on `EndOfInputMessage`
9. persistence worker or supervisor collector drains the completed-track queue
10. supervisor joins every thread and reports any thread that did not stop cleanly

## Persistence behavior

- Supabase writes never run in the tracking thread
- the persistence worker reuses `TrackingPersistenceService`
- persistence errors are reported through `error_queue`
- nonfatal persistence errors can be recorded while the rest of the pipeline continues

## Error propagation

- camera read errors are camera-scoped by default and still emit `EndOfCameraMessage`
- detector errors are fatal by default and set the shared shutdown event
- tracking errors are fatal by default and set the shared shutdown event
- persistence errors are nonfatal by default
- all worker failures become structured `WorkerErrorMessage` entries and appear in the worker report
