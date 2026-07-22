# Worker Pipeline Architecture

## Sequential behavior

The current sequential pipeline reads one frame at a time, runs YOLO immediately, routes detections into the per-camera tracker state, finalizes tracks in-process, and optionally persists completed tracks before generating the report.

## Worker behavior

The worker pipeline splits the same logic into cooperating stages:

- one camera-reader thread per camera
- one shared detection worker
- one tracking-router worker
- one optional persistence worker

The detector, tracker adapter, lifecycle logic, and persistence service are reused from the sequential path.

## Queue ownership

- `frame_queue`: camera-reader workers -> detection worker
- `detection_queue`: detection worker -> tracking worker
- `completed_track_queue`: tracking worker -> persistence worker or supervisor collector
- `error_queue`: every worker -> supervisor

All queues are bounded.

## Ordering guarantees

- each camera-reader worker emits frames for its own camera in increasing frame order
- one shared detection worker preserves queue-consumption order
- the tracking worker validates `frame_number` monotonicity per camera before updating ByteTrack
- one camera's detections are never sent to another camera's tracker

## Shutdown sequence

1. downstream workers start first
2. camera-reader workers start
3. each camera-reader emits `EndOfCameraMessage`
4. detection worker forwards camera-end events and then one `EndOfInputMessage`
5. tracking worker flushes ended cameras, then flushes all on final end-of-input
6. persistence worker drains completed tracks until final end-of-input
7. supervisor joins all threads

## Error handling behavior

- camera-reader errors are camera-scoped by default and can allow other cameras to continue
- detector errors are fatal by default
- tracking errors are fatal by default
- persistence errors are non-fatal by default
- every worker reports structured errors through the bounded error queue

## Current limitations

- one shared YOLO worker only
- no GPU batching yet
- no frame dropping
- no live multi-thread preview
- no multiprocessing
- no RTSP or VMS integration
- no OCR, colour enrichment, or cross-camera matching
