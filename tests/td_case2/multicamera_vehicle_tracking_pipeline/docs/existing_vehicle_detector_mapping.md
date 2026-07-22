# Existing Vehicle Detector Mapping

This document explains which stable detector patterns from `tests/td_case2/streaming_tracking_pipeline` were reused for the new experimental multi-camera detection stage.

## Reused patterns

- Model loading through `ultralytics.YOLO(...)`
- One-frame-at-a-time `model.predict(...)`
- Confidence and IoU inference arguments
- Optional device and image-size arguments
- Class-name extraction from `model.names`
- Bounding-box conversion from YOLO `xyxy`
- Frame-boundary box clamping
- Invalid-box rejection

## Wrapped code, not copied wholesale

The new detector does not import or reuse the old pipeline classes directly. Instead it wraps the same stable ideas in a smaller input-to-detection flow:

- old: `UltralyticsYoloDetectionStage.process(FramePacket) -> DetectionPacket`
- new: `SharedVehicleDetector.detect(FramePacket) -> DetectionPacket`

The new wrapper keeps multi-camera frame metadata and returns only normalized vehicle detections.

## Intentionally excluded parts

These old-pipeline pieces were not carried over:

- tracking integration
- person detection support
- crop collection
- lifecycle management
- output JSONL pipeline artifacts
- search indexing
- ANPR, OCR, colour, and event logic

## Old detector behavior mapped to the new stage

- default confidence threshold reused: `0.25`
- default IoU threshold reused: `0.45`
- common custom vehicle model path reused: `object/vehical_detection/best_old.pt`
- class-name normalization reused as a pattern, but reduced to only:
  - `car`
  - `bus`
  - `truck`
  - `motorcycle`

## Detector output mapping

Old detector output shape:

- frame metadata
- `DetectionRecord` objects with normalized class names and clipped boxes

New detector output shape:

- original camera/frame metadata from the multi-camera `FramePacket`
- `VehicleDetection` rows:
  - `class_id`
  - `class_name`
  - `confidence`
  - `bbox_xyxy`
- `DetectionPacket` with:
  - camera metadata
  - frame index and timestamps
  - frame size
  - detection list
  - inference time
  - actual detector model
  - detector device

## Current detector constraints

- one shared model instance for all cameras
- sequential one-frame inference
- no batching
- no tracking
- no Supabase track writes
