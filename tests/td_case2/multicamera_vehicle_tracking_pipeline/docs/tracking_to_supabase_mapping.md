# Tracking To Supabase Mapping

## Track row mapping

- `LocalVehicleTrack.track_uuid` -> `vehicle_tracks.track_uuid`
- `LocalVehicleTrack.camera_code` -> resolve `cameras.id` by `cameras.camera_code` -> `vehicle_tracks.camera_id`
- `LocalVehicleTrack.local_track_id` -> `vehicle_tracks.local_track_id`
- `LocalVehicleTrack.class_name` -> `vehicle_tracks.vehicle_class`
- `LocalVehicleTrack.first_seen_at` -> `vehicle_tracks.first_seen_at`
- `LocalVehicleTrack.last_seen_at` -> `vehicle_tracks.last_seen_at`
- `LocalVehicleTrack.first_frame_number` -> `vehicle_tracks.first_frame_number`
- `LocalVehicleTrack.last_frame_number` -> `vehicle_tracks.last_frame_number`
- `LocalVehicleTrack.observation_count` -> `vehicle_tracks.observation_count`
- `LocalVehicleTrack.best_confidence` -> `vehicle_tracks.best_confidence`
- `best_frame_path` -> `null`
- `best_crop_path` -> `null`

## Observation row mapping

- inserted `vehicle_tracks.id` -> `vehicle_observations.vehicle_track_id`
- `TrackObservation.frame_number` -> `vehicle_observations.frame_number`
- `TrackObservation.camera_timestamp` -> `vehicle_observations.observed_at`
- `TrackObservation.bbox_xyxy[0]` -> `vehicle_observations.bbox_x1`
- `TrackObservation.bbox_xyxy[1]` -> `vehicle_observations.bbox_y1`
- `TrackObservation.bbox_xyxy[2]` -> `vehicle_observations.bbox_x2`
- `TrackObservation.bbox_xyxy[3]` -> `vehicle_observations.bbox_y2`
- `TrackObservation.confidence` -> `vehicle_observations.confidence`

## Persistence rules in this stage

- persistence is disabled by default
- only `completed` tracks are written by default
- `discarded` tracks are skipped by default
- active, tentative, and temporarily lost tracks are skipped
- observations are written in batches
- `track_uuid` is the duplicate-protection key
- camera synchronization happens once per run and caches `camera_code -> camera_id`
- no `vehicle_attributes` rows are inserted in this stage
