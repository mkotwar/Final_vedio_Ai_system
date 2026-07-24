create index if not exists idx_cross_camera_match_processing_run
on analytics.cross_camera_match(processing_run_id, decision, overall_score desc);

create index if not exists idx_cross_camera_match_created_global_vehicle
on analytics.cross_camera_match(created_global_vehicle_id);

create index if not exists idx_global_vehicle_processing_run
on analytics.global_vehicle(processing_run_id, status, created_at desc);

create index if not exists idx_global_vehicle_track_vehicle_track
on analytics.global_vehicle_track(vehicle_track_id, is_current, attached_at desc);
