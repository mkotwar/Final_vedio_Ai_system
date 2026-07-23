create index if not exists idx_analytics_camera_site_code
on analytics.camera(site_code);

create index if not exists idx_analytics_camera_enabled
on analytics.camera(camera_code)
where enabled = true;

create index if not exists idx_video_source_camera_time
on analytics.video_source(camera_id, source_start_at, source_end_at);

create index if not exists idx_video_source_external_recording_id
on analytics.video_source(external_recording_id);

create index if not exists idx_camera_relation_source_enabled
on analytics.camera_relation(source_camera_id, destination_camera_id)
where enabled = true;

create index if not exists idx_processing_run_status_started_at
on analytics.processing_run(status, started_at desc);

create index if not exists idx_camera_run_processing_run
on analytics.camera_run(processing_run_id);

create index if not exists idx_camera_run_camera_status
on analytics.camera_run(camera_id, status);

create index if not exists idx_processing_job_run_status
on analytics.processing_job(processing_run_id, status, job_type);

create index if not exists idx_processing_job_camera_run
on analytics.processing_job(camera_run_id, created_at desc);

create index if not exists idx_vehicle_track_camera_time
on analytics.vehicle_track(camera_id, first_seen_at, last_seen_at);

create index if not exists idx_vehicle_track_processing_run
on analytics.vehicle_track(processing_run_id, camera_id);

create index if not exists idx_vehicle_track_vehicle_class
on analytics.vehicle_track(vehicle_class);

create index if not exists idx_vehicle_track_searchable
on analytics.vehicle_track(first_seen_at desc)
where searchable = true;

create index if not exists idx_track_observation_vehicle_track_time
on analytics.track_observation(vehicle_track_id, observed_at);

create index if not exists idx_track_observation_camera_time
on analytics.track_observation(camera_id, observed_at);

create index if not exists idx_track_observation_key_only
on analytics.track_observation(vehicle_track_id, observed_at)
where is_key_observation = true;

create index if not exists idx_track_media_vehicle_track_type
on analytics.track_media(vehicle_track_id, media_type, captured_at desc);

create index if not exists idx_track_media_primary
on analytics.track_media(vehicle_track_id, selection_rank)
where is_primary = true;

create index if not exists idx_vehicle_attribute_vehicle_track
on analytics.vehicle_attribute(vehicle_track_id, attribute_status);

create index if not exists idx_vehicle_attribute_primary_color
on analytics.vehicle_attribute(primary_color);

create index if not exists idx_plate_detection_vehicle_track
on analytics.plate_detection(vehicle_track_id, detected_at desc);

create index if not exists idx_plate_reading_detection_selected
on analytics.plate_reading(plate_detection_id, is_selected, confidence desc);

create index if not exists idx_plate_reading_normalized_text
on analytics.plate_reading(normalized_text);

create index if not exists idx_plate_reading_normalized_text_trgm
on analytics.plate_reading
using gin (normalized_text gin_trgm_ops);

create index if not exists idx_plate_summary_canonical_plate
on analytics.plate_summary(canonical_plate);

create index if not exists idx_plate_summary_canonical_plate_trgm
on analytics.plate_summary
using gin (canonical_plate gin_trgm_ops);

create index if not exists idx_cross_camera_match_source
on analytics.cross_camera_match(source_track_id, overall_score desc);

create index if not exists idx_cross_camera_match_candidate
on analytics.cross_camera_match(candidate_track_id, overall_score desc);

create index if not exists idx_cross_camera_match_decision
on analytics.cross_camera_match(decision, overall_score desc);

create index if not exists idx_global_vehicle_plate
on analytics.global_vehicle(canonical_plate);

create index if not exists idx_global_vehicle_track_current
on analytics.global_vehicle_track(global_vehicle_id, attached_at desc);

create unique index if not exists uq_global_vehicle_track_current_vehicle_track
on analytics.global_vehicle_track(vehicle_track_id)
where is_current = true;

create index if not exists idx_ai_model_active
on analytics.ai_model(model_type, active);

create index if not exists idx_run_model_processing_run
on analytics.run_model(processing_run_id, stage_name);

create index if not exists idx_processing_error_run_severity
on analytics.processing_error(processing_run_id, severity, created_at desc);

create index if not exists idx_processing_error_track
on analytics.processing_error(vehicle_track_id, created_at desc);

create index if not exists idx_event_candidate_time
on analytics.event_candidate(event_type, event_time desc);

create index if not exists idx_event_candidate_track
on analytics.event_candidate(vehicle_track_id, event_time desc);

create index if not exists idx_analytics_event_time
on analytics.analytics_event(event_type, event_time desc);

create index if not exists idx_analytics_event_global_vehicle
on analytics.analytics_event(global_vehicle_id, event_time desc);
