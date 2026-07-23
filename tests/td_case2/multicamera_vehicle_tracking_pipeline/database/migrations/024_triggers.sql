create or replace function analytics.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create or replace function analytics.validate_cross_camera_match()
returns trigger
language plpgsql
as $$
declare
    source_camera uuid;
    candidate_camera uuid;
begin
    select camera_id into source_camera from analytics.vehicle_track where id = new.source_track_id;
    select camera_id into candidate_camera from analytics.vehicle_track where id = new.candidate_track_id;
    if source_camera is null or candidate_camera is null then
        raise exception 'cross_camera_match references missing vehicle_track rows';
    end if;
    if source_camera = candidate_camera then
        raise exception 'cross_camera_match source and candidate tracks must belong to different cameras';
    end if;
    return new;
end;
$$;

drop trigger if exists trg_camera_set_updated_at on analytics.camera;
create trigger trg_camera_set_updated_at before update on analytics.camera for each row execute function analytics.set_updated_at();

drop trigger if exists trg_video_source_set_updated_at on analytics.video_source;
create trigger trg_video_source_set_updated_at before update on analytics.video_source for each row execute function analytics.set_updated_at();

drop trigger if exists trg_camera_relation_set_updated_at on analytics.camera_relation;
create trigger trg_camera_relation_set_updated_at before update on analytics.camera_relation for each row execute function analytics.set_updated_at();

drop trigger if exists trg_processing_run_set_updated_at on analytics.processing_run;
create trigger trg_processing_run_set_updated_at before update on analytics.processing_run for each row execute function analytics.set_updated_at();

drop trigger if exists trg_camera_run_set_updated_at on analytics.camera_run;
create trigger trg_camera_run_set_updated_at before update on analytics.camera_run for each row execute function analytics.set_updated_at();

drop trigger if exists trg_processing_job_set_updated_at on analytics.processing_job;
create trigger trg_processing_job_set_updated_at before update on analytics.processing_job for each row execute function analytics.set_updated_at();

drop trigger if exists trg_vehicle_track_set_updated_at on analytics.vehicle_track;
create trigger trg_vehicle_track_set_updated_at before update on analytics.vehicle_track for each row execute function analytics.set_updated_at();

drop trigger if exists trg_track_media_set_updated_at on analytics.track_media;
create trigger trg_track_media_set_updated_at before update on analytics.track_media for each row execute function analytics.set_updated_at();

drop trigger if exists trg_vehicle_attribute_set_updated_at on analytics.vehicle_attribute;
create trigger trg_vehicle_attribute_set_updated_at before update on analytics.vehicle_attribute for each row execute function analytics.set_updated_at();

drop trigger if exists trg_plate_detection_set_updated_at on analytics.plate_detection;
create trigger trg_plate_detection_set_updated_at before update on analytics.plate_detection for each row execute function analytics.set_updated_at();

drop trigger if exists trg_plate_reading_set_updated_at on analytics.plate_reading;
create trigger trg_plate_reading_set_updated_at before update on analytics.plate_reading for each row execute function analytics.set_updated_at();

drop trigger if exists trg_plate_summary_set_updated_at on analytics.plate_summary;
create trigger trg_plate_summary_set_updated_at before update on analytics.plate_summary for each row execute function analytics.set_updated_at();

drop trigger if exists trg_cross_camera_match_set_updated_at on analytics.cross_camera_match;
create trigger trg_cross_camera_match_set_updated_at before update on analytics.cross_camera_match for each row execute function analytics.set_updated_at();

drop trigger if exists trg_cross_camera_match_validate on analytics.cross_camera_match;
create trigger trg_cross_camera_match_validate before insert or update on analytics.cross_camera_match for each row execute function analytics.validate_cross_camera_match();

drop trigger if exists trg_global_vehicle_set_updated_at on analytics.global_vehicle;
create trigger trg_global_vehicle_set_updated_at before update on analytics.global_vehicle for each row execute function analytics.set_updated_at();

drop trigger if exists trg_global_vehicle_track_set_updated_at on analytics.global_vehicle_track;
create trigger trg_global_vehicle_track_set_updated_at before update on analytics.global_vehicle_track for each row execute function analytics.set_updated_at();

drop trigger if exists trg_ai_model_set_updated_at on analytics.ai_model;
create trigger trg_ai_model_set_updated_at before update on analytics.ai_model for each row execute function analytics.set_updated_at();

drop trigger if exists trg_processing_error_set_updated_at on analytics.processing_error;
create trigger trg_processing_error_set_updated_at before update on analytics.processing_error for each row execute function analytics.set_updated_at();

drop trigger if exists trg_event_candidate_set_updated_at on analytics.event_candidate;
create trigger trg_event_candidate_set_updated_at before update on analytics.event_candidate for each row execute function analytics.set_updated_at();

drop trigger if exists trg_analytics_event_set_updated_at on analytics.analytics_event;
create trigger trg_analytics_event_set_updated_at before update on analytics.analytics_event for each row execute function analytics.set_updated_at();
