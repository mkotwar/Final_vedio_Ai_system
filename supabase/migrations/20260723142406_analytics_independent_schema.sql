-- FILE: 001_extensions_and_schema.sql

create extension if not exists pgcrypto;
create extension if not exists pg_trgm;

create schema if not exists analytics;


-- FILE: 002_camera.sql

create table if not exists analytics.camera (
    id uuid primary key default gen_random_uuid(),
    camera_code varchar(100) not null,
    external_camera_id varchar(150),
    camera_name varchar(255),
    site_code varchar(100),
    location_name varchar(255),
    timezone varchar(100) not null default 'Asia/Kolkata',
    enabled boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_analytics_camera_code unique (camera_code)
);


-- FILE: 003_video_source.sql

create table if not exists analytics.video_source (
    id uuid primary key default gen_random_uuid(),
    camera_id uuid not null,
    source_type varchar(30) not null,
    external_recording_id varchar(255),
    source_reference text not null,
    source_start_at timestamptz,
    source_end_at timestamptz,
    source_fps numeric(8,3),
    frame_width integer,
    frame_height integer,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_video_source_camera foreign key (camera_id) references analytics.camera(id),
    constraint chk_video_source_type check (source_type in ('LOCAL_FILE', 'RTSP', 'LIVE_STREAM', 'VMS_RECORDING', 'PLAYBACK_API')),
    constraint chk_video_source_fps check (source_fps is null or source_fps > 0),
    constraint chk_video_source_frame_width check (frame_width is null or frame_width > 0),
    constraint chk_video_source_frame_height check (frame_height is null or frame_height > 0),
    constraint chk_video_source_time_range check (source_start_at is null or source_end_at is null or source_start_at <= source_end_at)
);


-- FILE: 004_camera_relation.sql

create table if not exists analytics.camera_relation (
    id uuid primary key default gen_random_uuid(),
    source_camera_id uuid not null,
    destination_camera_id uuid not null,
    relation_type varchar(50) not null default 'POSSIBLE_ROUTE',
    min_travel_seconds numeric,
    max_travel_seconds numeric,
    typical_travel_seconds numeric,
    distance_meters numeric,
    matching_weight numeric not null default 1,
    bidirectional boolean not null default false,
    enabled boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_camera_relation_source foreign key (source_camera_id) references analytics.camera(id),
    constraint fk_camera_relation_destination foreign key (destination_camera_id) references analytics.camera(id),
    constraint uq_camera_relation_route unique (source_camera_id, destination_camera_id, relation_type),
    constraint chk_camera_relation_distinct_cameras check (source_camera_id <> destination_camera_id),
    constraint chk_camera_relation_matching_weight check (matching_weight >= 0 and matching_weight <= 1),
    constraint chk_camera_relation_min_seconds check (min_travel_seconds is null or min_travel_seconds >= 0),
    constraint chk_camera_relation_max_seconds check (max_travel_seconds is null or max_travel_seconds >= 0),
    constraint chk_camera_relation_typical_seconds check (typical_travel_seconds is null or typical_travel_seconds >= 0),
    constraint chk_camera_relation_distance check (distance_meters is null or distance_meters >= 0),
    constraint chk_camera_relation_min_lte_max check (min_travel_seconds is null or max_travel_seconds is null or min_travel_seconds <= max_travel_seconds)
);


-- FILE: 005_processing_run.sql

create table if not exists analytics.processing_run (
    id uuid primary key default gen_random_uuid(),
    run_code varchar(100) not null,
    pipeline_name varchar(255),
    pipeline_version varchar(100),
    execution_mode varchar(30),
    status varchar(30),
    configured_camera_count integer not null default 0,
    active_camera_count integer not null default 0,
    started_at timestamptz,
    completed_at timestamptz,
    total_frames_processed bigint not null default 0,
    total_detections bigint not null default 0,
    total_track_observations bigint not null default 0,
    total_tracks bigint not null default 0,
    host_name varchar(255),
    runtime_device varchar(100),
    configuration jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_processing_run_code unique (run_code),
    constraint chk_processing_run_execution_mode check (execution_mode is null or execution_mode in ('SEQUENTIAL', 'THREADED', 'BATCHED', 'LIVE')),
    constraint chk_processing_run_status check (status is null or status in ('QUEUED', 'RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED')),
    constraint chk_processing_run_configured_camera_count check (configured_camera_count >= 0),
    constraint chk_processing_run_active_camera_count check (active_camera_count >= 0 and active_camera_count <= configured_camera_count),
    constraint chk_processing_run_total_frames check (total_frames_processed >= 0),
    constraint chk_processing_run_total_detections check (total_detections >= 0),
    constraint chk_processing_run_total_observations check (total_track_observations >= 0),
    constraint chk_processing_run_total_tracks check (total_tracks >= 0),
    constraint chk_processing_run_time_range check (started_at is null or completed_at is null or started_at <= completed_at)
);


-- FILE: 006_camera_run.sql

create table if not exists analytics.camera_run (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid not null,
    camera_id uuid not null,
    video_source_id uuid,
    status varchar(30),
    reader_worker_name varchar(255),
    resolved_source_fps numeric(8,3),
    effective_processing_fps numeric(8,3),
    first_frame_number bigint,
    last_frame_number bigint,
    frames_read bigint not null default 0,
    frames_processed bigint not null default 0,
    detections_count bigint not null default 0,
    track_observations_count bigint not null default 0,
    completed_tracks_count integer not null default 0,
    discarded_tracks_count integer not null default 0,
    started_at timestamptz,
    completed_at timestamptz,
    metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_camera_run_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_camera_run_camera foreign key (camera_id) references analytics.camera(id),
    constraint fk_camera_run_video_source foreign key (video_source_id) references analytics.video_source(id),
    constraint uq_camera_run_processing_run_camera unique (processing_run_id, camera_id),
    constraint chk_camera_run_status check (status is null or status in ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    constraint chk_camera_run_resolved_fps check (resolved_source_fps is null or resolved_source_fps > 0),
    constraint chk_camera_run_effective_fps check (effective_processing_fps is null or effective_processing_fps > 0),
    constraint chk_camera_run_frames check (frames_read >= 0 and frames_processed >= 0 and detections_count >= 0 and track_observations_count >= 0),
    constraint chk_camera_run_track_counts check (completed_tracks_count >= 0 and discarded_tracks_count >= 0),
    constraint chk_camera_run_frame_range check (first_frame_number is null or last_frame_number is null or first_frame_number <= last_frame_number),
    constraint chk_camera_run_time_range check (started_at is null or completed_at is null or started_at <= completed_at)
);


-- FILE: 007_processing_job.sql

create table if not exists analytics.processing_job (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid not null,
    camera_run_id uuid,
    job_type varchar(80) not null,
    status varchar(30) not null,
    priority integer not null default 0,
    attempt_number integer not null default 1,
    worker_name varchar(255),
    started_at timestamptz,
    completed_at timestamptz,
    processing_time_ms bigint,
    input_summary jsonb,
    output_summary jsonb,
    error_code varchar(100),
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_processing_job_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_processing_job_camera_run foreign key (camera_run_id) references analytics.camera_run(id),
    constraint chk_processing_job_type check (job_type in ('READ', 'DETECT', 'TRACK', 'PERSIST', 'BEST_FRAME_SELECTION', 'PLATE_DETECTION', 'OCR', 'COLOR', 'CROSS_CAMERA_MATCH', 'EVENT_ANALYSIS')),
    constraint chk_processing_job_status check (status in ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'RETRYING', 'CANCELLED')),
    constraint chk_processing_job_attempt_number check (attempt_number > 0),
    constraint chk_processing_job_processing_time check (processing_time_ms is null or processing_time_ms >= 0),
    constraint chk_processing_job_time_range check (started_at is null or completed_at is null or started_at <= completed_at)
);


-- FILE: 008_vehicle_track.sql

create table if not exists analytics.vehicle_track (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid not null,
    camera_run_id uuid not null,
    camera_id uuid not null,
    track_uuid varchar(200) not null,
    local_track_id bigint not null,
    vehicle_class varchar(40) not null,
    first_seen_at timestamptz not null,
    last_seen_at timestamptz not null,
    first_frame_number bigint not null,
    last_frame_number bigint not null,
    first_video_time_seconds numeric,
    last_video_time_seconds numeric,
    observation_count integer not null default 0,
    best_detection_confidence numeric,
    average_detection_confidence numeric,
    lifecycle_state varchar(30) not null,
    completion_reason varchar(50),
    tracker_backend varchar(100) not null,
    tracker_configuration jsonb not null default '{}'::jsonb,
    searchable boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_vehicle_track_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_vehicle_track_camera_run foreign key (camera_run_id) references analytics.camera_run(id),
    constraint fk_vehicle_track_camera foreign key (camera_id) references analytics.camera(id),
    constraint uq_vehicle_track_track_uuid unique (track_uuid),
    constraint uq_vehicle_track_local_scope unique (processing_run_id, camera_id, local_track_id),
    constraint chk_vehicle_track_vehicle_class check (vehicle_class in ('3WHEELER', 'BUS', 'CAR', 'MOTORCYCLE', 'TRUCK', 'UNKNOWN')),
    constraint chk_vehicle_track_lifecycle_state check (lifecycle_state in ('TENTATIVE', 'ACTIVE', 'TEMPORARILY_LOST', 'COMPLETED', 'DISCARDED')),
    constraint chk_vehicle_track_observation_count check (observation_count >= 0),
    constraint chk_vehicle_track_time_range check (first_seen_at <= last_seen_at),
    constraint chk_vehicle_track_frame_range check (first_frame_number <= last_frame_number),
    constraint chk_vehicle_track_video_time_range check (first_video_time_seconds is null or last_video_time_seconds is null or first_video_time_seconds <= last_video_time_seconds),
    constraint chk_vehicle_track_best_detection_confidence check (best_detection_confidence is null or (best_detection_confidence >= 0 and best_detection_confidence <= 1)),
    constraint chk_vehicle_track_average_detection_confidence check (average_detection_confidence is null or (average_detection_confidence >= 0 and average_detection_confidence <= 1))
);


-- FILE: 009_track_observation.sql

create table if not exists analytics.track_observation (
    id bigint generated by default as identity primary key,
    vehicle_track_id uuid not null,
    camera_id uuid not null,
    frame_number bigint not null,
    observed_at timestamptz not null,
    video_time_seconds numeric,
    bbox_x1 real not null,
    bbox_y1 real not null,
    bbox_x2 real not null,
    bbox_y2 real not null,
    center_x real,
    center_y real,
    detection_confidence real,
    tracker_confidence real,
    is_key_observation boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint fk_track_observation_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_track_observation_camera foreign key (camera_id) references analytics.camera(id),
    constraint uq_track_observation_vehicle_track_frame unique (vehicle_track_id, frame_number),
    constraint chk_track_observation_frame_number check (frame_number >= 0),
    constraint chk_track_observation_bbox_x check (bbox_x2 > bbox_x1),
    constraint chk_track_observation_bbox_y check (bbox_y2 > bbox_y1),
    constraint chk_track_observation_detection_confidence check (detection_confidence is null or (detection_confidence >= 0 and detection_confidence <= 1)),
    constraint chk_track_observation_tracker_confidence check (tracker_confidence is null or (tracker_confidence >= 0 and tracker_confidence <= 1))
);


-- FILE: 010_track_media.sql

create table if not exists analytics.track_media (
    id uuid primary key default gen_random_uuid(),
    vehicle_track_id uuid not null,
    media_type varchar(40) not null,
    storage_provider varchar(30) not null default 'LOCAL',
    storage_uri text not null,
    thumbnail_uri text,
    mime_type varchar(255),
    file_size_bytes bigint,
    checksum_sha256 varchar(64),
    frame_number bigint,
    captured_at timestamptz,
    video_time_seconds numeric,
    bbox jsonb,
    width integer,
    height integer,
    quality_score numeric,
    sharpness_score numeric,
    visibility_score numeric,
    occlusion_score numeric,
    selection_rank integer,
    is_primary boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_track_media_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint chk_track_media_type check (media_type in ('FULL_FRAME', 'VEHICLE_CROP', 'BEST_VEHICLE_CROP', 'PLATE_CROP', 'THUMBNAIL')),
    constraint chk_track_media_storage_provider check (storage_provider in ('LOCAL', 'NAS', 'S3', 'SUPABASE_STORAGE')),
    constraint chk_track_media_file_size check (file_size_bytes is null or file_size_bytes >= 0),
    constraint chk_track_media_frame_number check (frame_number is null or frame_number >= 0),
    constraint chk_track_media_width check (width is null or width > 0),
    constraint chk_track_media_height check (height is null or height > 0),
    constraint chk_track_media_quality_score check (quality_score is null or (quality_score >= 0 and quality_score <= 1)),
    constraint chk_track_media_sharpness_score check (sharpness_score is null or (sharpness_score >= 0 and sharpness_score <= 1)),
    constraint chk_track_media_visibility_score check (visibility_score is null or (visibility_score >= 0 and visibility_score <= 1)),
    constraint chk_track_media_occlusion_score check (occlusion_score is null or (occlusion_score >= 0 and occlusion_score <= 1))
);


-- FILE: 011_vehicle_attribute.sql

create table if not exists analytics.vehicle_attribute (
    id uuid primary key default gen_random_uuid(),
    vehicle_track_id uuid not null,
    attribute_scope varchar(30) not null default 'TRACK',
    primary_color varchar(50),
    secondary_color varchar(50),
    color_confidence numeric,
    make varchar(100),
    make_confidence numeric,
    model varchar(100),
    model_confidence numeric,
    body_type varchar(100),
    body_type_confidence numeric,
    vehicle_class varchar(40),
    class_confidence numeric,
    attribute_source varchar(50),
    attribute_status varchar(30) not null default 'CURRENT',
    observation_count integer not null default 1,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_vehicle_attribute_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint chk_vehicle_attribute_scope check (attribute_scope in ('TRACK', 'GLOBAL')),
    constraint chk_vehicle_attribute_status check (attribute_status in ('CURRENT', 'HISTORICAL', 'REJECTED')),
    constraint chk_vehicle_attribute_vehicle_class check (vehicle_class is null or vehicle_class in ('3WHEELER', 'BUS', 'CAR', 'MOTORCYCLE', 'TRUCK', 'UNKNOWN')),
    constraint chk_vehicle_attribute_color_confidence check (color_confidence is null or (color_confidence >= 0 and color_confidence <= 1)),
    constraint chk_vehicle_attribute_make_confidence check (make_confidence is null or (make_confidence >= 0 and make_confidence <= 1)),
    constraint chk_vehicle_attribute_model_confidence check (model_confidence is null or (model_confidence >= 0 and model_confidence <= 1)),
    constraint chk_vehicle_attribute_body_type_confidence check (body_type_confidence is null or (body_type_confidence >= 0 and body_type_confidence <= 1)),
    constraint chk_vehicle_attribute_class_confidence check (class_confidence is null or (class_confidence >= 0 and class_confidence <= 1)),
    constraint chk_vehicle_attribute_observation_count check (observation_count >= 0)
);


-- FILE: 012_plate_detection.sql

create table if not exists analytics.plate_detection (
    id uuid primary key default gen_random_uuid(),
    vehicle_track_id uuid not null,
    track_observation_id bigint,
    track_media_id uuid,
    detected_at timestamptz not null,
    frame_number bigint,
    bbox_x1 real not null,
    bbox_y1 real not null,
    bbox_x2 real not null,
    bbox_y2 real not null,
    confidence numeric,
    detector_name varchar(100),
    detector_version varchar(100),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_plate_detection_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_plate_detection_track_observation foreign key (track_observation_id) references analytics.track_observation(id),
    constraint fk_plate_detection_track_media foreign key (track_media_id) references analytics.track_media(id),
    constraint chk_plate_detection_frame_number check (frame_number is null or frame_number >= 0),
    constraint chk_plate_detection_bbox_x check (bbox_x2 > bbox_x1),
    constraint chk_plate_detection_bbox_y check (bbox_y2 > bbox_y1),
    constraint chk_plate_detection_confidence check (confidence is null or (confidence >= 0 and confidence <= 1))
);


-- FILE: 013_plate_reading.sql

create table if not exists analytics.plate_reading (
    id uuid primary key default gen_random_uuid(),
    plate_detection_id uuid not null,
    ocr_engine varchar(100),
    ocr_version varchar(100),
    raw_text varchar(100),
    normalized_text varchar(100),
    plate_pattern varchar(100),
    confidence numeric,
    status varchar(30) not null default 'UNKNOWN',
    is_selected boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_plate_reading_plate_detection foreign key (plate_detection_id) references analytics.plate_detection(id),
    constraint chk_plate_reading_confidence check (confidence is null or (confidence >= 0 and confidence <= 1)),
    constraint chk_plate_reading_status check (status in ('VERIFIED', 'PROBABLE', 'PARTIAL', 'UNKNOWN'))
);


-- FILE: 014_plate_summary.sql

create table if not exists analytics.plate_summary (
    id uuid primary key default gen_random_uuid(),
    vehicle_track_id uuid not null,
    selected_plate_reading_id uuid,
    canonical_plate varchar(100),
    plate_pattern varchar(100),
    status varchar(30),
    confidence numeric,
    reading_count integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_plate_summary_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_plate_summary_selected_plate_reading foreign key (selected_plate_reading_id) references analytics.plate_reading(id),
    constraint uq_plate_summary_vehicle_track unique (vehicle_track_id),
    constraint chk_plate_summary_status check (status is null or status in ('VERIFIED', 'PROBABLE', 'PARTIAL', 'UNKNOWN')),
    constraint chk_plate_summary_confidence check (confidence is null or (confidence >= 0 and confidence <= 1)),
    constraint chk_plate_summary_reading_count check (reading_count >= 0)
);


-- FILE: 015_cross_camera_match.sql

create table if not exists analytics.cross_camera_match (
    id uuid primary key default gen_random_uuid(),
    source_track_id uuid not null,
    candidate_track_id uuid not null,
    camera_relation_id uuid,
    time_gap_seconds numeric,
    plate_score numeric,
    color_score numeric,
    class_score numeric,
    temporal_score numeric,
    route_score numeric,
    appearance_score numeric,
    overall_score numeric not null,
    decision varchar(30) not null,
    decision_reason text,
    matcher_version varchar(100),
    reviewed boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    track_pair_min uuid generated always as (least(source_track_id, candidate_track_id)) stored,
    track_pair_max uuid generated always as (greatest(source_track_id, candidate_track_id)) stored,
    constraint fk_cross_camera_match_source_track foreign key (source_track_id) references analytics.vehicle_track(id),
    constraint fk_cross_camera_match_candidate_track foreign key (candidate_track_id) references analytics.vehicle_track(id),
    constraint fk_cross_camera_match_camera_relation foreign key (camera_relation_id) references analytics.camera_relation(id),
    constraint uq_cross_camera_match_directed unique (source_track_id, candidate_track_id),
    constraint uq_cross_camera_match_pair unique (track_pair_min, track_pair_max),
    constraint chk_cross_camera_match_distinct_tracks check (source_track_id <> candidate_track_id),
    constraint chk_cross_camera_match_time_gap check (time_gap_seconds is null or time_gap_seconds >= 0),
    constraint chk_cross_camera_match_plate_score check (plate_score is null or (plate_score >= 0 and plate_score <= 1)),
    constraint chk_cross_camera_match_color_score check (color_score is null or (color_score >= 0 and color_score <= 1)),
    constraint chk_cross_camera_match_class_score check (class_score is null or (class_score >= 0 and class_score <= 1)),
    constraint chk_cross_camera_match_temporal_score check (temporal_score is null or (temporal_score >= 0 and temporal_score <= 1)),
    constraint chk_cross_camera_match_route_score check (route_score is null or (route_score >= 0 and route_score <= 1)),
    constraint chk_cross_camera_match_appearance_score check (appearance_score is null or (appearance_score >= 0 and appearance_score <= 1)),
    constraint chk_cross_camera_match_overall_score check (overall_score >= 0 and overall_score <= 1),
    constraint chk_cross_camera_match_decision check (decision in ('CANDIDATE', 'CONFIRMED', 'PROBABLE', 'AMBIGUOUS', 'REJECTED'))
);


-- FILE: 016_global_vehicle.sql

create table if not exists analytics.global_vehicle (
    id uuid primary key default gen_random_uuid(),
    global_vehicle_code varchar(120) not null,
    canonical_plate varchar(100),
    canonical_color varchar(50),
    canonical_vehicle_class varchar(40),
    first_seen_at timestamptz,
    last_seen_at timestamptz,
    identity_confidence numeric,
    status varchar(30) not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_global_vehicle_code unique (global_vehicle_code),
    constraint chk_global_vehicle_class check (canonical_vehicle_class is null or canonical_vehicle_class in ('3WHEELER', 'BUS', 'CAR', 'MOTORCYCLE', 'TRUCK', 'UNKNOWN')),
    constraint chk_global_vehicle_confidence check (identity_confidence is null or (identity_confidence >= 0 and identity_confidence <= 1)),
    constraint chk_global_vehicle_status check (status in ('CANDIDATE', 'PROBABLE', 'CONFIRMED', 'AMBIGUOUS', 'ARCHIVED')),
    constraint chk_global_vehicle_time_range check (first_seen_at is null or last_seen_at is null or first_seen_at <= last_seen_at)
);


-- FILE: 017_global_vehicle_track.sql

create table if not exists analytics.global_vehicle_track (
    id uuid primary key default gen_random_uuid(),
    global_vehicle_id uuid not null,
    vehicle_track_id uuid not null,
    association_score numeric,
    association_method varchar(100),
    association_status varchar(30),
    is_current boolean not null default true,
    attached_at timestamptz not null default now(),
    detached_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_global_vehicle_track_global_vehicle foreign key (global_vehicle_id) references analytics.global_vehicle(id),
    constraint fk_global_vehicle_track_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint uq_global_vehicle_track_pair unique (global_vehicle_id, vehicle_track_id),
    constraint chk_global_vehicle_track_association_score check (association_score is null or (association_score >= 0 and association_score <= 1)),
    constraint chk_global_vehicle_track_status check (association_status is null or association_status in ('CANDIDATE', 'PROBABLE', 'CONFIRMED', 'AMBIGUOUS', 'REJECTED', 'DETACHED')),
    constraint chk_global_vehicle_track_time_range check (detached_at is null or attached_at <= detached_at)
);


-- FILE: 018_ai_model.sql

create table if not exists analytics.ai_model (
    id uuid primary key default gen_random_uuid(),
    model_code varchar(120) not null,
    model_name varchar(255),
    model_type varchar(100),
    provider varchar(100),
    model_reference text,
    model_version varchar(100),
    checksum varchar(128),
    configuration jsonb not null default '{}'::jsonb,
    active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_ai_model_code unique (model_code)
);


-- FILE: 019_run_model.sql

create table if not exists analytics.run_model (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid not null,
    ai_model_id uuid not null,
    stage_name varchar(100) not null,
    device varchar(100),
    resolved_configuration jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint fk_run_model_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_run_model_ai_model foreign key (ai_model_id) references analytics.ai_model(id),
    constraint uq_run_model_stage unique (processing_run_id, ai_model_id, stage_name)
);


-- FILE: 020_processing_error.sql

create table if not exists analytics.processing_error (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid,
    camera_run_id uuid,
    vehicle_track_id uuid,
    processing_job_id uuid,
    stage_name varchar(100),
    worker_name varchar(255),
    severity varchar(20) not null,
    exception_type varchar(255),
    error_code varchar(100),
    message text not null,
    traceback text,
    frame_number bigint,
    structured_context jsonb not null default '{}'::jsonb,
    resolution_state varchar(30) not null default 'OPEN',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_processing_error_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_processing_error_camera_run foreign key (camera_run_id) references analytics.camera_run(id),
    constraint fk_processing_error_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_processing_error_processing_job foreign key (processing_job_id) references analytics.processing_job(id),
    constraint chk_processing_error_severity check (severity in ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    constraint chk_processing_error_frame_number check (frame_number is null or frame_number >= 0),
    constraint chk_processing_error_resolution_state check (resolution_state in ('OPEN', 'ACKNOWLEDGED', 'RESOLVED', 'IGNORED'))
);


-- FILE: 021_event_candidate.sql

create table if not exists analytics.event_candidate (
    id uuid primary key default gen_random_uuid(),
    processing_run_id uuid,
    camera_run_id uuid,
    camera_id uuid,
    vehicle_track_id uuid,
    global_vehicle_id uuid,
    event_type varchar(80) not null,
    event_status varchar(30) not null default 'CANDIDATE',
    event_time timestamptz not null,
    event_end_time timestamptz,
    confidence numeric,
    event_payload jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_event_candidate_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_event_candidate_camera_run foreign key (camera_run_id) references analytics.camera_run(id),
    constraint fk_event_candidate_camera foreign key (camera_id) references analytics.camera(id),
    constraint fk_event_candidate_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_event_candidate_global_vehicle foreign key (global_vehicle_id) references analytics.global_vehicle(id),
    constraint chk_event_candidate_status check (event_status in ('CANDIDATE', 'PROBABLE', 'CONFIRMED', 'REJECTED')),
    constraint chk_event_candidate_confidence check (confidence is null or (confidence >= 0 and confidence <= 1)),
    constraint chk_event_candidate_time_range check (event_end_time is null or event_time <= event_end_time)
);


-- FILE: 022_analytics_event.sql

create table if not exists analytics.analytics_event (
    id uuid primary key default gen_random_uuid(),
    event_candidate_id uuid,
    processing_run_id uuid,
    camera_id uuid,
    vehicle_track_id uuid,
    global_vehicle_id uuid,
    event_code varchar(120) not null,
    event_type varchar(80) not null,
    event_status varchar(30) not null default 'CONFIRMED',
    event_time timestamptz not null,
    event_end_time timestamptz,
    confidence numeric,
    title varchar(255),
    description text,
    event_payload jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint fk_analytics_event_candidate foreign key (event_candidate_id) references analytics.event_candidate(id),
    constraint fk_analytics_event_processing_run foreign key (processing_run_id) references analytics.processing_run(id),
    constraint fk_analytics_event_camera foreign key (camera_id) references analytics.camera(id),
    constraint fk_analytics_event_vehicle_track foreign key (vehicle_track_id) references analytics.vehicle_track(id),
    constraint fk_analytics_event_global_vehicle foreign key (global_vehicle_id) references analytics.global_vehicle(id),
    constraint uq_analytics_event_code unique (event_code),
    constraint chk_analytics_event_status check (event_status in ('CONFIRMED', 'ACTIVE', 'CLOSED', 'CANCELLED')),
    constraint chk_analytics_event_confidence check (confidence is null or (confidence >= 0 and confidence <= 1)),
    constraint chk_analytics_event_time_range check (event_end_time is null or event_time <= event_end_time)
);


-- FILE: 023_indexes.sql

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


-- FILE: 024_triggers.sql

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


-- FILE: 025_views.sql

create or replace view analytics.searchable_vehicle as
select
    vt.id as vehicle_track_id,
    vt.track_uuid,
    c.camera_code,
    c.camera_name,
    vt.vehicle_class,
    vt.first_seen_at,
    vt.last_seen_at,
    va.primary_color,
    va.make,
    va.model,
    ps.canonical_plate as plate_text,
    ps.status as plate_status,
    ps.confidence as plate_confidence,
    gv.id as global_vehicle_id,
    gv.global_vehicle_code,
    tm.storage_uri as primary_image_uri,
    vt.searchable
from analytics.vehicle_track vt
join analytics.camera c
    on c.id = vt.camera_id
left join analytics.vehicle_attribute va
    on va.vehicle_track_id = vt.id
   and va.attribute_status = 'CURRENT'
left join analytics.plate_summary ps
    on ps.vehicle_track_id = vt.id
left join analytics.global_vehicle_track gvt
    on gvt.vehicle_track_id = vt.id
   and gvt.is_current = true
left join analytics.global_vehicle gv
    on gv.id = gvt.global_vehicle_id
left join analytics.track_media tm
    on tm.vehicle_track_id = vt.id
   and tm.media_type = 'BEST_VEHICLE_CROP'
   and tm.is_primary = true;


-- FILE: 026_permissions.sql

grant usage on schema analytics to authenticated;
grant usage on schema analytics to service_role;

grant select on analytics.searchable_vehicle to authenticated;

grant select on all tables in schema analytics to authenticated;
grant all privileges on all tables in schema analytics to service_role;
grant all privileges on all sequences in schema analytics to service_role;

alter default privileges in schema analytics grant select on tables to authenticated;
alter default privileges in schema analytics grant all privileges on tables to service_role;
alter default privileges in schema analytics grant all privileges on sequences to service_role;


-- FILE: 027_optional_partitioning.sql

-- Optional production partitioning template.
-- Apply only when ingestion volume justifies range partitioning by observed_at.
-- Example approach:
-- 1. Create a partitioned replacement for analytics.track_observation.
-- 2. Partition by RANGE (observed_at).
-- 3. Start with monthly partitions, not daily partitions.
-- 4. Migrate data in a controlled maintenance window.


