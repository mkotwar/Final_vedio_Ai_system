-- Destructive test reset migration for the proof-of-concept schema.
-- Use this only for local or development Supabase projects.

create extension if not exists pgcrypto;
create extension if not exists pg_trgm;

drop view if exists public.searchable_vehicle_tracks cascade;
drop view if exists public.global_vehicle_timeline cascade;
drop view if exists public.pending_association_review cascade;
drop view if exists public.searchable_vehicles cascade;

drop table if exists public.pipeline_errors cascade;
drop table if exists public.processing_jobs cascade;
drop table if exists public.association_decisions cascade;
drop table if exists public.association_candidates cascade;
drop table if exists public.global_vehicle_tracks cascade;
drop table if exists public.global_vehicles cascade;
drop table if exists public.track_evidence cascade;
drop table if exists public.plate_readings cascade;
drop table if exists public.vehicle_track_observations cascade;
drop table if exists public.processing_windows cascade;
drop table if exists public.vms_recordings cascade;
drop table if exists public.stream_sessions cascade;
drop table if exists public.camera_connections cascade;
drop table if exists public.vehicle_matches cascade;
drop table if exists public.vehicle_observations cascade;
drop table if exists public.vehicle_attributes cascade;
drop table if exists public.vehicle_tracks cascade;
drop table if exists public.cameras cascade;

drop function if exists public.refresh_global_vehicle_summary() cascade;
drop function if exists public.set_updated_at() cascade;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create table public.cameras (
    id uuid primary key default gen_random_uuid(),
    camera_code text not null unique,
    camera_name text,
    source_path text,
    enabled boolean not null default true,
    created_at timestamptz not null default now()
);

create table public.vehicle_tracks (
    id uuid primary key default gen_random_uuid(),
    track_uuid text not null unique,
    camera_id uuid not null references public.cameras(id) on delete cascade,
    local_track_id integer not null,
    vehicle_class text not null check (vehicle_class in ('car', 'bus', 'truck', 'motorcycle', 'unknown')),
    first_seen_at timestamptz not null,
    last_seen_at timestamptz not null,
    first_frame_number integer,
    last_frame_number integer,
    observation_count integer not null default 0,
    best_confidence real,
    best_frame_path text,
    best_crop_path text,
    created_at timestamptz not null default now(),
    unique (camera_id, local_track_id, first_seen_at),
    check (last_seen_at >= first_seen_at),
    check (first_frame_number is null or first_frame_number >= 0),
    check (last_frame_number is null or last_frame_number >= 0),
    check (last_frame_number is null or first_frame_number is null or last_frame_number >= first_frame_number)
);

create table public.vehicle_attributes (
    id uuid primary key default gen_random_uuid(),
    vehicle_track_id uuid not null unique references public.vehicle_tracks(id) on delete cascade,
    vehicle_colour text,
    colour_confidence real,
    plate_text text,
    plate_pattern text,
    plate_confidence real,
    plate_verified boolean not null default false,
    plate_readings jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.vehicle_observations (
    id bigint generated always as identity primary key,
    vehicle_track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    frame_number integer not null,
    observed_at timestamptz not null,
    bbox_x1 real not null,
    bbox_y1 real not null,
    bbox_x2 real not null,
    bbox_y2 real not null,
    confidence real,
    created_at timestamptz not null default now(),
    check (frame_number >= 0),
    check (bbox_x2 > bbox_x1),
    check (bbox_y2 > bbox_y1)
);

create table public.vehicle_matches (
    id uuid primary key default gen_random_uuid(),
    source_track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    candidate_track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    plate_similarity real,
    colour_match boolean not null default false,
    class_match boolean not null default false,
    time_gap_seconds real,
    match_score real,
    match_status text not null check (match_status in ('confirmed', 'probable', 'ambiguous', 'rejected')),
    created_at timestamptz not null default now(),
    unique (source_track_id, candidate_track_id),
    check (source_track_id <> candidate_track_id)
);

create or replace function public.validate_vehicle_match_cameras()
returns trigger
language plpgsql
as $$
declare
    source_camera uuid;
    candidate_camera uuid;
begin
    select camera_id into source_camera from public.vehicle_tracks where id = new.source_track_id;
    select camera_id into candidate_camera from public.vehicle_tracks where id = new.candidate_track_id;
    if source_camera is null or candidate_camera is null then
        raise exception 'vehicle match references missing track rows';
    end if;
    if source_camera = candidate_camera then
        raise exception 'vehicle match source and candidate must belong to different cameras';
    end if;
    return new;
end;
$$;

create trigger trg_vehicle_attributes_updated_at
before update on public.vehicle_attributes
for each row execute function public.set_updated_at();

create trigger trg_vehicle_matches_validate_cameras
before insert or update on public.vehicle_matches
for each row execute function public.validate_vehicle_match_cameras();

create index idx_vehicle_tracks_camera_first_seen on public.vehicle_tracks(camera_id, first_seen_at);
create index idx_vehicle_tracks_vehicle_class on public.vehicle_tracks(vehicle_class);
create index idx_vehicle_attributes_plate_text on public.vehicle_attributes(plate_text);
create index idx_vehicle_attributes_vehicle_colour on public.vehicle_attributes(vehicle_colour);
create index idx_vehicle_matches_source_track_id on public.vehicle_matches(source_track_id);
create index idx_vehicle_matches_candidate_track_id on public.vehicle_matches(candidate_track_id);
create index idx_vehicle_matches_match_score_desc on public.vehicle_matches(match_score desc);
create index idx_vehicle_observations_track_time on public.vehicle_observations(vehicle_track_id, observed_at);
create index idx_vehicle_attributes_plate_text_trgm on public.vehicle_attributes using gin (plate_text gin_trgm_ops);

create or replace view public.searchable_vehicles as
select
    vt.id as track_id,
    vt.track_uuid,
    c.camera_code,
    c.camera_name,
    vt.vehicle_class,
    vt.first_seen_at,
    vt.last_seen_at,
    va.vehicle_colour,
    va.plate_text,
    va.plate_pattern,
    va.plate_confidence,
    vt.best_frame_path,
    vt.best_crop_path
from public.vehicle_tracks vt
join public.cameras c on c.id = vt.camera_id
left join public.vehicle_attributes va on va.vehicle_track_id = vt.id;

alter table public.cameras enable row level security;
alter table public.vehicle_tracks enable row level security;
alter table public.vehicle_attributes enable row level security;
alter table public.vehicle_observations enable row level security;
alter table public.vehicle_matches enable row level security;

create policy authenticated_read_cameras on public.cameras for select to authenticated using (true);
create policy authenticated_read_vehicle_tracks on public.vehicle_tracks for select to authenticated using (true);
create policy authenticated_read_vehicle_attributes on public.vehicle_attributes for select to authenticated using (true);
create policy authenticated_read_vehicle_observations on public.vehicle_observations for select to authenticated using (true);
create policy authenticated_read_vehicle_matches on public.vehicle_matches for select to authenticated using (true);

create policy service_role_all_cameras on public.cameras for all to service_role using (true) with check (true);
create policy service_role_all_vehicle_tracks on public.vehicle_tracks for all to service_role using (true) with check (true);
create policy service_role_all_vehicle_attributes on public.vehicle_attributes for all to service_role using (true) with check (true);
create policy service_role_all_vehicle_observations on public.vehicle_observations for all to service_role using (true) with check (true);
create policy service_role_all_vehicle_matches on public.vehicle_matches for all to service_role using (true) with check (true);
