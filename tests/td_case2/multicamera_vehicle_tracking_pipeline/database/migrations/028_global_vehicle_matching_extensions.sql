alter table analytics.cross_camera_match
    add column if not exists processing_run_id uuid references analytics.processing_run(id),
    add column if not exists created_global_vehicle_id uuid references analytics.global_vehicle(id),
    add column if not exists rule_version varchar(100);

do $$
begin
    if exists (
        select 1
        from pg_constraint
        where conname = 'chk_cross_camera_match_decision'
          and connamespace = 'analytics'::regnamespace
    ) then
        alter table analytics.cross_camera_match drop constraint chk_cross_camera_match_decision;
    end if;
end;
$$;

alter table analytics.cross_camera_match
    add constraint chk_cross_camera_match_decision
    check (decision in ('CANDIDATE', 'CONFIRMED', 'PROBABLE', 'AMBIGUOUS', 'REJECTED', 'POSSIBLE', 'INSUFFICIENT_EVIDENCE', 'REVIEW_REQUIRED'));

alter table analytics.global_vehicle
    add column if not exists processing_run_id uuid references analytics.processing_run(id),
    add column if not exists object_type varchar(20) not null default 'VEHICLE',
    add column if not exists camera_count integer not null default 0,
    add column if not exists track_count integer not null default 0,
    add column if not exists creation_method varchar(40) not null default 'RULE_BASED';

do $$
begin
    if exists (
        select 1
        from pg_constraint
        where conname = 'chk_global_vehicle_status'
          and connamespace = 'analytics'::regnamespace
    ) then
        alter table analytics.global_vehicle drop constraint chk_global_vehicle_status;
    end if;
end;
$$;

alter table analytics.global_vehicle
    add constraint chk_global_vehicle_status
    check (status in ('ACTIVE', 'CONFIRMED', 'POSSIBLE', 'REVIEW_REQUIRED', 'INVALIDATED', 'CANDIDATE', 'PROBABLE', 'AMBIGUOUS', 'ARCHIVED'));

alter table analytics.global_vehicle
    add constraint chk_global_vehicle_creation_method
    check (creation_method in ('VERIFIED_PLATE', 'RULE_BASED', 'MANUAL', 'SINGLE_TRACK'));

create unique index if not exists uq_global_vehicle_processing_run_code
on analytics.global_vehicle(processing_run_id, global_vehicle_code);

do $$
begin
    if exists (
        select 1
        from pg_constraint
        where conname = 'chk_global_vehicle_track_status'
          and connamespace = 'analytics'::regnamespace
    ) then
        alter table analytics.global_vehicle_track drop constraint chk_global_vehicle_track_status;
    end if;
end;
$$;

alter table analytics.global_vehicle_track
    add constraint chk_global_vehicle_track_status
    check (association_status is null or association_status in ('CANDIDATE', 'PROBABLE', 'CONFIRMED', 'AMBIGUOUS', 'REJECTED', 'DETACHED', 'POSSIBLE', 'REVIEW_REQUIRED'));
