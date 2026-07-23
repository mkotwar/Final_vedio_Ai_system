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
