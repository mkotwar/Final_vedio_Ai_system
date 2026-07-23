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
