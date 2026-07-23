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
