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
