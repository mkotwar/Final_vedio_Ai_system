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
