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
