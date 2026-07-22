# Supabase Setup

This setup is for a development or test Supabase project only.

## Migration method used

The schema was simplified by replacing the earlier large design with a new destructive reset migration:

`database/migrations/simplified_schema.sql`

It drops the earlier proof-of-concept tables and views if they exist, then creates the smaller five-table schema. This is appropriate for test data resets, but not for preserving old experimental data.

## Exact setup steps

1. Create or choose a development Supabase project.
2. Set:
   - `supabase_database_url`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_ANON_KEY`
   - `DATABASE_SCHEMA_VERSION=simplified_schema`
3. Link the project with the Supabase CLI.
4. Reset the local/dev database.
5. Push the simplified schema.

## Exact commands

```powershell
supabase login
supabase link --project-ref <your-project-ref>
supabase db reset
supabase db push
python -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.check_supabase_connection
```

## Validation command

```powershell
python -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_database_schema
```

## Security note

RLS is intentionally simple for testing:

- authenticated users can read all rows
- `service_role` can perform all operations
- anonymous users are not given write policies

Keep the service-role key backend-only.
