grant usage on schema analytics to authenticated;
grant usage on schema analytics to service_role;

grant select on analytics.searchable_vehicle to authenticated;

grant select on all tables in schema analytics to authenticated;
grant all privileges on all tables in schema analytics to service_role;
grant all privileges on all sequences in schema analytics to service_role;

alter default privileges in schema analytics grant select on tables to authenticated;
alter default privileges in schema analytics grant all privileges on tables to service_role;
alter default privileges in schema analytics grant all privileges on sequences to service_role;
