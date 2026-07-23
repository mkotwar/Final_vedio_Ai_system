-- DEVELOPMENT ONLY
-- This file is destructive and must never be run against a production database.
-- It removes the old simplified public-schema analytics prototype only.

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
drop function if exists public.validate_vehicle_match_cameras() cascade;
