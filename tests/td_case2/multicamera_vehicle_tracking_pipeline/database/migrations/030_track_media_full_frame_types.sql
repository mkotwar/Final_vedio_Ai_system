-- Additive migration for live analytics.track_media media_type validation.
-- This preserves every previously accepted media type and adds the new
-- full-frame evidence categories used by the multicamera evidence pipeline.
--
-- Rollback guidance:
-- 1. Delete any rows using media_type = 'ANNOTATED_FULL_FRAME' if rollback is required.
-- 2. Recreate chk_track_media_type with the historical value list only.

alter table if exists analytics.track_media
drop constraint if exists chk_track_media_type;

alter table if exists analytics.track_media
add constraint chk_track_media_type
check (
    media_type in (
        'FULL_FRAME',
        'ANNOTATED_FULL_FRAME',
        'VEHICLE_CROP',
        'BEST_VEHICLE_CROP',
        'PLATE_CROP',
        'THUMBNAIL'
    )
);
