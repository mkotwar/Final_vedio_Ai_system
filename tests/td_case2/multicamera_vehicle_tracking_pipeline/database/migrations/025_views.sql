create or replace view analytics.searchable_vehicle as
select
    vt.id as vehicle_track_id,
    vt.track_uuid,
    c.camera_code,
    c.camera_name,
    vt.vehicle_class,
    vt.first_seen_at,
    vt.last_seen_at,
    va.primary_color,
    va.make,
    va.model,
    ps.canonical_plate as plate_text,
    ps.status as plate_status,
    ps.confidence as plate_confidence,
    gv.id as global_vehicle_id,
    gv.global_vehicle_code,
    tm.storage_uri as primary_image_uri,
    vt.searchable
from analytics.vehicle_track vt
join analytics.camera c
    on c.id = vt.camera_id
left join analytics.vehicle_attribute va
    on va.vehicle_track_id = vt.id
   and va.attribute_status = 'CURRENT'
left join analytics.plate_summary ps
    on ps.vehicle_track_id = vt.id
left join analytics.global_vehicle_track gvt
    on gvt.vehicle_track_id = vt.id
   and gvt.is_current = true
left join analytics.global_vehicle gv
    on gv.id = gvt.global_vehicle_id
left join analytics.track_media tm
    on tm.vehicle_track_id = vt.id
   and tm.media_type = 'BEST_VEHICLE_CROP'
   and tm.is_primary = true;
