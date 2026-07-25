import type { MediaReference } from './media'

export interface GlobalVehicleListItem {
  global_vehicle_code: string
  run_code?: string | null
  status?: string | null
  canonical_plate?: string | null
  canonical_colour?: string | null
  canonical_vehicle_class?: string | null
  confidence?: number | null
  camera_count?: number | null
  track_count?: number | null
  creation_method?: string | null
  first_seen_at?: string | null
  last_seen_at?: string | null
  primary_evidence_reference?: MediaReference | null
}

export interface GlobalVehicleMember {
  vehicle_track_id?: string | null
  track_uuid?: string | null
  camera_code?: string | null
  vehicle_class?: string | null
  first_seen_at?: string | null
  last_seen_at?: string | null
  association_score?: number | null
  association_method?: string | null
  association_status?: string | null
  is_current?: boolean | null
  attached_at?: string | null
}

export interface GlobalVehicleDetailResponse {
  global_vehicle: {
    global_vehicle_code: string
    run_code?: string | null
    status?: string | null
    canonical_plate?: string | null
    canonical_colour?: string | null
    canonical_vehicle_class?: string | null
    confidence?: number | null
    camera_count?: number | null
    track_count?: number | null
    creation_method?: string | null
    first_seen_at?: string | null
    last_seen_at?: string | null
  }
  members: GlobalVehicleMember[]
  camera_sequence: Array<{
    camera_code?: string | null
    track_uuid?: string | null
    first_seen_at?: string | null
    last_seen_at?: string | null
  }>
  confirmed_matches: unknown[]
  possible_matches: unknown[]
  evidence: MediaReference[]
}
