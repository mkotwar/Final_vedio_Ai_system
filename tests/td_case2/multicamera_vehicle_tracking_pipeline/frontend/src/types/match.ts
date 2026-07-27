import type { PlateResult } from './plate'

export interface MatchListItem {
  id?: string | null
  source_track_uuid?: string | null
  candidate_track_uuid?: string | null
  source_camera_code?: string | null
  candidate_camera_code?: string | null
  decision?: string | null
  overall_score?: number | null
  plate_score?: number | null
  route_score?: number | null
  time_score?: number | null
  class_score?: number | null
  colour_score?: number | null
  visual_score?: number | null
  decision_reasons: string[]
  rule_version?: string | null
  linked_global_vehicle_code?: string | null
  source_track?: {
    track_uuid?: string | null
    camera_code?: string | null
    vehicle_class?: string | null
    lifecycle_state?: string | null
    first_seen_at?: string | null
    last_seen_at?: string | null
    best_detection_confidence?: number | null
    primary_colour?: string | null
    colour_confidence?: number | null
    plate_result?: PlateResult | null
    canonical_plate?: string | null
    plate_status?: string | null
    plate_confidence?: number | null
    primary_media?: import('./media').MediaReference | null
    primary_vehicle_media?: import('./media').MediaReference | null
    primary_plate_media?: import('./media').MediaReference | null
  } | null
  candidate_track?: {
    track_uuid?: string | null
    camera_code?: string | null
    vehicle_class?: string | null
    lifecycle_state?: string | null
    first_seen_at?: string | null
    last_seen_at?: string | null
    best_detection_confidence?: number | null
    primary_colour?: string | null
    colour_confidence?: number | null
    plate_result?: PlateResult | null
    canonical_plate?: string | null
    plate_status?: string | null
    plate_confidence?: number | null
    primary_media?: import('./media').MediaReference | null
    primary_vehicle_media?: import('./media').MediaReference | null
    primary_plate_media?: import('./media').MediaReference | null
  } | null
}

export type MatchDetailResponse = MatchListItem & {
  metadata?: Record<string, unknown>
}
