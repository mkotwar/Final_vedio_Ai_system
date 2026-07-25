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
}

export type MatchDetailResponse = MatchListItem & {
  metadata?: Record<string, unknown>
}
