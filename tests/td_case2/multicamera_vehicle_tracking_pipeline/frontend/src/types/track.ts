import type { MediaReference } from './media'
import type { PlateResult } from './plate'

export interface TrackListItem {
  track_uuid: string
  camera_code?: string | null
  local_track_id?: number | null
  vehicle_class?: string | null
  lifecycle_state?: string | null
  first_seen_at?: string | null
  last_seen_at?: string | null
  first_video_time_seconds?: number | null
  last_video_time_seconds?: number | null
  observation_count?: number | null
  best_detection_confidence?: number | null
  average_detection_confidence?: number | null
  class_confidence?: number | null
  class_is_stable?: boolean | null
  class_observation_count?: number | null
  primary_colour?: string | null
  colour_confidence?: number | null
  plate_result?: PlateResult | null
  canonical_plate?: string | null
  plate_status?: string | null
  plate_confidence?: number | null
  primary_media?: MediaReference | null
  primary_vehicle_media?: MediaReference | null
  primary_plate_media?: MediaReference | null
}

export interface ObservationItem {
  frame_number: number
  timestamp?: string | null
  video_time_seconds?: number | null
  bbox: {
    x1?: number | null
    y1?: number | null
    x2?: number | null
    y2?: number | null
  }
  detection_confidence?: number | null
  tracker_confidence?: number | null
  is_key_observation: boolean
  class_name?: string | null
  raw_class_name?: string | null
}

export interface TrackDetailResponse {
  track: TrackListItem
  camera: {
    camera_code?: string | null
    camera_name?: string | null
    location?: string | null
  }
  colour: {
    primary_colour?: string | null
    colour_confidence?: number | null
  }
  plate: {
    plate_result?: PlateResult | null
    canonical_plate?: string | null
    plate_status?: string | null
    plate_confidence?: number | null
  }
  media: MediaReference[]
  observation_summary: {
    count: number
    first_frame?: number | null
    last_frame?: number | null
    key_observation_count: number
  }
  class_diagnostics?: {
    provisional_class_name?: string | null
    stable_class_name?: string | null
    class_is_locked?: boolean
    class_confidence?: number | null
    class_winner_margin?: number | null
    class_observation_count?: number | null
    class_conflict_count?: number | null
    class_scores?: Record<string, number>
    class_observation_counts?: Record<string, number>
    class_max_confidences?: Record<string, number>
    latest_observation_class_name?: string | null
  } | null
  global_membership?: {
    linked: boolean
    global_vehicle_id?: string | null
    global_vehicle_code?: string | null
    membership_confidence?: number | null
    membership_status?: string | null
    member_track_count?: number | null
  } | null
  cross_camera_matches: Array<{
    id?: string | null
    decision?: string | null
    overall_score?: number | null
    source_track_id?: string | null
    candidate_track_id?: string | null
    created_global_vehicle_id?: string | null
  }>
  errors: Array<{
    id?: string | null
    stage_name?: string | null
    error_code?: string | null
    message?: string | null
    severity?: string | null
    created_at?: string | null
  }>
}

export interface TrackFilters {
  runCode: string
  cameraCode?: string
  vehicleClass?: string
  colour?: string
  plate?: string
  plateStatus?: string
  lifecycleState?: string
  minimumConfidence?: number
  hasMedia?: boolean
}
