import type { MediaReference } from './media'

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
  primary_colour?: string | null
  colour_confidence?: number | null
  canonical_plate?: string | null
  plate_status?: string | null
  plate_confidence?: number | null
  primary_media?: MediaReference | null
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
  global_membership?: {
    global_vehicle_id?: string | null
    global_vehicle_code?: string | null
    association_status?: string | null
    association_score?: number | null
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
