export interface RunListItem {
  id?: string | null
  run_code: string
  status?: string | null
  started_at?: string | null
  completed_at?: string | null
  created_at?: string | null
  camera_count: number
  track_count: number
  global_vehicle_count: number
  processing_error_count: number
}

export interface RunDetailResponse {
  id?: string | null
  run_code: string
  status?: string | null
  started_at?: string | null
  completed_at?: string | null
  created_at?: string | null
  pipeline_name?: string | null
  pipeline_version?: string | null
  execution_mode?: string | null
  runtime_device?: string | null
  camera_summary?: {
    configured_camera_count: number
    active_camera_count: number
    camera_run_count: number
    completed_camera_runs: number
  }
  track_summary?: {
    track_count: number
    total_track_observations: number
  }
  enrichment_summary?: {
    tracks_with_colour: number
    tracks_with_plate_summary: number
    tracks_with_media: number
  }
  global_object_summary?: {
    global_vehicle_count: number
  }
  processing_error_summary?: {
    processing_error_count: number
  }
}

export interface CameraListItem {
  id?: string | null
  camera_code?: string | null
  camera_name?: string | null
  location?: string | null
  camera_run_status?: string | null
  frames_read: number
  frames_processed: number
  detection_count: number
  completed_track_count: number
  discarded_track_count: number
}

export interface CameraDetailResponse extends CameraListItem {
  track_count: number
  colour_coverage: number
  plate_coverage: number
  media_coverage: number
  processing_errors: Array<{
    id?: string | null
    stage_name?: string | null
    error_code?: string | null
    message?: string | null
    created_at?: string | null
  }>
}
