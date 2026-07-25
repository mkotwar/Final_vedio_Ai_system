export interface MediaReference {
  media_id?: string | null
  media_type?: string | null
  storage_provider?: string | null
  storage_uri?: string | null
  frame_number?: number | null
  captured_at?: string | null
  video_time_seconds?: number | null
  width?: number | null
  height?: number | null
  quality_score?: number | null
  sharpness_score?: number | null
  visibility_score?: number | null
  selection_rank?: number | null
  is_primary?: boolean | null
}

export interface MediaDeliveryResponse {
  media_id: string
  availability: string
  storage_uri?: string | null
  media_type?: string | null
}
