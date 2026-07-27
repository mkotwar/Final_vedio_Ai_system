export interface MediaReference {
  media_id?: string | null
  media_type?: string | null
  availability?: string | null
  content_url?: string | null
  thumbnail_url?: string | null
  track_uuid?: string | null
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
  error_detail?: string | null
}

export interface MediaDeliveryResponse {
  media_id: string
  availability: string
  media_type?: string | null
  content_url?: string | null
  thumbnail_url?: string | null
  track_uuid?: string | null
  frame_number?: number | null
  width?: number | null
  height?: number | null
  quality_score?: number | null
  sharpness_score?: number | null
  visibility_score?: number | null
  selection_rank?: number | null
  is_primary?: boolean | null
  error_detail?: string | null
}

export interface MediaSignedUrlResponse {
  media_id: string
  availability: string
  url?: string | null
  expires_in?: number | null
}
