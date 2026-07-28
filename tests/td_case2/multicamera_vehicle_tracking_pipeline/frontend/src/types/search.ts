import type { MediaReference } from './media'
import type { PlateResult } from './plate'

export type SearchResultScope = 'LOCAL_TRACKS' | 'GLOBAL_VEHICLES' | 'ALL'
export type PlateMatchType = 'EXACT' | 'CONTAINS' | 'STARTS_WITH' | 'ENDS_WITH'
export type VehicleSearchSortBy = 'RELEVANCE' | 'FIRST_SEEN' | 'LAST_SEEN' | 'CONFIDENCE' | 'PLATE'
export type VehicleSearchSortOrder = 'ASC' | 'DESC'

export interface VehicleSearchFilters {
  run_code?: string
  result_scope?: SearchResultScope
  vehicle_class?: string
  colour?: string
  plate?: string
  plate_match_type?: PlateMatchType
  camera_codes?: string
  date?: string
  start_time?: string
  end_time?: string
  minimum_confidence?: string
  multi_camera_only?: boolean
  verified_plate_only?: boolean
  limit?: number
  offset?: number
  sort_by?: VehicleSearchSortBy
  sort_order?: VehicleSearchSortOrder
}

export interface VehicleSearchResultItem {
  result_type: 'LOCAL_TRACK' | 'GLOBAL_VEHICLE'
  global_vehicle_code?: string | null
  track_uuid?: string | null
  class_name?: string | null
  colour?: string | null
  plate_result?: PlateResult | null
  plate?: string | null
  plate_status?: string | null
  camera_codes: string[]
  first_seen_at?: string | null
  last_seen_at?: string | null
  confidence?: number | null
  class_confidence?: number | null
  class_is_stable?: boolean | null
  class_observation_count?: number | null
  member_track_count?: number | null
  primary_media?: MediaReference | null
  primary_vehicle_media?: MediaReference | null
  primary_plate_media?: MediaReference | null
  primary_full_frame_media?: MediaReference | null
  primary_annotated_full_frame_media?: MediaReference | null
  match_reasons: string[]
  relevance_score?: number | null
}

export interface VehicleSearchResponse {
  filters: Record<string, unknown>
  pagination: {
    limit: number
    offset: number
    returned: number
    total: number
    has_more: boolean
  }
  results: VehicleSearchResultItem[]
}

export interface NaturalLanguageSearchRequest {
  query: string
  run_code?: string
  result_scope?: SearchResultScope
  default_time_tolerance_minutes?: number
  limit?: number
  offset?: number
}

export interface InterpretedVehicleSearchFilters {
  run_code?: string
  result_scope?: SearchResultScope
  vehicle_class?: string
  colour?: string
  plate?: string
  plate_match_type?: PlateMatchType
  camera_codes?: string[]
  date?: string
  start_time?: string
  end_time?: string
  target_time?: string
  time_tolerance_minutes?: number
  minimum_confidence?: number
  multi_camera_only?: boolean
  verified_plate_only?: boolean
  sort_by?: VehicleSearchSortBy
  sort_order?: VehicleSearchSortOrder
  clarification_required?: boolean
  clarification_message?: string | null
  limit?: number
  offset?: number
}

export interface NaturalLanguageParserMetadata {
  provider: string
  model?: string | null
  fallback_used: boolean
}

export interface ParsedVehicleSearchIntent extends InterpretedVehicleSearchFilters {}

export interface NaturalLanguageSearchResponse {
  original_query: string
  parser: NaturalLanguageParserMetadata
  clarification_required: boolean
  clarification_message?: string | null
  interpreted_filters: InterpretedVehicleSearchFilters
  pagination: VehicleSearchResponse['pagination']
  results: VehicleSearchResultItem[]
}

export interface NaturalLanguageParseResponse {
  original_query: string
  parser: NaturalLanguageParserMetadata
  parsed_intent: ParsedVehicleSearchIntent
  interpreted_filters: InterpretedVehicleSearchFilters
  clarification_required: boolean
  clarification_message?: string | null
}
