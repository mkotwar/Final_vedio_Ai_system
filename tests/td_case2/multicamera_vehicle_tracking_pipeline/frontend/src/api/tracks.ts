import { apiGet } from './client'
import type { PaginatedResponse } from '../types/common'
import type { MediaReference } from '../types/media'
import type { ObservationItem, TrackDetailResponse, TrackListItem } from '../types/track'

export function listTracks(runCode: string, params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<TrackListItem>>(`/runs/${encodeURIComponent(runCode)}/tracks`, { params })
}

export function getTrack(trackUuid: string) {
  return apiGet<TrackDetailResponse>(`/tracks/${encodeURIComponent(trackUuid)}`)
}

export function listTrackObservations(trackUuid: string, params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<ObservationItem>>(`/tracks/${encodeURIComponent(trackUuid)}/observations`, { params })
}

export function listTrackMedia(trackUuid: string) {
  return apiGet<MediaReference[]>(`/tracks/${encodeURIComponent(trackUuid)}/media`)
}
