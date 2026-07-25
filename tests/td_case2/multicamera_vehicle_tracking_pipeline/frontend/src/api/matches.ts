import { apiGet } from './client'
import type { PaginatedResponse } from '../types/common'
import type { MatchDetailResponse, MatchListItem } from '../types/match'

export function listMatches(params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<MatchListItem>>('/cross-camera-matches', { params })
}

export function getMatch(matchId: string) {
  return apiGet<MatchDetailResponse>(`/cross-camera-matches/${encodeURIComponent(matchId)}`)
}
