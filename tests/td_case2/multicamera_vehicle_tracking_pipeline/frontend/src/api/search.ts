import { apiGet, apiPost } from './client'
import type {
  NaturalLanguageParseResponse,
  NaturalLanguageSearchRequest,
  NaturalLanguageSearchResponse,
  VehicleSearchFilters,
  VehicleSearchResponse,
} from '../types/search'

export function searchVehicles(params: VehicleSearchFilters) {
  const queryParams: Record<string, string | number | boolean | null | undefined> = {}

  for (const [key, value] of Object.entries(params)) {
    queryParams[key] = value
  }

  return apiGet<VehicleSearchResponse>('/search/vehicles', {
    params: queryParams,
  })
}

export function searchVehiclesNaturalLanguage(request: NaturalLanguageSearchRequest) {
  return apiPost<NaturalLanguageSearchResponse>('/search/natural-language', {
    body: request,
  })
}

export function parseVehicleSearchNaturalLanguage(request: NaturalLanguageSearchRequest) {
  return apiPost<NaturalLanguageParseResponse>('/search/natural-language/parse', {
    body: request,
  })
}
