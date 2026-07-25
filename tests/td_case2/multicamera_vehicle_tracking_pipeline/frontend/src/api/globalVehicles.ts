import { apiGet } from './client'
import type { PaginatedResponse } from '../types/common'
import type { GlobalVehicleDetailResponse, GlobalVehicleListItem, GlobalVehicleMember } from '../types/globalVehicle'

export function listGlobalVehicles(params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<GlobalVehicleListItem>>('/global-vehicles', { params })
}

export function getGlobalVehicle(globalVehicleCode: string) {
  return apiGet<GlobalVehicleDetailResponse>(`/global-vehicles/${encodeURIComponent(globalVehicleCode)}`)
}

export function listGlobalVehicleTracks(globalVehicleCode: string) {
  return apiGet<GlobalVehicleMember[]>(`/global-vehicles/${encodeURIComponent(globalVehicleCode)}/tracks`)
}
