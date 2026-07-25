import { apiGet } from './client'
import type { PaginatedResponse } from '../types/common'
import type { CameraDetailResponse, CameraListItem, RunDetailResponse, RunListItem } from '../types/run'

export function listRuns(params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<RunListItem>>('/runs', { params })
}

export function getRun(runCode: string) {
  return apiGet<RunDetailResponse>(`/runs/${encodeURIComponent(runCode)}`)
}

export function listRunCameras(runCode: string, params: Record<string, string | number | boolean | undefined>) {
  return apiGet<PaginatedResponse<CameraListItem>>(`/runs/${encodeURIComponent(runCode)}/cameras`, { params })
}

export function getRunCamera(runCode: string, cameraCode: string) {
  return apiGet<CameraDetailResponse>(`/runs/${encodeURIComponent(runCode)}/cameras/${encodeURIComponent(cameraCode)}`)
}
