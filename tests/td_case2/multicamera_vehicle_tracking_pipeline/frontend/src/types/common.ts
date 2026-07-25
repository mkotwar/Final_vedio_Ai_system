export interface PaginatedResponse<T> {
  items: T[]
  page: number
  page_size: number
  total: number
  has_next: boolean
}

export interface ApiErrorBody {
  code: string
  message: string
  details: unknown
}

export interface ApiErrorResponse {
  error: ApiErrorBody
}

export interface HealthResponse {
  status: string
  service: string
  database: string
  schema: string
}

export type SortOrder = 'asc' | 'desc'

export interface QueryListParams {
  page?: number
  page_size?: number
}
