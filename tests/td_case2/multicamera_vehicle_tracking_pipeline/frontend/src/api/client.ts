import type { ApiErrorResponse } from '../types/common'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000/api/v1'
const DEFAULT_TIMEOUT_MS = 15000

export class ApiClientError extends Error {
  status: number
  code: string
  details: unknown

  constructor(message: string, options: { status?: number; code?: string; details?: unknown } = {}) {
    super(message)
    this.name = 'ApiClientError'
    this.status = options.status ?? 500
    this.code = options.code ?? 'API_REQUEST_FAILED'
    this.details = options.details ?? null
  }
}

export function getApiBaseUrl(): string {
  const candidate = import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL
  try {
    const url = new URL(candidate)
    return url.toString().replace(/\/$/, '')
  } catch {
    throw new ApiClientError('Frontend API configuration is invalid.', {
      status: 500,
      code: 'INVALID_API_BASE_URL',
      details: null,
    })
  }
}

export function buildApiUrl(path: string, params?: Record<string, string | number | boolean | undefined | null>): string {
  const baseUrl = getApiBaseUrl()
  const url = new URL(`${baseUrl}${path.startsWith('/') ? path : `/${path}`}`)
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === '') {
        return
      }
      url.searchParams.set(key, String(value))
    })
  }
  return url.toString()
}

export function resolveApiAssetUrl(path: string): string {
  const trimmed = path.trim()
  if (!trimmed) {
    throw new ApiClientError('Media URL is invalid.', {
      status: 500,
      code: 'INVALID_MEDIA_URL',
      details: null,
    })
  }
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    return trimmed
  }
  const apiOrigin = new URL(getApiBaseUrl()).origin
  return new URL(trimmed.startsWith('/') ? trimmed : `/${trimmed}`, apiOrigin).toString()
}

async function parseApiError(response: Response): Promise<ApiClientError> {
  let payload: ApiErrorResponse | null = null
  try {
    payload = (await response.json()) as ApiErrorResponse
  } catch {
    payload = null
  }
  return new ApiClientError(
    payload?.error.message || `Request failed with status ${response.status}.`,
    {
      status: response.status,
      code: payload?.error.code || 'API_REQUEST_FAILED',
      details: payload?.error.details ?? null,
    },
  )
}

export async function apiGet<T>(
  path: string,
  options: {
    params?: Record<string, string | number | boolean | undefined | null>
    signal?: AbortSignal
    timeoutMs?: number
  } = {},
): Promise<T> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS)

  if (options.signal) {
    options.signal.addEventListener('abort', () => controller.abort(), { once: true })
  }

  try {
    const response = await fetch(buildApiUrl(path, options.params), {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: controller.signal,
    })

    if (!response.ok) {
      throw await parseApiError(response)
    }

    return (await response.json()) as T
  } catch (error) {
    if (error instanceof ApiClientError) {
      throw error
    }
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiClientError('The request timed out or was cancelled.', {
        status: 408,
        code: 'REQUEST_TIMEOUT',
        details: null,
      })
    }
    throw new ApiClientError('The backend is unavailable.', {
      status: 503,
      code: 'BACKEND_UNAVAILABLE',
      details: null,
    })
  } finally {
    window.clearTimeout(timeoutId)
  }
}
