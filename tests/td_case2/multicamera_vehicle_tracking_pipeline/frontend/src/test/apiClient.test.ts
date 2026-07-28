import { describe, expect, it, vi } from 'vitest'
import { apiGet, ApiClientError, describeApiClientError, getApiBaseUrl } from '../api/client'

describe('api client', () => {
  it('uses the configured API base URL', () => {
    expect(getApiBaseUrl()).toBe('http://127.0.0.1:8000/api/v1')
  })

  it('parses a paginated response', async () => {
    const response = await apiGet<{ items: unknown[]; total: number }>('/runs')
    expect(response.total).toBe(1)
  })

  it('parses backend errors into ApiClientError', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: { code: 'RUN_NOT_FOUND', message: 'Run was not found.', details: null },
        }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    )

    await expect(apiGet('/runs')).rejects.toMatchObject({
      code: 'RUN_NOT_FOUND',
      status: 404,
    } satisfies Partial<ApiClientError>)
  })

  it('describes backend-unavailable failures with the configured API origin', () => {
    const description = describeApiClientError(
      new ApiClientError('The backend is unavailable.', {
        status: 503,
        code: 'BACKEND_UNAVAILABLE',
        details: { apiBaseUrl: 'http://127.0.0.1:8000/api/v1' },
      }),
    )
    expect(description.message).toBe('Backend API unavailable at http://127.0.0.1:8000')
  })

  it('describes backend query failures separately', () => {
    const description = describeApiClientError(
      new ApiClientError('A database query failed.', {
        status: 502,
        code: 'DATABASE_QUERY_FAILED',
      }),
    )
    expect(description.message).toBe('Search service returned an error.')
  })
})
