import { describe, expect, it, vi } from 'vitest'
import { apiGet, ApiClientError, getApiBaseUrl } from '../api/client'

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
})
