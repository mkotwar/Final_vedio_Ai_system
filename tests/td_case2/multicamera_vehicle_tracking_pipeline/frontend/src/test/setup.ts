import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import {
  camerasFixture,
  globalVehicleDetailFixture,
  globalVehicleMembersFixture,
  globalVehiclesFixture,
  healthFixture,
  matchesFixture,
  mediaFixture,
  naturalLanguageVehicleSearchFixture,
  observationsFixture,
  runDetailFixture,
  runsFixture,
  trackDetailFixture,
  tracksFixture,
  vehicleSearchFixture,
} from './fixtures'

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: {
        'Content-Type': 'application/json',
      },
    }),
  )
}

function buildMockFetch() {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input.toString()
    const url = new URL(raw)
    const path = decodeURIComponent(url.pathname)

    if (path.endsWith('/health')) {
      return jsonResponse(healthFixture)
    }
    if (path.endsWith('/runs') && !path.includes('/runs/')) {
      return jsonResponse(runsFixture)
    }
    if (path.endsWith('/runs/RUN_20260724_151402')) {
      return jsonResponse(runDetailFixture)
    }
    if (path.endsWith('/runs/RUN_20260724_151402/cameras')) {
      return jsonResponse(camerasFixture)
    }
    if (path.endsWith('/runs/RUN_20260724_151402/tracks')) {
      return jsonResponse(tracksFixture)
    }
    if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4')) {
      return jsonResponse(trackDetailFixture)
    }
    if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4/observations')) {
      return jsonResponse(observationsFixture)
    }
    if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4/media')) {
      return jsonResponse(mediaFixture)
    }
    if (path.endsWith('/global-vehicles') && !path.includes('/global-vehicles/')) {
      return jsonResponse(globalVehiclesFixture)
    }
    if (path.endsWith('/global-vehicles/GVO:RUN_20260724_151402:943BD1FE7C62')) {
      return jsonResponse(globalVehicleDetailFixture)
    }
    if (path.endsWith('/global-vehicles/GVO:RUN_20260724_151402:943BD1FE7C62/tracks')) {
      return jsonResponse(globalVehicleMembersFixture)
    }
    if (path.endsWith('/cross-camera-matches')) {
      return jsonResponse(matchesFixture)
    }
    if (path.endsWith('/search/vehicles')) {
      const params = url.searchParams
      if (params.get('vehicle_class') === 'PLANE') {
        return jsonResponse({ error: { code: 'VALIDATION_ERROR', message: 'Request validation failed.', details: null } }, 422)
      }
      if (params.get('colour') === 'RED') {
        return jsonResponse({
          ...vehicleSearchFixture,
          pagination: { ...vehicleSearchFixture.pagination, returned: 0, total: 0, has_more: false },
          results: [],
        })
      }
      const limit = Number(params.get('limit') || String(vehicleSearchFixture.pagination.limit))
      const offset = Number(params.get('offset') || '0')
      const pagedResults = vehicleSearchFixture.results.slice(offset, offset + limit)
      return jsonResponse({
        ...vehicleSearchFixture,
        pagination: {
          limit,
          offset,
          returned: pagedResults.length,
          total: vehicleSearchFixture.results.length,
          has_more: offset + pagedResults.length < vehicleSearchFixture.results.length,
        },
        results: pagedResults,
      })
    }
    if (path.endsWith('/search/natural-language')) {
      const body = init?.body ? JSON.parse(String(init.body)) : {}

      if (body.query === 'Find vehicles in CAM_999.') {
        return jsonResponse({
          original_query: body.query,
          parser: {
            provider: 'fallback',
            model: null,
            fallback_used: true,
          },
          clarification_required: true,
          clarification_message: 'Camera CAM_999 is not available for the selected run.',
          interpreted_filters: {
            run_code: body.run_code,
            result_scope: body.result_scope ?? 'ALL',
            camera_codes: ['CAM_999'],
            clarification_required: true,
            clarification_message: 'Camera CAM_999 is not available for the selected run.',
          },
          pagination: {
            limit: body.limit ?? 25,
            offset: body.offset ?? 0,
            returned: 0,
            total: 0,
            has_more: false,
          },
          results: [],
        })
      }

      if (body.query === 'Show red cars.') {
        return jsonResponse({
          ...naturalLanguageVehicleSearchFixture,
          original_query: body.query,
          parser: {
            provider: 'fallback',
            model: null,
            fallback_used: true,
          },
          interpreted_filters: {
            ...naturalLanguageVehicleSearchFixture.interpreted_filters,
            result_scope: body.result_scope ?? 'ALL',
            colour: 'RED',
            plate: undefined,
            plate_match_type: undefined,
            camera_codes: [],
            multi_camera_only: false,
          },
          pagination: {
            limit: body.limit ?? 25,
            offset: body.offset ?? 0,
            returned: 0,
            total: 0,
            has_more: false,
          },
          results: [],
        })
      }

      if (body.query === 'Trigger provider failure.') {
        return jsonResponse({
          error: {
            code: 'NATURAL_LANGUAGE_SEARCH_FAILED',
            message: 'Natural-language search could not be completed.',
            details: null,
          },
        }, 503)
      }

      return jsonResponse({
        ...naturalLanguageVehicleSearchFixture,
        original_query: body.query || naturalLanguageVehicleSearchFixture.original_query,
        interpreted_filters: {
          ...naturalLanguageVehicleSearchFixture.interpreted_filters,
          run_code: body.run_code ?? naturalLanguageVehicleSearchFixture.interpreted_filters.run_code,
          result_scope: body.result_scope ?? naturalLanguageVehicleSearchFixture.interpreted_filters.result_scope,
          limit: body.limit ?? naturalLanguageVehicleSearchFixture.pagination.limit,
          offset: body.offset ?? naturalLanguageVehicleSearchFixture.pagination.offset,
        },
        pagination: {
          ...naturalLanguageVehicleSearchFixture.pagination,
          limit: body.limit ?? naturalLanguageVehicleSearchFixture.pagination.limit,
          offset: body.offset ?? naturalLanguageVehicleSearchFixture.pagination.offset,
        },
      })
    }
    if (path.endsWith('/media/media-1')) {
      return jsonResponse({
        media_id: 'media-1',
        availability: 'LOCAL_FILE',
        content_url: '/api/v1/media/media-1/content',
        media_type: 'BEST_VEHICLE_CROP',
      })
    }
    if (path.endsWith('/media/media-1/url')) {
      return jsonResponse({
        media_id: 'media-1',
        availability: 'SIGNED_URL',
        url: 'https://signed.example/media-1.jpg',
        expires_in: 300,
      })
    }
    if (path.endsWith('/media/media-2/url')) {
      return jsonResponse({
        media_id: 'media-2',
        availability: 'SIGNED_URL',
        url: 'https://signed.example/media-2.jpg',
        expires_in: 300,
      })
    }

    return jsonResponse(
      {
        error: {
          code: 'NOT_FOUND',
          message: 'Not found.',
          details: null,
        },
      },
      404,
    )
  })
}

Object.defineProperty(globalThis, 'fetch', {
  value: buildMockFetch(),
  writable: true,
})

afterEach(() => {
  globalThis.fetch = buildMockFetch()
})
