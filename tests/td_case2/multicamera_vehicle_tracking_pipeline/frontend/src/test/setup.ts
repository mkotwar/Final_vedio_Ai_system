import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { camerasFixture, globalVehicleDetailFixture, globalVehicleMembersFixture, globalVehiclesFixture, healthFixture, matchesFixture, mediaFixture, observationsFixture, runDetailFixture, runsFixture, trackDetailFixture, tracksFixture } from './fixtures'

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
  return vi.fn(async (input: RequestInfo | URL) => {
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
