import { http, HttpResponse } from 'msw'
import { healthFixture, runsFixture, runDetailFixture, camerasFixture, tracksFixture, trackDetailFixture, observationsFixture, mediaFixture, globalVehiclesFixture, globalVehicleDetailFixture, globalVehicleMembersFixture, matchesFixture } from './fixtures'
import { getApiBaseUrl } from '../api/client'

const base = getApiBaseUrl()

export const handlers = [
  http.get(`${base}/health`, () => HttpResponse.json(healthFixture)),
  http.get(`${base}/runs`, () => HttpResponse.json(runsFixture)),
  http.get(`${base}/runs/:runCode`, ({ params }) =>
    params.runCode === 'RUN_20260724_151402'
      ? HttpResponse.json(runDetailFixture)
      : HttpResponse.json({ error: { code: 'RUN_NOT_FOUND', message: 'Run was not found.', details: null } }, { status: 404 }),
  ),
  http.get(`${base}/runs/:runCode/cameras`, () => HttpResponse.json(camerasFixture)),
  http.get(`${base}/runs/:runCode/tracks`, () => HttpResponse.json(tracksFixture)),
  http.get(`${base}/tracks/:trackUuid`, ({ params }) =>
    params.trackUuid === encodeURIComponent('RUN_20260724_151402:CAM_001:TRACK_4')
      ? HttpResponse.json(trackDetailFixture)
      : HttpResponse.json({ error: { code: 'TRACK_NOT_FOUND', message: 'Track was not found.', details: null } }, { status: 404 }),
  ),
  http.get(`${base}/tracks/:trackUuid/observations`, () => HttpResponse.json(observationsFixture)),
  http.get(`${base}/tracks/:trackUuid/media`, () => HttpResponse.json(mediaFixture)),
  http.get(`${base}/global-vehicles`, () => HttpResponse.json(globalVehiclesFixture)),
  http.get(`${base}/global-vehicles/:globalVehicleCode`, ({ params }) =>
    params.globalVehicleCode === encodeURIComponent('GVO:RUN_20260724_151402:943BD1FE7C62')
      ? HttpResponse.json(globalVehicleDetailFixture)
      : HttpResponse.json({ error: { code: 'GLOBAL_VEHICLE_NOT_FOUND', message: 'Global vehicle was not found.', details: null } }, { status: 404 }),
  ),
  http.get(`${base}/global-vehicles/:globalVehicleCode/tracks`, () => HttpResponse.json(globalVehicleMembersFixture)),
  http.get(`${base}/cross-camera-matches`, () => HttpResponse.json(matchesFixture)),
  http.get(`${base}/media/:mediaId`, () =>
    HttpResponse.json({
      media_id: 'media-1',
      availability: 'LOCAL_FILE',
      content_url: '/api/v1/media/media-1/content',
      media_type: 'BEST_VEHICLE_CROP',
    }),
  ),
]
