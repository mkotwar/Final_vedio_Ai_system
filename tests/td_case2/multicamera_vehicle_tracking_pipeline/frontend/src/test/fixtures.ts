import type { HealthResponse, PaginatedResponse } from '../types/common'
import type { GlobalVehicleDetailResponse, GlobalVehicleListItem, GlobalVehicleMember } from '../types/globalVehicle'
import type { MatchListItem } from '../types/match'
import type { CameraListItem, RunDetailResponse, RunListItem } from '../types/run'
import type { MediaReference } from '../types/media'
import type { ObservationItem, TrackDetailResponse, TrackListItem } from '../types/track'

export const healthFixture: HealthResponse = {
  status: 'ok',
  service: 'multicamera-vehicle-api',
  database: 'reachable',
  schema: 'analytics',
}

export const runsFixture: PaginatedResponse<RunListItem> = {
  items: [
    {
      id: 'run-1',
      run_code: 'RUN_20260724_151402',
      status: 'COMPLETED',
      started_at: '2026-07-24T15:14:02+05:30',
      completed_at: '2026-07-24T15:19:02+05:30',
      created_at: '2026-07-24T15:14:01+05:30',
      camera_count: 2,
      track_count: 8,
      global_vehicle_count: 7,
      processing_error_count: 0,
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}

export const runDetailFixture: RunDetailResponse = {
  run_code: 'RUN_20260724_151402',
  status: 'COMPLETED',
  camera_summary: {
    configured_camera_count: 2,
    active_camera_count: 2,
    camera_run_count: 2,
    completed_camera_runs: 2,
  },
  track_summary: {
    track_count: 8,
    total_track_observations: 120,
  },
  enrichment_summary: {
    tracks_with_colour: 8,
    tracks_with_plate_summary: 8,
    tracks_with_media: 8,
  },
  global_object_summary: {
    global_vehicle_count: 7,
  },
  processing_error_summary: {
    processing_error_count: 0,
  },
}

export const camerasFixture: PaginatedResponse<CameraListItem> = {
  items: [
    {
      id: 'cam-1',
      camera_code: 'CAM_001',
      camera_name: 'North Gate',
      location: 'North',
      camera_run_status: 'COMPLETED',
      frames_read: 120,
      frames_processed: 120,
      detection_count: 30,
      completed_track_count: 4,
      discarded_track_count: 1,
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}

export const mediaFixture: MediaReference[] = [
  {
    media_id: 'media-1',
    media_type: 'BEST_VEHICLE_CROP',
    storage_provider: 'LOCAL',
    storage_uri: 'debug_runs/reference_only/car_1.jpg',
    frame_number: 12,
    quality_score: 0.92,
    selection_rank: 1,
  },
]

export const tracksFixture: PaginatedResponse<TrackListItem> = {
  items: [
    {
      track_uuid: 'RUN_20260724_151402:CAM_001:TRACK_4',
      camera_code: 'CAM_001',
      local_track_id: 4,
      vehicle_class: 'CAR',
      lifecycle_state: 'COMPLETED',
      first_seen_at: '2026-07-24T15:14:30+05:30',
      last_seen_at: '2026-07-24T15:14:45+05:30',
      observation_count: 10,
      best_detection_confidence: 0.95,
      primary_colour: 'GREY',
      canonical_plate: 'DL8CBF6268',
      plate_status: 'VERIFIED',
      primary_media: mediaFixture[0],
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}

export const observationsFixture: PaginatedResponse<ObservationItem> = {
  items: [
    {
      frame_number: 12,
      timestamp: '2026-07-24T15:14:31+05:30',
      video_time_seconds: 12.4,
      bbox: { x1: 1, y1: 2, x2: 3, y2: 4 },
      detection_confidence: 0.92,
      tracker_confidence: 0.89,
      is_key_observation: true,
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}

export const trackDetailFixture: TrackDetailResponse = {
  track: tracksFixture.items[0],
  camera: {
    camera_code: 'CAM_001',
    camera_name: 'North Gate',
    location: 'North',
  },
  colour: {
    primary_colour: 'GREY',
    colour_confidence: 0.95,
  },
  plate: {
    canonical_plate: 'DL8CBF6268',
    plate_status: 'VERIFIED',
    plate_confidence: 0.98,
  },
  media: mediaFixture,
  observation_summary: {
    count: 10,
    first_frame: 12,
    last_frame: 20,
    key_observation_count: 2,
  },
  global_membership: {
    global_vehicle_code: 'GVO:RUN_20260724_151402:943BD1FE7C62',
    association_status: 'CONFIRMED',
    association_score: 0.95,
  },
  cross_camera_matches: [
    {
      id: 'match-1',
      decision: 'CONFIRMED',
      overall_score: 0.95,
      created_global_vehicle_id: 'global-1',
    },
  ],
  errors: [],
}

export const globalVehiclesFixture: PaginatedResponse<GlobalVehicleListItem> = {
  items: [
    {
      global_vehicle_code: 'GVO:RUN_20260724_151402:943BD1FE7C62',
      run_code: 'RUN_20260724_151402',
      status: 'CONFIRMED',
      canonical_plate: 'DL8CBF6268',
      canonical_colour: 'GREY',
      canonical_vehicle_class: 'CAR',
      confidence: 0.95,
      camera_count: 2,
      track_count: 2,
      creation_method: 'CROSS_CAMERA_MATCH',
      first_seen_at: '2026-07-24T15:14:30+05:30',
      last_seen_at: '2026-07-24T15:14:50+05:30',
      primary_evidence_reference: mediaFixture[0],
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}

export const globalVehicleMembersFixture: GlobalVehicleMember[] = [
  {
    track_uuid: 'RUN_20260724_151402:CAM_001:TRACK_4',
    camera_code: 'CAM_001',
    association_status: 'CONFIRMED',
    association_score: 0.95,
  },
  {
    track_uuid: 'RUN_20260724_151402:CAM_002:TRACK_4',
    camera_code: 'CAM_002',
    association_status: 'CONFIRMED',
    association_score: 0.95,
  },
]

export const globalVehicleDetailFixture: GlobalVehicleDetailResponse = {
  global_vehicle: {
    global_vehicle_code: 'GVO:RUN_20260724_151402:943BD1FE7C62',
    run_code: 'RUN_20260724_151402',
    status: 'CONFIRMED',
    canonical_plate: 'DL8CBF6268',
    canonical_colour: 'GREY',
    canonical_vehicle_class: 'CAR',
    confidence: 0.95,
    camera_count: 2,
    track_count: 2,
    creation_method: 'CROSS_CAMERA_MATCH',
  },
  members: globalVehicleMembersFixture,
  camera_sequence: globalVehicleMembersFixture.map((member) => ({
    camera_code: member.camera_code,
    track_uuid: member.track_uuid,
  })),
  confirmed_matches: [{ id: 'match-1' }],
  possible_matches: [],
  evidence: mediaFixture,
}

export const matchesFixture: PaginatedResponse<MatchListItem> = {
  items: [
    {
      id: 'match-1',
      source_track_uuid: 'RUN_20260724_151402:CAM_001:TRACK_4',
      source_camera_code: 'CAM_001',
      candidate_track_uuid: 'RUN_20260724_151402:CAM_002:TRACK_4',
      candidate_camera_code: 'CAM_002',
      decision: 'CONFIRMED',
      overall_score: 0.95,
      plate_score: 1,
      class_score: 1,
      colour_score: 0.95,
      route_score: 0.8,
      time_score: 0.9,
      visual_score: 0.75,
      decision_reasons: ['plate match', 'time gap ok'],
      linked_global_vehicle_code: 'GVO:RUN_20260724_151402:943BD1FE7C62',
      rule_version: 'global_match_v1',
    },
  ],
  page: 1,
  page_size: 10,
  total: 1,
  has_next: false,
}
