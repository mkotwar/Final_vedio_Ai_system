import { describe, expect, it } from 'vitest'
import { ANNOTATED_FULL_FRAME_MEDIA_TYPES, FULL_FRAME_MEDIA_TYPES, groupTrackMedia, pickMediaPair, PLATE_MEDIA_TYPES, VEHICLE_MEDIA_TYPES } from './mediaGroups'
import type { MediaReference } from '../../types/media'

const vehicleCrop: MediaReference = {
  media_id: 'vehicle-1',
  media_type: 'BEST_VEHICLE_CROP',
  availability: 'LOCAL_FILE',
  content_url: '/api/v1/media/vehicle-1/content',
  track_uuid: 'track-1',
  selection_rank: 1,
  is_primary: true,
}

const plateCrop: MediaReference = {
  media_id: 'plate-1',
  media_type: 'PLATE_CROP',
  availability: 'LOCAL_FILE',
  content_url: '/api/v1/media/plate-1/content',
  track_uuid: 'track-1',
  selection_rank: 1,
  is_primary: true,
}

const fullFrame: MediaReference = {
  media_id: 'full-frame-1',
  media_type: 'FULL_FRAME',
  availability: 'LOCAL_FILE',
  content_url: '/api/v1/media/full-frame-1/content',
  track_uuid: 'track-1',
  frame_number: 12,
  selection_rank: 5,
}

const annotatedFrame: MediaReference = {
  media_id: 'annotated-frame-1',
  media_type: 'ANNOTATED_FULL_FRAME',
  availability: 'LOCAL_FILE',
  content_url: '/api/v1/media/annotated-frame-1/content',
  track_uuid: 'track-1',
  frame_number: 12,
  selection_rank: 5,
}

describe('mediaGroups', () => {
  it('keeps explicit vehicle and plate media categories separate', () => {
    expect(VEHICLE_MEDIA_TYPES.has('PLATE_CROP')).toBe(false)
    expect(PLATE_MEDIA_TYPES.has('BEST_VEHICLE_CROP')).toBe(false)
    expect(FULL_FRAME_MEDIA_TYPES.has('FULL_FRAME')).toBe(true)
    expect(ANNOTATED_FULL_FRAME_MEDIA_TYPES.has('ANNOTATED_FULL_FRAME')).toBe(true)
  })

  it('never promotes a plate crop into the vehicle slot', () => {
    const pair = pickMediaPair([plateCrop], [plateCrop])
    expect(pair.vehicleMedia).toBeNull()
    expect(pair.plateMedia?.media_id).toBe('plate-1')
  })

  it('groups only track-scoped media and keeps both previews in one bundle', () => {
    const grouped = groupTrackMedia(
      [
        vehicleCrop,
        plateCrop,
        fullFrame,
        annotatedFrame,
        {
          media_id: 'other-track',
          media_type: 'BEST_VEHICLE_CROP',
          track_uuid: 'track-2',
        },
      ],
      'track-1',
    )

    expect(grouped.vehicleMedia?.media_id).toBe('vehicle-1')
    expect(grouped.plateMedia?.media_id).toBe('plate-1')
    expect(grouped.fullFrameMedia?.media_id).toBe('full-frame-1')
    expect(grouped.annotatedFrameMedia?.media_id).toBe('annotated-frame-1')
    expect(grouped.otherMedia).toHaveLength(0)
  })
})
