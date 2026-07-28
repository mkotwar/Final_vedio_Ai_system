import type { MediaReference } from '../../types/media'

export interface VehicleMediaGroup {
  vehicleMedia: MediaReference | null
  plateMedia: MediaReference | null
  fullFrameMedia: MediaReference | null
  annotatedFrameMedia: MediaReference | null
  additionalVehicleMedia: MediaReference[]
  additionalPlateMedia: MediaReference[]
  additionalFullFrameMedia: MediaReference[]
  additionalAnnotatedFrameMedia: MediaReference[]
  otherMedia: MediaReference[]
}

export const VEHICLE_MEDIA_TYPES = new Set([
  'BEST_VEHICLE_CROP',
  'BEST_OVERALL',
  'VEHICLE_CROP',
  'TRACK_CROP',
])

export const PLATE_MEDIA_TYPES = new Set([
  'PLATE_CROP',
  'NUMBER_PLATE_CROP',
  'ANPR_CROP',
  'OCR_CROP',
])

export const FULL_FRAME_MEDIA_TYPES = new Set(['FULL_FRAME'])
export const ANNOTATED_FULL_FRAME_MEDIA_TYPES = new Set(['ANNOTATED_FULL_FRAME'])

const VEHICLE_MEDIA_PRIORITY: Record<string, number> = {
  BEST_VEHICLE_CROP: 0,
  BEST_OVERALL: 1,
  VEHICLE_CROP: 2,
  TRACK_CROP: 3,
}

const PLATE_MEDIA_PRIORITY: Record<string, number> = {
  PLATE_CROP: 0,
  NUMBER_PLATE_CROP: 1,
  ANPR_CROP: 2,
  OCR_CROP: 3,
}

export function groupTrackMedia(media: MediaReference[], trackUuid?: string | null): VehicleMediaGroup {
  const scoped = trackUuid ? media.filter((item) => !item.track_uuid || item.track_uuid === trackUuid) : media
  const vehicleCandidates = scoped.filter((item) => isVehicleMediaType(item.media_type))
  const plateCandidates = scoped.filter((item) => isPlateMediaType(item.media_type))
  const fullFrameCandidates = scoped.filter((item) => isFullFrameMediaType(item.media_type))
  const annotatedFrameCandidates = scoped.filter((item) => isAnnotatedFullFrameMediaType(item.media_type))
  const { vehicleMedia, plateMedia } = pickMediaPair(vehicleCandidates, plateCandidates)
  const fullFrameMedia = pickPrimaryMedia(fullFrameCandidates, getFullFrameMediaPriority)
  const annotatedFrameMedia = pickPrimaryMedia(annotatedFrameCandidates, getAnnotatedFrameMediaPriority)

  return {
    vehicleMedia,
    plateMedia,
    fullFrameMedia,
    annotatedFrameMedia,
    additionalVehicleMedia: vehicleCandidates.filter((item) => item.media_id !== vehicleMedia?.media_id),
    additionalPlateMedia: plateCandidates.filter((item) => item.media_id !== plateMedia?.media_id),
    additionalFullFrameMedia: fullFrameCandidates.filter((item) => item.media_id !== fullFrameMedia?.media_id),
    additionalAnnotatedFrameMedia: annotatedFrameCandidates.filter((item) => item.media_id !== annotatedFrameMedia?.media_id),
    otherMedia: scoped.filter(
      (item) =>
        item.media_id !== vehicleMedia?.media_id &&
        item.media_id !== plateMedia?.media_id &&
        item.media_id !== fullFrameMedia?.media_id &&
        item.media_id !== annotatedFrameMedia?.media_id &&
        !vehicleCandidates.some((candidate) => candidate.media_id === item.media_id) &&
        !plateCandidates.some((candidate) => candidate.media_id === item.media_id) &&
        !fullFrameCandidates.some((candidate) => candidate.media_id === item.media_id) &&
        !annotatedFrameCandidates.some((candidate) => candidate.media_id === item.media_id),
    ),
  }
}

export function pickMediaPair(
  vehicleCandidates: Array<MediaReference | null | undefined>,
  plateCandidates: Array<MediaReference | null | undefined>,
): { vehicleMedia: MediaReference | null; plateMedia: MediaReference | null } {
  const normalizedVehicleCandidates = vehicleCandidates.filter((item): item is MediaReference => Boolean(item))
  const vehicleMedia = pickPrimaryMedia(normalizedVehicleCandidates, getVehicleMediaPriority)
  const normalizedPlateCandidates = plateCandidates
    .filter((item): item is MediaReference => Boolean(item))
    .filter((item) => item.media_id !== vehicleMedia?.media_id)
  const plateMedia = pickPrimaryMedia(normalizedPlateCandidates, getPlateMediaPriority)
  return { vehicleMedia, plateMedia }
}

function pickPrimaryMedia(
  media: MediaReference[],
  priorityResolver: (mediaType?: string | null) => number,
): MediaReference | null {
  const candidates = media.filter((item) => priorityResolver(item.media_type) < Number.POSITIVE_INFINITY)
  if (candidates.length === 0) {
    return null
  }
  return [...candidates].sort((left, right) => {
    const leftPriority = priorityResolver(left.media_type)
    const rightPriority = priorityResolver(right.media_type)
    if (leftPriority !== rightPriority) {
      return leftPriority - rightPriority
    }
    if (Boolean(left.is_primary) !== Boolean(right.is_primary)) {
      return left.is_primary ? -1 : 1
    }
    return (left.selection_rank ?? Number.MAX_SAFE_INTEGER) - (right.selection_rank ?? Number.MAX_SAFE_INTEGER)
  })[0]
}

function getVehicleMediaPriority(mediaType?: string | null) {
  return VEHICLE_MEDIA_PRIORITY[String(mediaType || '').toUpperCase()] ?? Number.POSITIVE_INFINITY
}

function getPlateMediaPriority(mediaType?: string | null) {
  return PLATE_MEDIA_PRIORITY[String(mediaType || '').toUpperCase()] ?? Number.POSITIVE_INFINITY
}

function getFullFrameMediaPriority(mediaType?: string | null) {
  return FULL_FRAME_MEDIA_TYPES.has(String(mediaType || '').toUpperCase()) ? 0 : Number.POSITIVE_INFINITY
}

function getAnnotatedFrameMediaPriority(mediaType?: string | null) {
  return ANNOTATED_FULL_FRAME_MEDIA_TYPES.has(String(mediaType || '').toUpperCase()) ? 0 : Number.POSITIVE_INFINITY
}

export function isVehicleMediaType(mediaType?: string | null) {
  return VEHICLE_MEDIA_TYPES.has(String(mediaType || '').toUpperCase())
}

export function isPlateMediaType(mediaType?: string | null) {
  return PLATE_MEDIA_TYPES.has(String(mediaType || '').toUpperCase())
}

export function isFullFrameMediaType(mediaType?: string | null) {
  return FULL_FRAME_MEDIA_TYPES.has(String(mediaType || '').toUpperCase())
}

export function isAnnotatedFullFrameMediaType(mediaType?: string | null) {
  return ANNOTATED_FULL_FRAME_MEDIA_TYPES.has(String(mediaType || '').toUpperCase())
}
