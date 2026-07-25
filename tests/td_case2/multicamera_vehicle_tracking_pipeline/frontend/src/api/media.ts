import { apiGet } from './client'
import type { MediaDeliveryResponse, MediaSignedUrlResponse } from '../types/media'

export function getMediaReference(mediaId: string) {
  return apiGet<MediaDeliveryResponse>(`/media/${encodeURIComponent(mediaId)}`)
}

export function getMediaSignedUrl(mediaId: string) {
  return apiGet<MediaSignedUrlResponse>(`/media/${encodeURIComponent(mediaId)}/url`)
}
