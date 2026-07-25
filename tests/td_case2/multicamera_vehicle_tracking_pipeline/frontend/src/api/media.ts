import { apiGet } from './client'
import type { MediaDeliveryResponse } from '../types/media'

export function getMediaReference(mediaId: string) {
  return apiGet<MediaDeliveryResponse>(`/media/${encodeURIComponent(mediaId)}`)
}
