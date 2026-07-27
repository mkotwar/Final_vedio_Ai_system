import type { TrackListItem } from '../../types/track'
import VehicleIdentityCard from '../vehicle/VehicleIdentityCard'

interface TrackSummaryCardProps {
  track: TrackListItem
  detailHref?: string
  compact?: boolean
}

export default function TrackSummaryCard({ track, detailHref, compact = true }: TrackSummaryCardProps) {
  return (
    <VehicleIdentityCard
      compact={compact}
      eyebrow={track.camera_code || 'Camera'}
      title="Local track"
      identifier={track.track_uuid}
      status={track.lifecycle_state}
      vehicleClass={track.vehicle_class}
      colour={track.primary_colour}
      plateResult={track.plate_result}
      plate={track.canonical_plate}
      plateStatus={track.plate_status}
      plateConfidence={track.plate_confidence}
      cameraCodes={track.camera_code ? [track.camera_code] : []}
      firstSeenAt={track.first_seen_at}
      lastSeenAt={track.last_seen_at}
      confidence={track.best_detection_confidence}
      vehicleMedia={track.primary_vehicle_media || track.primary_media}
      plateMedia={track.primary_plate_media}
      memberCount={track.observation_count}
      detailHref={detailHref}
    />
  )
}
