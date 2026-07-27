import type { GlobalVehicleListItem } from '../../types/globalVehicle'
import VehicleIdentityCard from '../vehicle/VehicleIdentityCard'

interface GlobalVehicleCardProps {
  vehicle: GlobalVehicleListItem
  detailHref?: string
}

export default function GlobalVehicleCard({ vehicle, detailHref }: GlobalVehicleCardProps) {
  return (
    <VehicleIdentityCard
      compact
      eyebrow={vehicle.run_code || 'Run'}
      title="Global vehicle"
      identifier={vehicle.global_vehicle_code}
      subtitle={vehicle.creation_method || null}
      status={vehicle.status}
      vehicleClass={vehicle.canonical_vehicle_class}
      colour={vehicle.canonical_colour}
      plateResult={vehicle.plate_result}
      plate={vehicle.canonical_plate}
      plateStatus={vehicle.plate_result?.status || null}
      cameraCodes={vehicle.camera_count ? [`${vehicle.camera_count} camera${vehicle.camera_count > 1 ? 's' : ''}`] : []}
      firstSeenAt={vehicle.first_seen_at}
      lastSeenAt={vehicle.last_seen_at}
      confidence={vehicle.confidence}
      vehicleMedia={vehicle.primary_vehicle_media || vehicle.primary_evidence_reference}
      plateMedia={vehicle.primary_plate_media}
      memberCount={vehicle.track_count}
      detailHref={detailHref}
    />
  )
}
