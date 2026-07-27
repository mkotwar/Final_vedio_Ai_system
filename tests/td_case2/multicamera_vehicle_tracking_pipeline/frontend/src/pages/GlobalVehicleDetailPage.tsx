import type { ReactNode } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getGlobalVehicle, listGlobalVehicleTracks } from '../api/globalVehicles'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import StatusBadge from '../components/states/StatusBadge'
import ConfidenceBadge from '../components/states/ConfidenceBadge'
import CameraSequence from '../components/cards/CameraSequence'
import EvidenceCard from '../components/evidence/EvidenceCard'
import VehicleIdentityCard from '../components/vehicle/VehicleIdentityCard'
import { groupTrackMedia, isFullFrameMediaType, isPlateMediaType, isVehicleMediaType } from '../components/vehicle/mediaGroups'

export default function GlobalVehicleDetailPage() {
  const { globalVehicleCode = '' } = useParams()

  const detailQuery = useQuery({
    queryKey: ['global-vehicle-detail', globalVehicleCode],
    queryFn: () => getGlobalVehicle(globalVehicleCode),
  })
  const membersQuery = useQuery({
    queryKey: ['global-vehicle-members', globalVehicleCode],
    queryFn: () => listGlobalVehicleTracks(globalVehicleCode),
  })

  if (detailQuery.isPending || membersQuery.isPending) {
    return <LoadingState label="Loading global vehicle detail..." />
  }

  if (detailQuery.isError || membersQuery.isError || !detailQuery.data) {
    return (
      <ErrorState
        title="Global vehicle unavailable"
        message="The requested global vehicle could not be loaded."
        onRetry={() => {
          void detailQuery.refetch()
          void membersQuery.refetch()
        }}
      />
    )
  }

  const detail = detailQuery.data
  const groupedEvidenceByTrack = new Map(
    membersQuery.data.map((member) => [member.track_uuid || '', groupTrackMedia(detail.evidence, member.track_uuid)]),
  )
  const representativeCameras = membersQuery.data.map((member) => member.camera_code).filter(Boolean) as string[]
  const representativeFullFrames = membersQuery.data
    .map((member) => groupedEvidenceByTrack.get(member.track_uuid || '')?.fullFrameMedia || null)
    .filter((item, index, list): item is NonNullable<typeof item> => Boolean(item) && list.findIndex((candidate) => candidate?.media_id === item?.media_id) === index)

  return (
    <div className="page-stack">
      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Global vehicle</p>
            <h3>{detail.global_vehicle.global_vehicle_code}</h3>
          </div>
          <StatusBadge value={detail.global_vehicle.status} />
        </div>
        <VehicleIdentityCard
          eyebrow={detail.global_vehicle.run_code || 'Run'}
          title="Global vehicle"
          identifier={detail.global_vehicle.global_vehicle_code}
          subtitle={detail.global_vehicle.creation_method || null}
          status={detail.global_vehicle.status}
          vehicleClass={detail.global_vehicle.canonical_vehicle_class}
          colour={detail.global_vehicle.canonical_colour}
          plateResult={detail.global_vehicle.plate_result}
          plate={detail.global_vehicle.canonical_plate}
          plateStatus={detail.global_vehicle.plate_result?.status || null}
          cameraCodes={representativeCameras}
          firstSeenAt={detail.global_vehicle.first_seen_at}
          lastSeenAt={detail.global_vehicle.last_seen_at}
          confidence={detail.global_vehicle.confidence}
          vehicleMedia={detail.global_vehicle.primary_vehicle_media}
          plateMedia={detail.global_vehicle.primary_plate_media}
          memberCount={detail.global_vehicle.track_count}
        />
        <div className="metric-grid">
          <Metric label="Camera count" value={detail.global_vehicle.camera_count} />
          <Metric label="Track count" value={detail.global_vehicle.track_count} />
          <Metric label="Creation method" value={detail.global_vehicle.creation_method} />
          <Metric label="Confidence" value={<ConfidenceBadge value={detail.global_vehicle.confidence} />} />
        </div>
      </section>

      {representativeFullFrames.length > 0 ? (
        <section className="panel">
          <div className="panel__header">
            <div>
              <p className="panel__eyebrow">Scene frames</p>
              <h3>Representative full frames</h3>
            </div>
          </div>
          <div className="card-grid">
            {representativeFullFrames.map((item) => (
              <EvidenceCard key={item.media_id || `${item.media_type || 'full-frame'}-${item.frame_number || 'na'}`} media={item} />
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Camera sequence</p>
            <h3>Cross-camera progression</h3>
          </div>
        </div>
        <CameraSequence steps={detail.camera_sequence} />
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Member tracks</p>
            <h3>Attached local tracks</h3>
          </div>
        </div>
        <div className="vehicle-results-grid vehicle-results-grid--compact">
          {membersQuery.data.map((member) => (
            <VehicleIdentityCard
              key={member.track_uuid || member.vehicle_track_id || 'member'}
              compact
              eyebrow={member.camera_code || 'Camera'}
              title="Member track"
              identifier={member.track_uuid || 'Unknown track'}
              subtitle={member.association_method || null}
              status={member.association_status}
              vehicleClass={member.vehicle_class}
              colour={member.primary_colour}
              plateResult={member.plate_result}
              plate={member.canonical_plate}
              plateStatus={member.plate_status}
              plateConfidence={member.plate_confidence}
              cameraCodes={member.camera_code ? [member.camera_code] : []}
              firstSeenAt={member.first_seen_at}
              lastSeenAt={member.last_seen_at}
              confidence={member.association_score ?? member.best_detection_confidence}
              vehicleMedia={member.primary_vehicle_media || groupedEvidenceByTrack.get(member.track_uuid || '')?.vehicleMedia}
              plateMedia={member.primary_plate_media || groupedEvidenceByTrack.get(member.track_uuid || '')?.plateMedia}
              detailHref={member.track_uuid ? `/tracks/${encodeURIComponent(member.track_uuid)}` : undefined}
            />
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Additional Evidence</p>
            <h3>Additional evidence</h3>
          </div>
        </div>
        <div className="card-grid">
          {detail.evidence.filter((item) => !isVehicleMediaType(item.media_type) && !isPlateMediaType(item.media_type) && !isFullFrameMediaType(item.media_type)).map((item) => (
            <EvidenceCard key={item.media_id || `${item.media_type || 'evidence'}-${item.frame_number || 'na'}`} media={item} />
          ))}
        </div>
      </section>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value || 'N/A'}</strong>
    </div>
  )
}
