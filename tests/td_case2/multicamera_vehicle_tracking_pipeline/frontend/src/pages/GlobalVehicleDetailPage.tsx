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
        <div className="metric-grid">
          <Metric label="Canonical plate" value={detail.global_vehicle.canonical_plate} />
          <Metric label="Canonical class" value={detail.global_vehicle.canonical_vehicle_class} />
          <Metric label="Canonical colour" value={detail.global_vehicle.canonical_colour} />
          <Metric label="Confidence" value={<ConfidenceBadge value={detail.global_vehicle.confidence} />} />
          <Metric label="Camera count" value={detail.global_vehicle.camera_count} />
          <Metric label="Track count" value={detail.global_vehicle.track_count} />
          <Metric label="Creation method" value={detail.global_vehicle.creation_method} />
        </div>
      </section>

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
        <div className="stack-list">
          {membersQuery.data.map((member) => (
            <div key={member.track_uuid || member.vehicle_track_id || 'member'} className="list-row list-row--static">
              <div>
                <strong>{member.camera_code || 'Unknown camera'}</strong>
                <p>{member.track_uuid || 'Unknown track'}</p>
              </div>
              <div className="list-row__meta">
                <StatusBadge value={member.association_status} />
                <ConfidenceBadge value={member.association_score} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Evidence references</p>
            <h3>Evidence media</h3>
          </div>
        </div>
        <div className="card-grid">
          {detail.evidence.map((item) => (
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
