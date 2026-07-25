import type { ReactNode } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getTrack, listTrackMedia, listTrackObservations } from '../api/tracks'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import StatusBadge from '../components/states/StatusBadge'
import ConfidenceBadge from '../components/states/ConfidenceBadge'
import EvidenceCard from '../components/evidence/EvidenceCard'
import Pagination from '../components/tables/Pagination'
import DataTable from '../components/tables/DataTable'
import type { ObservationItem } from '../types/track'

export default function TrackDetailPage() {
  const { trackUuid = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const page = Number(searchParams.get('page') || '1')

  const detailQuery = useQuery({
    queryKey: ['track-detail', trackUuid],
    queryFn: () => getTrack(trackUuid),
  })
  const observationsQuery = useQuery({
    queryKey: ['track-observations', trackUuid, page],
    queryFn: () => listTrackObservations(trackUuid, { page, page_size: 10, sort_order: 'asc' }),
  })
  const mediaQuery = useQuery({
    queryKey: ['track-media', trackUuid],
    queryFn: () => listTrackMedia(trackUuid),
  })

  if (detailQuery.isPending || observationsQuery.isPending || mediaQuery.isPending) {
    return <LoadingState label="Loading track detail..." />
  }

  if (detailQuery.isError || observationsQuery.isError || mediaQuery.isError || !detailQuery.data) {
    return (
      <ErrorState
        title="Track detail unavailable"
        message="The requested track could not be loaded."
        onRetry={() => {
          void detailQuery.refetch()
          void observationsQuery.refetch()
          void mediaQuery.refetch()
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
            <p className="panel__eyebrow">{detail.camera.camera_code || 'Camera'}</p>
            <h3>{detail.track.track_uuid}</h3>
          </div>
          <StatusBadge value={detail.track.lifecycle_state} />
        </div>
        <div className="metric-grid">
          <Metric label="Class" value={detail.track.vehicle_class} />
          <Metric label="Primary colour" value={detail.colour.primary_colour} />
          <Metric label="Plate" value={detail.plate.canonical_plate} />
          <Metric label="Plate status" value={<StatusBadge value={detail.plate.plate_status} />} />
          <Metric label="Best confidence" value={<ConfidenceBadge value={detail.track.best_detection_confidence} />} />
          <Metric label="Global membership" value={detail.global_membership?.global_vehicle_code ? <Link className="table-link" to={`/global-vehicles/${encodeURIComponent(detail.global_membership.global_vehicle_code)}`}>{detail.global_membership.global_vehicle_code}</Link> : 'Not linked'} />
          <Metric label="Observations" value={detail.observation_summary.count} />
          <Metric label="Timing" value={`${detail.track.first_seen_at || 'N/A'} → ${detail.track.last_seen_at || 'N/A'}`} />
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Cross-camera matches</p>
            <h3>Match context</h3>
          </div>
        </div>
        <div className="stack-list">
          {detail.cross_camera_matches.length === 0 ? (
            <p className="table-subtext">No cross-camera matches were returned for this track.</p>
          ) : (
            detail.cross_camera_matches.map((match) => (
              <div key={match.id || `${match.source_track_id}-${match.candidate_track_id}`} className="list-row list-row--static">
                <div>
                  <strong>{match.decision || 'UNKNOWN'}</strong>
                  <p>{match.created_global_vehicle_id || 'No linked global object'}</p>
                </div>
                <ConfidenceBadge value={match.overall_score} />
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Observation summary</p>
            <h3>Frame-level observations</h3>
          </div>
        </div>
        <DataTable
          columns={[
            { key: 'frame', header: 'Frame', render: (row: ObservationItem) => row.frame_number },
            { key: 'time', header: 'Timestamp', render: (row: ObservationItem) => row.timestamp || 'N/A' },
            { key: 'video', header: 'Video time', render: (row: ObservationItem) => row.video_time_seconds ?? 'N/A' },
            { key: 'bbox', header: 'Bounding box', render: (row: ObservationItem) => `${row.bbox.x1}, ${row.bbox.y1}, ${row.bbox.x2}, ${row.bbox.y2}` },
            { key: 'det', header: 'Detector', render: (row: ObservationItem) => <ConfidenceBadge value={row.detection_confidence} /> },
            { key: 'trk', header: 'Tracker', render: (row: ObservationItem) => <ConfidenceBadge value={row.tracker_confidence} /> },
            { key: 'key', header: 'Key observation', render: (row: ObservationItem) => row.is_key_observation ? 'Yes' : 'No' },
          ]}
          rows={observationsQuery.data.items}
          getRowKey={(row) => `${row.frame_number}-${row.timestamp || 'observation'}`}
        />
        <Pagination
          page={observationsQuery.data.page}
          pageSize={observationsQuery.data.page_size}
          total={observationsQuery.data.total}
          hasNext={observationsQuery.data.has_next}
          onPageChange={(nextPage) => {
            const next = new URLSearchParams(searchParams)
            next.set('page', String(nextPage))
            setSearchParams(next)
          }}
        />
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Evidence references</p>
            <h3>Media metadata</h3>
          </div>
        </div>
        <div className="card-grid">
          {mediaQuery.data.map((media) => (
            <EvidenceCard key={media.media_id || `${media.media_type || 'media'}-${media.frame_number || 'na'}`} media={media} />
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Processing errors</p>
            <h3>Track-scoped issues</h3>
          </div>
        </div>
        {detail.errors.length === 0 ? (
          <p className="table-subtext">No processing errors were attached to this track.</p>
        ) : (
          <div className="stack-list">
            {detail.errors.map((error) => (
              <div key={error.id || `${error.stage_name}-${error.created_at}`} className="list-row list-row--static">
                <div>
                  <strong>{error.error_code || error.stage_name || 'Processing error'}</strong>
                  <p>{error.message || 'No message available'}</p>
                </div>
                <StatusBadge value={error.severity} />
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function Metric({
  label,
  value,
}: {
  label: string
  value: ReactNode
}) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
