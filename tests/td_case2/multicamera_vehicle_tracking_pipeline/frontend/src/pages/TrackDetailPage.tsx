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
import VehicleIdentityCard from '../components/vehicle/VehicleIdentityCard'
import { groupTrackMedia } from '../components/vehicle/mediaGroups'
import type { ObservationItem, TrackDetailResponse } from '../types/track'

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
  const groupedMedia = groupTrackMedia(mediaQuery.data, detail.track.track_uuid)
  const additionalMedia = [
    ...groupedMedia.additionalVehicleMedia,
    ...groupedMedia.additionalPlateMedia,
    ...groupedMedia.additionalFullFrameMedia,
    ...groupedMedia.additionalAnnotatedFrameMedia,
    ...groupedMedia.otherMedia,
  ]

  return (
    <div className="page-stack">
      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Vehicle Identity</p>
            <h3>Vehicle Identity</h3>
          </div>
        </div>
        <VehicleIdentityCard
          eyebrow={detail.camera.camera_code || 'Camera'}
          title="Track detail"
          identifier={detail.track.track_uuid}
          subtitle={detail.camera.camera_name || detail.camera.location || null}
          status={detail.track.lifecycle_state}
          vehicleClass={detail.track.vehicle_class}
          colour={detail.colour.primary_colour}
          plateResult={detail.plate.plate_result || detail.track.plate_result}
          plate={detail.plate.canonical_plate}
          plateStatus={detail.plate.plate_status}
          plateConfidence={detail.plate.plate_confidence}
          cameraCodes={detail.camera.camera_code ? [detail.camera.camera_code] : []}
          firstSeenAt={detail.track.first_seen_at}
          lastSeenAt={detail.track.last_seen_at}
          confidence={detail.track.best_detection_confidence}
          vehicleMedia={groupedMedia.vehicleMedia || detail.track.primary_vehicle_media || detail.track.primary_media}
          plateMedia={groupedMedia.plateMedia || detail.track.primary_plate_media}
          globalMembership={renderGlobalMembership(detail)}
          detailHref={
            detail.global_membership?.linked && detail.global_membership.global_vehicle_code
              ? `/global-vehicles/${encodeURIComponent(detail.global_membership.global_vehicle_code)}`
              : undefined
          }
          detailLabel="Open Global Vehicle"
        />
        <div className="metric-grid">
          <Metric label="Observations" value={detail.observation_summary.count} />
          <Metric label="First frame" value={detail.observation_summary.first_frame ?? 'N/A'} />
          <Metric label="Last frame" value={detail.observation_summary.last_frame ?? 'N/A'} />
          <Metric label="Key observations" value={detail.observation_summary.key_observation_count} />
          <Metric label="Stable class" value={detail.track.class_is_stable ? 'Yes' : 'No'} />
          <Metric label="Class confidence" value={detail.track.class_confidence?.toFixed(2) || 'N/A'} />
        </div>
      </section>

      {detail.class_diagnostics ? (
        <section className="panel">
          <div className="panel__header">
            <div>
              <p className="panel__eyebrow">Class diagnostics</p>
              <h3>Class stabilization</h3>
            </div>
          </div>
          <div className="metric-grid">
            <Metric label="Final class" value={detail.class_diagnostics.stable_class_name || detail.track.vehicle_class || 'N/A'} />
            <Metric label="Provisional class" value={detail.class_diagnostics.provisional_class_name || 'N/A'} />
            <Metric label="Stable" value={detail.class_diagnostics.class_is_locked ? 'Yes' : 'No'} />
            <Metric label="Winner margin" value={detail.class_diagnostics.class_winner_margin?.toFixed(2) || 'N/A'} />
          </div>
          <div className="stack-list">
            {Object.entries(detail.class_diagnostics.class_observation_counts || {}).map(([className, count]) => (
              <div key={className} className="list-row list-row--static">
                <div>
                  <strong>{className}</strong>
                  <p>Observations: {count}</p>
                </div>
                <span>Score {(detail.class_diagnostics?.class_scores?.[className] ?? 0).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

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

      {groupedMedia.fullFrameMedia || groupedMedia.annotatedFrameMedia ? (
        <section className="panel">
          <div className="panel__header">
            <div>
              <p className="panel__eyebrow">Source context</p>
              <h3>Source frames</h3>
            </div>
          </div>
          <div className="card-grid">
            {groupedMedia.fullFrameMedia ? <EvidenceCard media={groupedMedia.fullFrameMedia} /> : null}
            {groupedMedia.annotatedFrameMedia ? <EvidenceCard media={groupedMedia.annotatedFrameMedia} /> : null}
          </div>
        </section>
      ) : null}

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
            { key: 'cls', header: 'Class', render: (row: ObservationItem) => row.class_name || 'N/A' },
            { key: 'raw', header: 'Raw class', render: (row: ObservationItem) => row.raw_class_name || 'N/A' },
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
            <p className="panel__eyebrow">Additional Evidence</p>
            <h3>Additional evidence</h3>
          </div>
        </div>
        <div className="card-grid">
          {additionalMedia.map((media) => (
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

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function renderGlobalMembership(detail: TrackDetailResponse) {
  const membership = detail.global_membership

  if (!membership || !membership.linked) {
    return 'Not linked'
  }

  if (!membership.global_vehicle_code) {
    return 'Membership unavailable'
  }

  return (
    <span className="metric-card__stack">
      <span>Linked to global vehicle</span>
      <strong>
        <Link className="table-link" to={`/global-vehicles/${encodeURIComponent(membership.global_vehicle_code)}`}>
          {membership.global_vehicle_code}
        </Link>
      </strong>
    </span>
  )
}
