import type { ReactNode } from 'react'
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getRun, listRunCameras } from '../api/runs'
import { listTracks } from '../api/tracks'
import { listGlobalVehicles } from '../api/globalVehicles'
import { listMatches } from '../api/matches'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import StatusBadge from '../components/states/StatusBadge'
import DataTable from '../components/tables/DataTable'
import type { CameraListItem } from '../types/run'
import type { MatchListItem } from '../types/match'
import TrackSummaryCard from '../components/cards/TrackSummaryCard'
import GlobalVehicleCard from '../components/cards/GlobalVehicleCard'

type DetailTab = 'cameras' | 'tracks' | 'global' | 'matches'

export default function RunDetailPage() {
  const { runCode = '' } = useParams()
  const [activeTab, setActiveTab] = useState<DetailTab>('cameras')

  const runQuery = useQuery({
    queryKey: ['run-detail', runCode],
    queryFn: () => getRun(runCode),
  })
  const camerasQuery = useQuery({
    queryKey: ['run-detail', runCode, 'cameras'],
    queryFn: () => listRunCameras(runCode, { page: 1, page_size: 10 }),
  })
  const tracksQuery = useQuery({
    queryKey: ['run-detail', runCode, 'tracks'],
    queryFn: () => listTracks(runCode, { page: 1, page_size: 10 }),
  })
  const globalVehiclesQuery = useQuery({
    queryKey: ['run-detail', runCode, 'global-vehicles'],
    queryFn: () => listGlobalVehicles({ run_code: runCode, page: 1, page_size: 10 }),
  })
  const matchesQuery = useQuery({
    queryKey: ['run-detail', runCode, 'matches'],
    queryFn: () => listMatches({ run_code: runCode, page: 1, page_size: 10 }),
  })

  const error = runQuery.isError || camerasQuery.isError || tracksQuery.isError || globalVehiclesQuery.isError || matchesQuery.isError

  if (runQuery.isPending) {
    return <LoadingState label="Loading run details..." />
  }

  if (error || !runQuery.data) {
    return (
      <ErrorState
        title="Run detail unavailable"
        message="The requested processing run could not be loaded."
        onRetry={() => {
          void runQuery.refetch()
          void camerasQuery.refetch()
          void tracksQuery.refetch()
          void globalVehiclesQuery.refetch()
          void matchesQuery.refetch()
        }}
      />
    )
  }

  const run = runQuery.data
  const tabs: Array<{ id: DetailTab; label: string; count: number }> = [
    { id: 'cameras', label: 'Cameras', count: camerasQuery.data?.items.length || 0 },
    { id: 'tracks', label: 'Tracks', count: tracksQuery.data?.items.length || 0 },
    { id: 'global', label: 'Global vehicles', count: globalVehiclesQuery.data?.items.length || 0 },
    { id: 'matches', label: 'Matches', count: matchesQuery.data?.items.length || 0 },
  ]

  return (
    <div className="page-stack">
      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Run detail</p>
            <h3>{run.run_code}</h3>
          </div>
          <StatusBadge value={run.status} />
        </div>
        <div className="metric-grid">
          <Metric label="Configured cameras" value={run.camera_summary?.configured_camera_count} />
          <Metric label="Active cameras" value={run.camera_summary?.active_camera_count} />
          <Metric label="Local tracks" value={run.track_summary?.track_count} />
          <Metric label="Global objects" value={run.global_object_summary?.global_vehicle_count} />
          <Metric label="Colour coverage" value={run.enrichment_summary?.tracks_with_colour} />
          <Metric label="Processing errors" value={run.processing_error_summary?.processing_error_count} />
          <Metric label="Confirmed matches" value={matchesQuery.data?.items.filter((item) => item.decision === 'CONFIRMED').length} />
        </div>
      </section>

      <section className="panel">
        <div className="tab-list" role="tablist" aria-label="Run detail sections">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button${activeTab === tab.id ? ' tab-button--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              role="tab"
              type="button"
            >
              {tab.label}
              <span className="tab-button__count">{tab.count}</span>
            </button>
          ))}
        </div>

        {activeTab === 'cameras' && camerasQuery.data ? (
          <DataTable
            columns={[
              {
                key: 'camera',
                header: 'Camera',
                render: (row: CameraListItem) => (
                  <div>
                    <strong>{row.camera_code}</strong>
                    <p className="table-subtext">{row.camera_name || row.location || 'No metadata'}</p>
                  </div>
                ),
              },
              { key: 'status', header: 'Status', render: (row: CameraListItem) => <StatusBadge value={row.camera_run_status} /> },
              { key: 'frames', header: 'Frames processed', render: (row: CameraListItem) => row.frames_processed },
              { key: 'tracks', header: 'Completed tracks', render: (row: CameraListItem) => row.completed_track_count },
            ]}
            rows={camerasQuery.data.items}
            getRowKey={(row) => row.camera_code || 'camera'}
          />
        ) : null}

        {activeTab === 'tracks' && tracksQuery.data ? (
          <div className="vehicle-results-grid vehicle-results-grid--compact">
            {tracksQuery.data.items.map((track) => (
              <TrackSummaryCard
                key={track.track_uuid}
                track={track}
                detailHref={`/tracks/${encodeURIComponent(track.track_uuid)}`}
              />
            ))}
          </div>
        ) : null}

        {activeTab === 'global' && globalVehiclesQuery.data ? (
          <div className="vehicle-results-grid vehicle-results-grid--compact">
            {globalVehiclesQuery.data.items.map((vehicle) => (
              <GlobalVehicleCard
                key={vehicle.global_vehicle_code}
                vehicle={vehicle}
                detailHref={`/global-vehicles/${encodeURIComponent(vehicle.global_vehicle_code)}`}
              />
            ))}
          </div>
        ) : null}

        {activeTab === 'matches' && matchesQuery.data ? (
          <DataTable
            columns={[
              {
                key: 'source',
                header: 'Source track',
                render: (row: MatchListItem) => row.source_track_uuid || 'N/A',
              },
              {
                key: 'candidate',
                header: 'Candidate track',
                render: (row: MatchListItem) => row.candidate_track_uuid || 'N/A',
              },
              { key: 'decision', header: 'Decision', render: (row: MatchListItem) => <StatusBadge value={row.decision} /> },
              { key: 'linked', header: 'Global vehicle', render: (row: MatchListItem) => row.linked_global_vehicle_code || 'N/A' },
            ]}
            rows={matchesQuery.data.items}
            getRowKey={(row) => row.id || row.source_track_uuid || 'match'}
          />
        ) : null}
      </section>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? 'N/A'}</strong>
    </div>
  )
}
