import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listMatches } from '../api/matches'
import type { MatchListItem } from '../types/match'
import FilterBar from '../components/filters/FilterBar'
import DataTable from '../components/tables/DataTable'
import Pagination from '../components/tables/Pagination'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import StatusBadge from '../components/states/StatusBadge'
import ConfidenceBadge from '../components/states/ConfidenceBadge'

export default function CrossCameraMatchesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const filters = useMemo(
    () => ({
      run_code: searchParams.get('run_code') || undefined,
      decision: searchParams.get('decision') || undefined,
      minimum_score: searchParams.get('minimum_score') || undefined,
      rule_version: searchParams.get('rule_version') || undefined,
      camera_code: searchParams.get('camera_code') || undefined,
      page: Number(searchParams.get('page') || '1'),
      page_size: Number(searchParams.get('page_size') || '10'),
      sort_by: searchParams.get('sort_by') || 'updated_at',
      sort_order: searchParams.get('sort_order') || 'desc',
    }),
    [searchParams],
  )

  const matchesQuery = useQuery({
    queryKey: ['matches', filters],
    queryFn: () => listMatches(filters),
  })

  function updateFilter(name: string, value: string) {
    const next = new URLSearchParams(searchParams)
    if (value) {
      next.set(name, value)
    } else {
      next.delete(name)
    }
    if (name !== 'page') {
      next.set('page', '1')
    }
    setSearchParams(next)
  }

  if (matchesQuery.isPending) {
    return <LoadingState label="Loading cross-camera matches..." />
  }

  if (matchesQuery.isError) {
    return <ErrorState title="Matches unavailable" message="The frontend could not load cross-camera match results." onRetry={() => void matchesQuery.refetch()} />
  }

  return (
    <div className="page-stack">
      <FilterBar>
        <label>
          <span>Run code</span>
          <input value={filters.run_code || ''} onChange={(event) => updateFilter('run_code', event.target.value)} placeholder="RUN_20260724_151402" />
        </label>
        <label>
          <span>Decision</span>
          <input value={filters.decision || ''} onChange={(event) => updateFilter('decision', event.target.value)} placeholder="CONFIRMED" />
        </label>
        <label>
          <span>Minimum score</span>
          <input value={filters.minimum_score || ''} onChange={(event) => updateFilter('minimum_score', event.target.value)} placeholder="0.9" type="number" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Rule version</span>
          <input value={filters.rule_version || ''} onChange={(event) => updateFilter('rule_version', event.target.value)} placeholder="global_match_v1" />
        </label>
        <label>
          <span>Camera code</span>
          <input value={filters.camera_code || ''} onChange={(event) => updateFilter('camera_code', event.target.value)} placeholder="CAM_001" />
        </label>
        <button className="button button--secondary" onClick={() => void matchesQuery.refetch()} type="button">
          Refresh
        </button>
      </FilterBar>

      {matchesQuery.data.items.length === 0 ? (
        <EmptyState title="No matches found" description="No cross-camera matches matched the current filters." />
      ) : (
        <>
          <DataTable
            columns={[
              { key: 'sourceTrack', header: 'Source track', render: (row: MatchListItem) => row.source_track_uuid || 'N/A' },
              { key: 'sourceCamera', header: 'Source camera', render: (row: MatchListItem) => row.source_camera_code || 'N/A' },
              { key: 'candidateTrack', header: 'Candidate track', render: (row: MatchListItem) => row.candidate_track_uuid || 'N/A' },
              { key: 'candidateCamera', header: 'Candidate camera', render: (row: MatchListItem) => row.candidate_camera_code || 'N/A' },
              { key: 'decision', header: 'Decision', render: (row: MatchListItem) => <StatusBadge value={row.decision} /> },
              { key: 'overall', header: 'Overall', render: (row: MatchListItem) => <ConfidenceBadge value={row.overall_score} /> },
              { key: 'plate', header: 'Plate', render: (row: MatchListItem) => <ConfidenceBadge value={row.plate_score} /> },
              { key: 'class', header: 'Class', render: (row: MatchListItem) => <ConfidenceBadge value={row.class_score} /> },
              { key: 'colour', header: 'Colour', render: (row: MatchListItem) => <ConfidenceBadge value={row.colour_score} /> },
              { key: 'route', header: 'Route', render: (row: MatchListItem) => <ConfidenceBadge value={row.route_score} /> },
              { key: 'time', header: 'Time', render: (row: MatchListItem) => <ConfidenceBadge value={row.time_score} /> },
              { key: 'visual', header: 'Visual', render: (row: MatchListItem) => <ConfidenceBadge value={row.visual_score} /> },
              { key: 'linked', header: 'Linked global vehicle', render: (row: MatchListItem) => row.linked_global_vehicle_code || 'N/A' },
              { key: 'reasons', header: 'Reasons', render: (row: MatchListItem) => row.decision_reasons.join(', ') || 'N/A' },
            ]}
            rows={matchesQuery.data.items}
            getRowKey={(row) => row.id || `${row.source_track_uuid}-${row.candidate_track_uuid}`}
          />
          <Pagination
            page={matchesQuery.data.page}
            pageSize={matchesQuery.data.page_size}
            total={matchesQuery.data.total}
            hasNext={matchesQuery.data.has_next}
            onPageChange={(page) => updateFilter('page', String(page))}
          />
        </>
      )}
    </div>
  )
}
